# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
### Error Handling ###
import os
import warnings
warnings.filterwarnings(action='ignore')
import logging
#import absl.logging
#logging.root.removeHandler(absl.logging._absl_handler)
logging.basicConfig(level=logging.WARNING)
#absl.logging._warn_preinit_stderr = False
#import sys
#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
######################
'''
from multiprocessing import Process
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import json
import torch
from schema import And, Optional

from nni.utils import OptimizeMode

from nni.compression.pytorch.compressor import Pruner
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.utils.num_param_counter import get_total_num_weights
from .constants_pruner import PRUNER_DICT

################### TVM build part addition ###############
from pruned_vgg_maxpool import VGG 
import _pickle as cPickle
import time
import torch.onnx
import onnxruntime
import tensorflow as tf

import socket
import sys 

import tvm
from tvm import relay, autotvm
import numpy as np
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm import rpc
from tvm.contrib import utils, ndk, graph_runtime as runtime
#from torchsummary import summary
from nni.compression.pytorch.utils.counter import count_flops_params

from nni.compression.pytorch import ModelSpeedup
import gc
###########################################################


_logger = logging.getLogger(__name__)


class NetAdaptPruner(Pruner):
    """
    A Pytorch implementation of NetAdapt compression algorithm.

    Parameters
    ----------
    model : pytorch model
        The model to be pruned.
    config_list : list
        Supported keys:
            - sparsity : The target overall sparsity.
            - op_types : The operation type to prune.
    short_term_fine_tuner : function
        function to short-term fine tune the masked model.
        This function should include `model` as the only parameter,
        and fine tune the model for a short term after each pruning iteration.
        Example::

            def short_term_fine_tuner(model, epoch=3):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                train_loader = ...
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                model.train()
                for _ in range(epoch):
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
    evaluator : function
        function to evaluate the masked model.
        This function should include `model` as the only parameter, and returns a scalar value.
        Example::

            def evaluator(model):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                val_loader = ...
                model.eval()
                correct = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        # get the index of the max log-probability
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / len(val_loader.dataset)
                return accuracy
    optimize_mode : str
        optimize mode, `maximize` or `minimize`, by default `maximize`.
    base_algo : str
        Base pruning algorithm. `level`, `l1`, `l2` or `fpgm`, by default `l1`. Given the sparsity distribution among the ops,
        the assigned `base_algo` is used to decide which filters/channels/weights to prune.
    sparsity_per_iteration : float
        sparsity to prune in each iteration.
    experiment_data_dir : str
        PATH to save experiment data,
        including the config_list generated for the base pruning algorithm and the performance of the pruned model.
    """

    def __init__(self, model, config_list, short_term_fine_tuner, evaluator, val_loader, num, dummy_input, criterion,
                 optimize_mode='maximize', base_algo='l1', sparsity_per_iteration=0.01, experiment_data_dir='./'):
        # models used for iterative pruning and evaluation
        self._model_to_prune = copy.deepcopy(model)
        self._base_algo = base_algo

        super().__init__(model, config_list)

        self._short_term_fine_tuner = short_term_fine_tuner
        self._evaluator = evaluator
        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters for NetAdapt algorithm
        self._sparsity_per_iteration = sparsity_per_iteration

        # overall pruning rate
        self._sparsity = config_list[0]['sparsity']

        # config_list
        self._config_list_generated = []

        self._experiment_data_dir = experiment_data_dir
        if not os.path.exists(self._experiment_data_dir):
            os.makedirs(self._experiment_data_dir)

        self._tmp_model_path = os.path.join(self._experiment_data_dir, 'tmp_model.pth')

        # addition
        self._num = num
        self._val_loader = val_loader
        self._criterion = criterion
        self._dummy_input = dummy_input
#        self._my_shape = my_shape

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list
            List on pruning configs
        """

        if self._base_algo == 'level':
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                Optional('op_types'): [str],
                Optional('op_names'): [str],
            }], model, _logger)
        elif self._base_algo in ['l1', 'l2', 'fpgm']:
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                'op_types': ['Conv2d'],
                Optional('op_names'): [str]
            }], model, _logger)

        schema.validate(config_list)

    def calc_mask(self, wrapper, **kwargs):
        return None

    def _update_config_list(self, config_list, op_name, sparsity):
        '''
        update sparsity of op_name in config_list
        '''
        config_list_updated = copy.deepcopy(config_list)
        if not op_name:
            return config_list_updated

        for idx, item in enumerate(config_list):
            if op_name in item['op_names']:
                config_list_updated[idx]['sparsity'] = sparsity
                return config_list_updated

        # if op_name is not in self._config_list_generated, create a new json item
        if self._base_algo in ['l1', 'l2', 'fpgm']:
            config_list_updated.append(
                {'sparsity': sparsity, 'op_types': ['Conv2d'], 'op_names': [op_name]})
        elif self._base_algo == 'level':
            config_list_updated.append(
                {'sparsity': sparsity, 'op_names': [op_name]})

        return config_list_updated

    def _get_op_num_weights_remained(self, op_name, module):
        '''
        Get the number of weights remained after channel pruning with current sparsity

        Returns
        -------
        int
            remained number of weights of the op
        '''

        # if op is wrapped by the pruner
        for wrapper in self.get_modules_wrapper():
            if wrapper.name == op_name:
                return wrapper.weight_mask.sum().item()

        # if op is not wrapped by the pruner
        return module.weight.data.numel()

    def _get_op_sparsity(self, op_name):
        for config in self._config_list_generated:
            if 'op_names' in config and op_name in config['op_names']:
                return config['sparsity']
        return 0

    def _calc_num_related_weights(self, op_name):
        '''
        Calculate total number weights of the op and the next op, applicable only for models without dependencies among ops

        Parameters
        ----------
        op_name : str

        Returns
        -------
        int
            total number of all the realted (current and the next) op weights
        '''
        num_weights = 0
        flag_found = False
        previous_name = None
        previous_module = None

        for name, module in self._model_to_prune.named_modules():
            if not flag_found and name != op_name and type(module).__name__ in ['Conv2d', 'Linear']:
                previous_name = name
                previous_module = module
            if not flag_found and name == op_name:
                _logger.debug("original module found: %s", name)
                num_weights = module.weight.data.numel()

                # consider related pruning in this op caused by previous op's pruning
                if previous_module:
                    sparsity_previous_op = self._get_op_sparsity(previous_name)
                    if sparsity_previous_op:
                        _logger.debug(
                            "decrease op's weights by %s due to previous op %s's pruning...", sparsity_previous_op, previous_name)
                        num_weights *= (1-sparsity_previous_op)

                flag_found = True
                continue
            if flag_found and type(module).__name__ in ['Conv2d', 'Linear']:
                _logger.debug("related module found: %s", name)
                # channel/filter pruning crossing is considered here, so only the num_weights after channel pruning is valuable
                num_weights += self._get_op_num_weights_remained(name, module)
                break

        _logger.debug("num related weights of op %s : %d", op_name, num_weights)

        return num_weights

    def _test3(self, model, input_name, ctx, text):
        test_loss = 0 
        correct = 0 
        total_time = 0 
        cases = 2 #20
        loc = 0 
        warm_up = 1 #10
        with torch.no_grad():
            for data, target in self._val_loader:
                loc = loc + 1 
                if loc % 5 == 0:
                    print(loc)
                if loc == cases + 1:
                    break
                output_arr = np.array([1,2])
                for i in range(len(target)):
                    model.set_input(input_name, np.expand_dims(data[i], 0)) 
                    t0 = time.time()
                    model.run()
                    t1 = time.time()
                    if loc > warm_up:
                        total_time += (t1 - t0)  
                    output = model.get_output(0)
                    output = output.asnumpy()
                    output = np.ravel(output, order='C')
                    if i == 0:
                        output_arr = output
                    else:
                        output_arr = np.append(output_arr, output, axis=0)
                output_arr = output_arr.reshape(len(target), 10) 
                output_arr = torch.from_numpy(output_arr)
                # sum up batch loss
                test_loss += self._criterion(output_arr, target).item()
                # get the index of the max log-probability
                pred = output_arr.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        real_cases = (cases - warm_up) * len(target)
#        test_loss /= real_cases
#        accuracy = correct / real_cases
        latency = (total_time*1000) / real_cases
#        fps2 = real_cases / total_time
        print('{} Latency: {:.6f}'.format(text, latency))
#        print('{} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Latency: {:.6f}'.format(
#                    text, test_loss, correct, real_cases, 100. * accuracy, latency))

        return latency

    def _tune_tasks(
        self,
        tasks,
        measure_option,
        tuner="xgb",
        n_trial=1000,
        early_stopping=None,
        log_filename="tuning.log",
        use_transfer_learning=True,
    ):
        # create tmp log file
        tmp_log_file = log_filename + ".tmp"
        if os.path.exists(tmp_log_file):
            os.remove(tmp_log_file)

        for i, tsk in enumerate(reversed(tasks)):
            prefix = "[Task %2d/%2d] " % (i+1, len(tasks))
            # create tuner
            if tuner == "xgb" or tuner == "xgb-rank":
                tuner_obj = XGBTuner(tsk, loss_type="rank")
            elif tuner == "ga":
                tuner_obj = GATuner(tsk, pop_size=100)
            elif tuner == "random":
                tuner_obj = RandomTuner(tsk)
            elif tuner == "gridsearch":
                tuner_obj = GridSearchTuner(tsk)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            if use_transfer_learning:
                if os.path.isfile(tmp_log_file):
                    tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

            # do tuning
            tsk_trial = min(n_trial, len(tsk.config_space))
            tuner_obj.tune(
                n_trial=tsk_trial,
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                    autotvm.callback.log_to_file(tmp_log_file),
                ],
            )

        # pick best records to a cache file
        autotvm.record.pick_best(tmp_log_file, log_filename)
        os.remove(tmp_log_file)

    def compress(self):
        """
        Compress the model.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        _logger.info('Starting NetAdapt Compression...')
        num = 1000
        from PIL import Image
        from tvm.contrib.download import download_testdata
        img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
        img_path = download_testdata(img_url, "cat.png", module="data")
        img = Image.open(img_path).resize((32, 32))
        from torchvision import transforms
        my_preprocess = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),   
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        )
        img = my_preprocess(img)
        img = np.expand_dims(img, 0)

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        arch = "arm64"
        target = "llvm -mtriple=%s-linux-android" % arch        
#        target = "opencl --device=mali"
#        target_host = "llvm -mtriple=arm64-linux-android"
#        my_shape = cPickle.load(open(os.path.join('/github/evta2/output', str(num), 'my_shape.p'),'rb'))
#        torch_model = VGG(my_shape=my_shape, depth=16).to(device)
#        torch_model.load_state_dict(torch.load(os.path.join('/github/evta2/output', str(num), 'model_trained.pth')))
#        torch_model.eval()
################# Autotune added
        network = "vgg-16"
        device_key = "android"
        log_file = "%s.%s.log" % (device_key, network)
        dtype = "float32"
        use_android = True

        tuning_option = {
            "log_filename": log_file,
            "tuner": "xgb",
            "n_trial": 100,
            "early_stopping": 50,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func="ndk" if use_android else "default"),
                runner=autotvm.RPCRunner(device_key, host="0.0.0.0", port=9190, number=5, timeout=10000)
            )
        }
################################
        self._model_to_prune.eval()
        input_shape = [1, 3, 32, 32]
        output_shape = [1, 10]
        input_data = torch.randn(input_shape).to(device)
        scripted_model = torch.jit.trace(self._model_to_prune, input_data).eval()
        input_name = "input0"
        shape_list = [(input_name, img.shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        
        tasks = autotvm.task.extract_from_program(
            mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"), relay.op.get("nn.dense"),)
        )
        self._tune_tasks(tasks, **tuning_option)

        tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
        tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
        key = "android"
        tracker = rpc.connect_tracker(tracker_host, tracker_port)
        remote = tracker.request(key, priority=0, session_timeout=0)
        ctx = remote.cpu(0)  # remote.cl(0)
        
        with autotvm.apply_history_best(log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod, target=target, params=params)
            tmp = utils.tempdir()
            lib_fname = tmp.relpath("net.so")
            lib.export_library(lib_fname, ndk.create_shared)
            remote.upload(lib_fname)
            rlib = remote.load_module("net.so")
            module = runtime.GraphModule(rlib["default"](ctx))

        current_latency = self._test3(module, input_name, ctx, "TVM_initial")
#        print('Relaunch app!')
#        time.sleep(60)
        
        pruning_iteration = 0
        delta_num_weights_per_iteration = \
            int(get_total_num_weights(self._model_to_prune, ['Conv2d', 'Linear']) * self._sparsity_per_iteration)
        print(delta_num_weights_per_iteration)
        init_resource_reduction_ratio = 0.025 # 0.05 
        resource_reduction_decay = 0.96 #0.98
        max_iter = 100

        budget = 0.5 * current_latency
        init_resource_reduction = init_resource_reduction_ratio * current_latency
        print('Budget: ' + str(budget) + ', Current latency: ' + str(current_latency))
        file_object = open('./record_tvm.txt', 'a')
        file_object.write('Budget: ' + str(budget) + ', Current latency: ' + str(current_latency) + '\n')
        file_object.close()
        current_accuracy = self._evaluator(self._model_to_prune)
#        improper_layer = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pass_target_latency = 0

        # stop condition
        while pruning_iteration < max_iter and current_latency > budget:
            _logger.info('Pruning iteration: %d', pruning_iteration)

            # calculate target sparsity of this iteration
#            target_sparsity = current_sparsity + self._sparsity_per_iteration
            if pass_target_latency == 1:
                target_latency = current_latency - init_resource_reduction * (
                        resource_reduction_decay ** (pruning_iteration - 1))
                pass_target_latency = 0
#            target_latency = current_latency - (init_resource_reduction * resource_reduction_decay)

            # Print the message
            print('=======================')
            print(('Process iteration {:>3}: current_accuracy = {:>8.3f}, '
                    'current_latency = {:>8.3f}, target_latency = {:>8.3f} \n').format(pruning_iteration, current_accuracy, current_latency, target_latency))            
            file_object = open('./record_tvm.txt', 'a')            
            file_object.write(('Process iteration {:>3}: current_accuracy = {:>8.3f}, '
                   'current_latency = {:>8.3f}, target_resource = {:>8.3f} \n').format(pruning_iteration, current_accuracy, current_latency, target_latency))
            file_object.close()

            # variable to store the info of the best layer found in this iteration
            best_op = {}
#            improper_idx = 0
            layer_idx = 1
            total_channel = 0

#                _logger.debug("op name : %s", wrapper.name)
#                _logger.debug("op weights : %d", wrapper.weight_mask.numel())
#                _logger.debug("op left weights : %d", wrapper.weight_mask.sum().item())
            for wrapper in self.get_modules_wrapper():

                current_op_sparsity = 1 - wrapper.weight_mask.sum().item() / wrapper.weight_mask.numel()

                if layer_idx > 7:
                    total_channel = 512
                elif layer_idx > 4:
                    total_channel = 256
                elif layer_idx > 2:
                    total_channel = 128
                else:
                    total_channel = 64
                # sparsity that this layer needs to prune to satisfy the requirement
                target_op_sparsity = current_op_sparsity + delta_num_weights_per_iteration / self._calc_num_related_weights(wrapper.name)
                temp_times = round(target_op_sparsity / (8 / total_channel))
                target_op_sparsity = temp_times * (8 / total_channel)
                layer_idx += 1
                print("target_op_sparsity: " + target_op_sparsity + '\n')

#                if improper_layer[improper_idx] == 1:
#                    print('Improper layer')
#                    file_object = open('./record_tvm.txt', 'a')
#                    file_object.write('Improper Layer: ' + wrapper.name + '\n')
#                    file_object.close()
#                    improper_idx += 1
#                    continue

                if target_op_sparsity >= 1:
                    _logger.info('Layer %s has no enough weights (remained) to prune', wrapper.name)
                    print('Improper layer')
                    file_object = open('./record_tvm.txt', 'a')
                    file_object.write('Improper Layer: ' + wrapper.name + '\n')
                    file_object.close()
#                    improper_idx += 1
                    continue


#                while True:
                config_list = self._update_config_list(self._config_list_generated, wrapper.name, target_op_sparsity)
                pruner = PRUNER_DICT[self._base_algo](copy.deepcopy(self._model_to_prune), config_list)
                model_masked = pruner.compress()

                # added 0: speed_up
                pruner.export_model('./model_masked.pth', './mask.pth')
                model = VGG(my_shape=self._my_shape, depth=16).to(device)
                model.load_state_dict(torch.load('./model_masked.pth'))
                masks_file = './mask.pth'
                m_speedup = ModelSpeedup(model, self._dummy_input, masks_file, device)
                m_speedup.speedup_model()
                # added 1: Autotune + TVM build
                model.eval()
                _, _, _ = count_flops_params(model, (1, 3, 32, 32))
                input_shape = [1, 3, 32, 32]
                output_shape = [1, 10]
                input_data = torch.randn(input_shape).to(device)
                scripted_model = torch.jit.trace(self._model_to_prune, input_data).eval()
                input_name = "input0"
                shape_list = [(input_name, img.shape)]
                mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

                tasks = autotvm.task.extract_from_program(
                    mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"), relay.op.get("nn.dense"),)
                )
                self._tune_tasks(tasks, **tuning_option)

                tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
                tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
                key = "android"
                tracker = rpc.connect_tracker(tracker_host, tracker_port)
                remote = tracker.request(key, priority=0, session_timeout=0)
                ctx = remote.cpu(0)   # remote.cl(0)

                with autotvm.apply_history_best(log_file):
                    with tvm.transform.PassContext(opt_level=3):
                        lib = relay.build_module.build(mod, target=target, params=params)
#                    def thread_func(module, ctx):
                    tmp = utils.tempdir()
                    lib_fname = tmp.relpath("net.so")
                    lib.export_library(lib_fname, ndk.create_shared)
                    temp_str = remote.upload(lib_fname)
                    rlib = remote.load_module("net.so")
                    module = runtime.GraphModule(rlib["default"](ctx))

#                    p = Process(target=thread_func, args=(module, ctx))
#                    p.start()
#                    p.join()
#                    print("threading finished")                    
                temp_latency = self._test3(module, input_name, ctx, "TVM")

                print('temp_latench: ' + str(temp_latency))
                print('Layer: ' + wrapper.name + ', target_latency: ' + str(target_latency) + ', temp_latency: ' + str(temp_latency) + ', target_op_sparsity: ' + str(target_op_sparsity))
                file_object = open('./record_tvm.txt', 'a')
                file_object.write('Layer: ' + wrapper.name + ', target_latency: ' + str(target_latency) + ', temp_latency: ' + str(temp_latency) + '\n')
                file_object.close()

#                    if temp_latency <= target_latency:
#                        break
#                    else:
#                        if target_op_sparsity < 0.5:
#                            target_op_sparsity += 0.1
#                        elif target_op_sparsity <= 0.90:
#                            target_op_sparsity += 0.05
#                        else:
#                            print('Improper layer: ' + wrapper.name)
#                            improper_layer[improper_idx] = 1
#                            file_object = open('./record_tvm.txt', 'a')
#                            file_object.write('Improper layer: ' + wrapper.name + '\n')
#                            file_object.close()
#                            break

#                # Short-term fine tune the pruned model
#                if improper_layer[improper_idx] == 0:
                self._short_term_fine_tuner(model_masked, epochs=2)
                performance = self._evaluator(model_masked)
                print('Layer: ' + wrapper.name + ', accuracy after short-term fine tuning: ' + str(performance))
                file_object = open('./record_tvm.txt', 'a')
                file_object.write('Layer: ' + wrapper.name + ', Accuracy: ' + str(performance) + '\n')
                file_object.close()

                if temp_latency <= target_latency and \
                    ( not best_op \
                    or (self._optimize_mode is OptimizeMode.Maximize and performance > best_op['performance']) \
                    or (self._optimize_mode is OptimizeMode.Minimize and performance < best_op['performance'])):
                    _logger.debug("updating best layer to %s...", wrapper.name)
                    pass_target_latency = 1
                    # find weight mask of this layer
                    for w in pruner.get_modules_wrapper():
                        if w.name == wrapper.name:
                            masks = {'weight_mask': w.weight_mask,
                                     'bias_mask': w.bias_mask}
                            break
                    best_op = {
                        'op_name': wrapper.name,
                        'sparsity': target_op_sparsity,
                        'latency': temp_latency,
                        'performance': performance,
                        'masks': masks
                    }

                    current_latency = temp_latency

                    # save model weights
                    pruner.export_model(self._tmp_model_path)

#                print('Relaunch app!')
#                time.sleep(60)
#                improper_idx += 1

#            if not best_op:
#                # decrease pruning step
#                self._sparsity_per_iteration *= 0.5
#                _logger.info("No more layers to prune, decrease pruning step to %s", self._sparsity_per_iteration)
#                pruning_iteration = max_iter
#                continue

            # Pick the best layer to prune, update iterative information
            # update config_list
            self._config_list_generated = self._update_config_list(
                self._config_list_generated, best_op['op_name'], best_op['sparsity'])

            # update weights parameters
            self._model_to_prune.load_state_dict(torch.load(self._tmp_model_path))
            print('Budget: ' + str(budget) + ', current_latency: ' + str(best_op['latency']))
            file_object = open('./record_tvm.txt', 'a')
            file_object.write('Budget: ' + str(budget) + ', current_latency: ' + str(best_op['latency']) + '\n')

            # update mask of the chosen op
            for wrapper in self.get_modules_wrapper():
                if wrapper.name == best_op['op_name']:
                    for k in best_op['masks']:
                        setattr(wrapper, k, best_op['masks'][k])
                    break

            file_object.write('Layer ' + best_op['op_name'] + ', selected with sparsity ' + str(best_op['sparsity']) + ', latency ' + str(best_op['latency']) + ', accuracy after pruning and short-term fine-tuning: ' + str(best_op['performance']) + '\n')
            file_object.close()
#            current_sparsity = target_sparsity
#            _logger.info('Pruning iteration %d finished, current sparsity: %s', pruning_iteration, current_sparsity)
            _logger.info('Layer %s seleted with sparsity %s, performance after pruning & short term fine-tuning : %s',
                         best_op['op_name'], best_op['sparsity'], best_op['performance'])
            pruning_iteration += 1

            self._final_performance = best_op['performance']

        # load weights parameters
        self.load_model_state_dict(torch.load(self._tmp_model_path))
        os.remove(self._tmp_model_path)

        _logger.info('----------Compression finished--------------')
        _logger.info('config_list generated: %s', self._config_list_generated)
        _logger.info("Performance after pruning: %s", self._final_performance)
        _logger.info("Masked sparsity: %.6f", current_sparsity)

        # save best config found and best performance
        with open(os.path.join(self._experiment_data_dir, 'search_result.json'), 'w') as jsonfile:
            json.dump({
                'performance': self._final_performance,
                'config_list': json.dumps(self._config_list_generated)
            }, jsonfile)

        _logger.info('search history and result saved to foler : %s', self._experiment_data_dir)

        return self.bound_model
