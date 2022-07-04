from multiprocessing import Process
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import json
import torch

from nni.compression.pytorch.compressor import Pruner
from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT

################### TVM build part addition ###############
from models.cifar10.resnet import ResNet18, ResNet50
import torchvision.models as models
import time
import sys 

import tvm
from tvm import relay, auto_scheduler
import numpy as np
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner
from tvm import rpc
from tvm.contrib import utils, ndk, graph_runtime as runtime
from tvm.contrib import graph_executor
from nni.compression.pytorch.utils.counter import count_flops_params

from nni.compression.pytorch import ModelSpeedup
from torch.optim.lr_scheduler import MultiStepLR
###########################################################

class CPruner(Pruner):
    '''
    Pruning the pre-trained model by utilizing measured latency from executable tuning
    
    Parameters
    ----------
    model : pytorch model
        The model to be pruned.
    config_list : list
        Supported keys:
            - sparsity : The target overall sparsity.
            - op_types : The operation type to prune.
    short_term_trainer : function
        function to short-term train the masked model
    evaluator : function
        function to evaluate the masked model
    '''
    def __init__(self, model, config_list, short_term_trainer, evaluator, val_loader, dummy_input, criterion, base_algo='l1', experiment_data_dir='./', cpu_or_gpu=1, input_size=(1, 3, 224, 224), dataset='imagenet', acc_requirement=0.85):
        # models used for iterative pruning and evaluation
        self._model_to_prune = copy.deepcopy(model)
        self._original_model = copy.deepcopy(model)
        self._base_algo = base_algo
        self._cpu_or_gpu = cpu_or_gpu

        super().__init__(model, config_list)

        self._short_term_trainer = short_term_trainer
        self._evaluator = evaluator

        # config_list
        self._config_list_generated = []

        self._experiment_data_dir = experiment_data_dir
        if not os.path.exists(self._experiment_data_dir):
            os.makedirs(self._experiment_data_dir)

        self._tmp_model_path = './tmp_model.pth'

        # addition
        self._val_loader = val_loader
        self._criterion = criterion
        self._dummy_input = dummy_input
        self._input_size = input_size
        self._dataset = dataset
        self._acc_requirement = acc_requirement

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

    def compress(self):
        """
        Compress the model.

        Return
        -------
        torch.nn.Module : the final pruned model
        """
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        arch = "arm64"
        target = "llvm -mtriple=%s-linux-android" % arch        
        
        device_key = "android"
        log_file = "%s.log" % (device_key)
        dtype = "float32"
        use_android = True
        self._model_to_prune.eval()
        _, _, temp_results = count_flops_params(self._model_to_prune, self._input_size)
        conv2d_num = 0
        others_num = 0
        downsample_subgraphs = []
        temp_results_len = len(temp_results)
        for idx in range(temp_results_len):
            if 'downsample' in temp_results[idx].get('name'):
                downsample_subgraphs.append(idx)
            elif 'shortcut' in temp_results[idx].get('name'):
                downsample_subgraphs.append(idx)
            if temp_results[idx].get('module_type') == 'Conv2d':
                conv2d_num+=1
            else:
                others_num+=1
        conv2d_subgraph_chs = [-1 for i in range(conv2d_num)]
        temp_idx = 0
        for idx in range(temp_results_len):
            if temp_results[idx].get('module_type') == 'Conv2d':
                conv2d_subgraph_chs[temp_idx] = temp_results[idx].get('weight_shape')[0]
                temp_idx += 1

        ##################### subgraph_task connection #######################
        pos = []
        last_idx = conv2d_num - 1
        list_filled = [0 for i in range(conv2d_num)]
        for idx in range(conv2d_num):
            n = conv2d_num - 1 - idx
            if list_filled[n] == 1:
                continue
            elif 'downsample' in temp_results[n].get('name'):
                continue
            elif 'shortcut' in temp_results[n].get('name'):
                continue
            else:
                pos.append(n)
                list_filled[n] = 1
            split_name = temp_results[n].get('name').split('.')
            for i in range(conv2d_num):
                if i == n: break
                temp_split = temp_results[i].get('name').split('.')
                if split_name[0] == temp_split[0] and \
                   split_name[len(split_name)-1] == temp_split[len(temp_split)-1] and \
                   temp_results[n].get('weight_shape') == temp_results[i].get('weight_shape') and \
                   temp_results[n].get('flops') == temp_results[i].get('flops') and \
                   temp_results[n].get('params') == temp_results[i].get('params'):
                    pos.append(i)
                    list_filled[i] = 1

        pos = pos + downsample_subgraphs

        input_shape = self._input_size
        input_data = torch.randn(input_shape).to(device)
        ######################################################################
        scripted_model = torch.jit.trace(self._model_to_prune, input_data).eval()
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        ########### NCHW -> NHWC ############
        desired_layouts = {'nn.conv2d': ['NHWC', 'default'], 'nn.dense': ['NHWC', 'default']}
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts),
                                        relay.transform.InferType(),
                                        relay.transform.FoldConstant(),
                                        relay.transform.DeadCodeElimination()])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        #####################################
        tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
        tracker_port = int(os.environ["TVM_TRACKER_PORT"])
        #################### Extract search tasks ###################
        print("Extract tasks...")
        if self._cpu_or_gpu == 1:
            tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        else:
            tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target="opencl -device=mali", target_host=target)

        subgraph_tasks = [-1 for i in range(conv2d_num)]
        task_times = [-1 for i in range(conv2d_num)]
        pos_idx = 0
        downsample_idx = 0
        for idx, task in enumerate(tasks):
            if idx < others_num:
                continue
            if len(task.workload_key) < 80:
                continue
            for i in range(task_weights[idx]):
                subgraph_tasks[pos[pos_idx]] = idx
                pos_idx += 1

        max_iter = 100
        pass_target_latency = 0
        alpha = 0.995  # target_accuracy = alpha * prev_best_accuracy
        beta = 0.99  # target_latency = beta * current_best_latency
        init_short_acc = 0
        performance = 0
        intermediate = 0
        pruning_times = [0.0 for i in range(conv2d_num)]
        real_pruning_times = [0.0 for i in range(conv2d_num)]
        at_least_trials = 10
        num_per_round = 60
        tune_trials = (at_least_trials + num_per_round) * len(tasks) #(conv2d_num + others_num)        
        minimum_acc_requirement = self._acc_requirement
        
        if intermediate == 1:
            #### Need to be changed ####
            task_times = [0]
            task_times_rank = np.array([1,3,5,2,4,15,16,9,6,13,19,10,11,14,8,18,7,0,17,12])
            current_latency = 28.1697
            current_accuracy = 0.9430
            total_estimated_latency = 86.0512
            init_short_acc = 0.9437     # the initial accuracy (one time revision)
            initial_latency = 84.6750    # the initial latency w/o pruning (one time revision)
            #### Fixed val ####
            target_latency = current_latency * alpha
            pruning_iteration = 0
            budget = 0.1 * initial_latency
        else:
            #################### Tuning #####################
            print("Begin tuning...")
            tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=tune_trials,
                builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_android else "default"),
                runner=auto_scheduler.RPCRunner(device_key, host=tracker_host, port=tracker_port, timeout=20, number=10, repeat=2,),
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	        verbose=1,
                #early_stopping=300,
                num_measures_per_round = num_per_round,
            )
            tuner.tune(tune_option)        
            total_estimated_latency = 0
            for i in range(conv2d_num):
                task_times[i] = tuner.best_costs[subgraph_tasks[i]] * task_weights[subgraph_tasks[i]]
                total_estimated_latency += tuner.best_costs[subgraph_tasks[i]] * 1000
            task_times_rank = np.argsort(task_times)
            task_times_rank = np.flip(task_times_rank)
            file_object = open('./record_tvm.txt', 'a')
            file_object.write('=============== task_times ===============\n')
            file_object.write(str(task_times))
            file_object.write('\n')
            file_object.write(str(task_times_rank))
            file_object.write('\n')
            file_object.write(str(np.argsort(task_times_rank) + 1))
            file_object.write('\n\n')
            file_object.close()
            #################### Compile ####################
            print("Compile...")
            with auto_scheduler.ApplyHistoryBest(log_file):
                with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                    if self._cpu_or_gpu == 1:
                        lib = relay.build_module.build(mod, params=params, target=target)
                    else:
                        lib = relay.build(mod, params=params, target="opencl -device=mali", target_host=target)
            
            tmp = utils.tempdir()
            lib_fname = tmp.relpath("net.so")
            lib.export_library(lib_fname, ndk.create_shared)
            remote = auto_scheduler.utils.request_remote(device_key, tracker_host, tracker_port, timeout=200)
            remote.upload(lib_fname)
            rlib = remote.load_module("net.so")

            # Create graph executor
            if self._cpu_or_gpu == 1:
                ctx = remote.cpu()
            else:
                ctx = remote.cl(0)
            module = graph_executor.GraphModule(rlib["default"](ctx))

            data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
            module.set_input(input_name, data_tvm)
            ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=2)
            prof_res = np.array(ftimer().results) * 1e3
            current_latency = np.mean(prof_res)
            print('ftimer_latency: ' + str(current_latency))
            time.sleep(250)
            #################################################        
            pruning_iteration = 1
            budget = 0.1 * current_latency
            print('Current latency: {:>8.4f}, Total estimated latency: {:>8.4f}'.format(current_latency, total_estimated_latency))
            file_object = open('./record_tvm.txt', 'a')
            file_object.write('Budget: {:>8.4f}, Current latency: {:>8.4f}, Total estimated latency: {:>8.4f}\n'.format(budget, current_latency, total_estimated_latency))
            file_object.close()
            if self._dataset == 'cifar10':
                current_accuracy = self._evaluator(self._model_to_prune)                
            elif self._dataset == 'imagenet':
                _, current_accuracy = self._evaluator(self._model_to_prune)
            target_latency = current_latency * beta

        # stop condition
        while pruning_iteration < max_iter and current_latency > budget:
            # calculate target sparsity of this iteration
            if pass_target_latency == 1:
                target_latency = current_latency * beta
                pass_target_latency = 0

            # Print the message
            print('=======================')
            print(('Process iteration {:>3}: current_accuracy = {:>8.4f}, '
                    'current_latency = {:>8.4f}, target_latency = {:>8.4f}, total_estimated_latency = {:>8.4f}, tune_trials = {:4d} \n').format(pruning_iteration, current_accuracy, current_latency, target_latency, total_estimated_latency, tune_trials))
            file_object = open('./record_tvm.txt', 'a')            
            file_object.write(('Process iteration {:>3}: current_accuracy = {:>8.4f}, '
                   'current_latency = {:>8.4f}, target_resource = {:>8.4f}, total_estimated_latency = {:>8.4f}, tune_trials = {:4d} \n').format(pruning_iteration, current_accuracy, current_latency, target_latency, total_estimated_latency, tune_trials))
            file_object.write('Current pruning_times: ' + str(pruning_times) + '\n')
            file_object.write('Real pruning_times: ' + str(real_pruning_times) + '\n')
            file_object.close()

            # variable to store the info of the best subgraph found in this iteration
            best_op = {}
            
            ########################### Pre-pruning (if it is necessary) ##########################
            if pruning_iteration == 0:
                real_pruning_times = [0]
                subgraph_idx = 0
                for wrapper in self.get_modules_wrapper():
                    if real_pruning_times[subgraph_idx] > 0:
                        target_op_sparsity = real_pruning_times[subgraph_idx]
                        self._config_list_generated = self._update_config_list(
                            self._config_list_generated, wrapper.name, target_op_sparsity)
                        pruner = PRUNER_DICT[self._base_algo](copy.deepcopy(self._model_to_prune), self._config_list_generated, dependency_aware=True, dummy_input=self._dummy_input)
                        model_masked = pruner.compress()
                        masks = {}
                        for w in pruner.get_modules_wrapper():
                            if w.name == wrapper.name:
                                masks = {'weight_mask': w.weight_mask,
                                         'bias_mask': w.bias_mask}
                                break
                        for k in masks:
                            setattr(wrapper, k, masks[k])
                    subgraph_idx += 1
                pruning_times = [0]
                pruning_iteration = 12
            ######################################################################
            
            cnt = 0
            while cnt < len(task_times_rank):
                init_cnt = cnt
                overlap_num = 1
                while True:
                    if cnt + 1 == len(task_times_rank):
                        break
                    if task_times[task_times_rank[cnt]] == task_times[task_times_rank[cnt+1]]:
                        overlap_num += 1
                        cnt += 1
                    else:
                        break
                cnt += 1
                for overlap_cnt in task_times_rank[init_cnt: init_cnt + overlap_num]:
                    pruning_times[overlap_cnt] += float(tuner.prune_num[subgraph_tasks[overlap_cnt]]) * float(1/conv2d_subgraph_chs[overlap_cnt])
                target_op_sparsity = pruning_times[task_times_rank[init_cnt]]
                ch_num = int(conv2d_subgraph_chs[task_times_rank[init_cnt]] * (1 - target_op_sparsity))
                if target_op_sparsity > 0.8:
                    print('Improper Subgraph')
                    wrapper = self.get_modules_wrapper()[task_times_rank[init_cnt]]
                    file_object = open('./record_tvm.txt', 'a')      
                    file_object.write('Improper Subgraph: ' + wrapper.name + ', Total: ' + str(overlap_num) + ' subgraphs\n')
                    file_object.close()
                    continue

                config_list = copy.deepcopy(self._config_list_generated)
                for wrapper_idx in task_times_rank[init_cnt: init_cnt + overlap_num]:
                    wrapper = self.get_modules_wrapper()[wrapper_idx]
                    config_list = self._update_config_list(config_list, wrapper.name, target_op_sparsity)

                wrapper = self.get_modules_wrapper()[task_times_rank[init_cnt]]
                print('Subgraph: ' + wrapper.name + ', overlap_num: ' + str(overlap_num) + ', ch_num: ' + str(ch_num))
                file_object = open('./record_tvm.txt', 'a')
                file_object.write('Subgraph: ' + wrapper.name + ', overlap_num: ' + str(overlap_num) + ', ch_num: ' + str(ch_num) + '\n')
                file_object.write('Temp_pruning_times:' + str(pruning_times) + '\n')
                file_object.close()

                pruner = PRUNER_DICT[self._base_algo](copy.deepcopy(self._model_to_prune), config_list, dependency_aware=True, dummy_input=self._dummy_input)
                model_masked = pruner.compress()

                # added 0: speed_up
                pruner.export_model('./model_masked.pth', './mask.pth')
                model = copy.deepcopy(self._original_model)
                model.load_state_dict(torch.load('./model_masked.pth'))
                masks_file = './mask.pth'
                m_speedup = ModelSpeedup(model, self._dummy_input, masks_file, device)
                m_speedup.speedup_model()
                # added 1: Autotune + TVM build
                model.eval()
                _, _, temp_results = count_flops_params(model, self._input_size)
                input_shape = self._input_size
                input_data = torch.randn(input_shape).to(device)
                scripted_model = torch.jit.trace(model, input_data).eval()
                input_name = "input0"
                shape_list = [(input_name, input_shape)]
                mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
                ########### NCHW -> NHWC ############
                desired_layouts = {'nn.conv2d': ['NHWC', 'default'], 'nn.dense': ['NHWC', 'default']}
                seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                      relay.transform.ConvertLayout(desired_layouts)])
                with tvm.transform.PassContext(opt_level=3):
                    mod = seq(mod)
                #################### subgraph_task connection ####################
                pos = []
                last_idx = conv2d_num - 1
                list_filled = [0 for i in range(conv2d_num)]
                for idx in range(conv2d_num):
                    n = conv2d_num - 1 - idx
                    if list_filled[n] == 1:
                        continue
                    elif 'downsample' in temp_results[n].get('name'):
                        continue
                    elif 'shortcut' in temp_results[n].get('name'):
                        continue
                    else:
                        pos.append(n)
                        list_filled[n] = 1
                    split_name = temp_results[n].get('name').split('.')
                    for i in range(conv2d_num):
                        if i == n: break
                        temp_split = temp_results[i].get('name').split('.')
                        if split_name[0] == temp_split[0] and \
                           split_name[len(split_name)-1] == temp_split[len(temp_split)-1] and \
                           temp_results[n].get('weight_shape') == temp_results[i].get('weight_shape') and \
                           temp_results[n].get('flops') == temp_results[i].get('flops') and \
                           temp_results[n].get('params') == temp_results[i].get('params'):
                            pos.append(i)
                            list_filled[i] = 1

                pos = pos + downsample_subgraphs             
                #################### Extract search tasks ###################
                print("Extract tasks...")
                if self._cpu_or_gpu == 1:
                    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
                else:
                    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target="opencl -device=mali", target_host=target)
                subgraph_tasks_temp = [-1 for i in range(conv2d_num)]
                task_times_temp = [-1 for i in range(conv2d_num)]
                pos_idx = 0
                for idx, task in enumerate(tasks):
                    if idx < others_num:
                        continue
                    if len(task.workload_key) < 80:
                        continue
                    for i in range(task_weights[idx]):
                        subgraph_tasks_temp[pos[pos_idx]] = idx
                        pos_idx += 1
                tune_trials = (at_least_trials + num_per_round) * len(tasks) #(conv2d_num + others_num)
                #################### Tuning #####################
                print("Begin tuning...")
                tuner = auto_scheduler.TaskScheduler(tasks, task_weights, target_execution_time=target_latency)
                tune_option = auto_scheduler.TuningOptions(
                    num_measure_trials=tune_trials,
                    builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_android else "default"),
                    runner=auto_scheduler.RPCRunner(device_key, host=tracker_host, port=tracker_port, timeout=20, number=10, repeat=2,),
                    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                    num_measures_per_round = num_per_round,
                )
                tuner.tune(tune_option)
                total_estimated_latency = 0
                for i in range(conv2d_num):
                    task_times_temp[i] = tuner.best_costs[subgraph_tasks_temp[i]] * task_weights[subgraph_tasks_temp[i]]
                    total_estimated_latency += tuner.best_costs[subgraph_tasks_temp[i]] * 1000
                task_times_rank_temp = np.argsort(task_times_temp)
                task_times_rank_temp = np.flip(task_times_rank_temp)
                #################### Compile ####################
                print("Compile...")
                with auto_scheduler.ApplyHistoryBest(log_file):
                    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                        if self._cpu_or_gpu == 1:
                            lib = relay.build(mod, target=target, params=params)
                        else:
                            lib = relay.build(mod, params=params, target="opencl -device=mali", target_host=target)
         
                tmp = utils.tempdir()
                lib_fname = tmp.relpath("net.so")
                lib.export_library(lib_fname, ndk.create_shared)
                remote = auto_scheduler.utils.request_remote(device_key, tracker_host, tracker_port, timeout=250)
                remote.upload(lib_fname)
                rlib = remote.load_module("net.so")
                if self._cpu_or_gpu == 1:
                    ctx = remote.cpu()
                else:
                    ctx = remote.cl()
                module = graph_executor.GraphModule(rlib["default"](ctx))

                data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
                module.set_input(input_name, data_tvm)
                ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=2)
                prof_res = np.array(ftimer().results) * 1e3                
                temp_latency = np.mean(prof_res)
                print('ftimer_latency: ' + str(temp_latency))
                #################################################
                print('Subgraph: {}, Temp latency: {:>8.4f}, Total estimated latency: {:>8.4f}, Channel: {:4d}, Next trials: {:4d}'.format(wrapper.name, temp_latency, total_estimated_latency, ch_num, tune_trials))
                file_object = open('./record_tvm.txt', 'a')
                file_object.write('Subgraph: {}, Temp latency: {:>8.4f}, Total estimated latency: {:>8.4f}, Channel: {:4d}, Next trials: {:4d}\n'.format(wrapper.name, temp_latency, total_estimated_latency, ch_num, tune_trials))
                file_object.close()
                ################# Added part to prune the slow subgraph quickly ##################
                if temp_latency > target_latency:
                    file_object = open('./record_tvm.txt', 'a')
                    file_object.write('Higher than target latency! Pruning_ratio of Subgraph {} increases one time more!\n'.format(wrapper.name))
                    file_object.close()
                ###############################################################################

                if temp_latency <= target_latency:
                    file_object = open('./train_epoch.txt', 'a')
                    file_object.write('Subgraph: {}, Temp latency: {:>8.4f}, Channel: {:4d}\n'.format(wrapper.name, temp_latency, ch_num))
                    file_object.close()
                    # Short-term fine tune the pruned model
                    optimizer = torch.optim.SGD(model_masked.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)                    
                    best_acc = 0
                    short_num = 5
                    if self._dataset == 'imagenet':
                        best_acc_5 = 0
                        short_num = 1
                    for epoch in range(short_num):
                        self._short_term_trainer(model_masked, optimizer, epochs=epoch)
                        if self._dataset == 'imagenet':
                            acc, acc_5 = self._evaluator(model_masked)
                            if acc_5 > best_acc_5:
                                best_acc_5 = acc_5
                            if acc > best_acc:
                                best_acc = acc
                        elif self._dataset == 'cifar10':
                            acc = self._evaluator(model_masked)
                            if acc > best_acc:
                                best_acc = acc
                    if self._dataset == 'cifar10':
                        print('Subgraph: {}, Short_tune - Top-1 Accuracy: {:>8.5f}'.format(wrapper.name, best_acc))
                        file_object = open('./record_tvm.txt', 'a')
                        file_object.write('Subgraph: {}, Top-1 Accuracy: {:>8.5f} \n'.format(wrapper.name, best_acc))
                        file_object.close()
                    elif self._dataset == 'imagenet':
                        print('Subgraph: {}, Short_tune - Top-1 Accuracy: {:>8.5f}, Top-5 Accuracy: {:>8.5f}'.format(wrapper.name, best_acc, best_acc_5))
                        file_object = open('./record_tvm.txt', 'a')
                        file_object.write('Subgraph: {}, Top-1 Accuracy: {:>8.5f}, Top-5 Accuracy: {:>8.5f}'.format(wrapper.name, best_acc, best_acc_5))
                        file_object.close()
                    ################ Added part to avoid excessive accuracy decrement ###############
                    temp_acc = best_acc_5 if self._dataset == 'imagenet' else best_acc
                    if temp_acc < alpha * current_accuracy: 
                        file_object = open('./record_tvm.txt', 'a')
                        file_object.write('Too low short-term accuracy! Improper subgraph: {}\n'.format(wrapper.name))
                        file_object.close()
                        for wrapper_idx in task_times_rank[init_cnt: init_cnt + overlap_num]:
                            pruning_times[wrapper_idx] = 1
                        continue
                    #################################################################################

                    for wrapper_idx in task_times_rank[init_cnt: init_cnt + overlap_num]:
                        real_pruning_times[wrapper_idx] = pruning_times[wrapper_idx]
                    pass_target_latency = 1
                    # find weight mask of this subgraph
                    for w in pruner.get_modules_wrapper():
                        if w.name == wrapper.name:
                            masks = {'weight_mask': w.weight_mask,
                                     'bias_mask': w.bias_mask}
                            break
                    best_op = {
                        'op_name': wrapper.name,
                        'sparsity': target_op_sparsity,
                        'ch_num': ch_num,
                        'latency': temp_latency,
                        'performance': temp_acc,
                        'masks': masks
                    }

                    current_latency = temp_latency
                    prev_task_times_rank = task_times_rank

                    # save model weights
                    pruner.export_model(self._tmp_model_path, './tmp_mask.pth')
                    subgraph_tasks = subgraph_tasks_temp
                    task_times = task_times_temp
                    task_times_rank = task_times_rank_temp
                    file_object = open('./record_tvm.txt', 'a')
                    file_object.write('=============== task_times ===============\n')
                    file_object.write(str(task_times))
                    file_object.write('\n')
                    file_object.write(str(task_times_rank))
                    file_object.write('\n')
                    file_object.write(str(np.argsort(task_times_rank) + 1))
                    file_object.write('\n\n')
                    file_object.close()
                    break
                else:
                    time.sleep(250)

            # Check the minimum accuracy requirement
            if alpha * best_op['performance'] < minimum_acc_requirement:
                break

            if pass_target_latency == 1:
                for wrapper_idx in prev_task_times_rank[init_cnt: init_cnt + overlap_num]:
                    wrapper = self.get_modules_wrapper()[wrapper_idx]
                    self._config_list_generated = self._update_config_list(
                        self._config_list_generated, wrapper.name, target_op_sparsity)
                    for w in pruner.get_modules_wrapper():
                        if w.name == wrapper.name:
                            masks = {'weight_mask': w.weight_mask, 'bias_mask': w.bias_mask}
                            break
                    for k in masks:
                        setattr(wrapper, k, masks[k])

                # update weights parameters
                self._model_to_prune.load_state_dict(torch.load(self._tmp_model_path))
                print('Budget: {:>8.4f}, Current latency: {:>8.4f}'.format(budget, best_op['latency']))
                file_object = open('./record_tvm.txt', 'a')
                file_object.write('Budget: {:>8.4f}, Current latency: {:>8.4f} \n'.format(budget, best_op['latency']))

                current_accuracy = temp_acc
                #########################
                file_object.write('Subgraph {} selected with {:4d} channels, latency {:>8.4f}, accuracy {:>8.4f} \n'.format(best_op['op_name'], best_op['ch_num'], best_op['latency'], best_op['performance']))
                file_object.close()
            pruning_iteration += 1

        # load weights parameters
        self.load_model_state_dict(torch.load(self._tmp_model_path))

        return self._model_to_prune
