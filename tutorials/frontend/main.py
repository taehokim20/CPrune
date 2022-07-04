import argparse
import os
import json
import torch
import torch.utils.data
from torch import nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms
import torchvision

from models.cifar10.vgg import VGG
from models.cifar10.resnet import ResNet18
import torchvision.models as models
from c_pruner import CPruner
from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params

############### TVM build part addition ##############
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

def get_data(dataset, data_dir, batch_size, test_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    if dataset == 'cifar10':
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()
    elif dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                 transform=transforms.Compose([
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=test_batch_size, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()
    return train_loader, val_loader, criterion


def train(args, model, device, train_loader, criterion, optimizer, epoch, callback=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # callback should be inserted between loss.backward() and optimizer.step()
        if callback:
            callback()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            file_object = open('./train_epoch.txt', 'a')
            file_object.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            file_object.close()

# Top-1 and Top-5 accuracy test (ImageNet)
def test(model, device, criterion, val_loader):
    model.eval()
    total_len = len(val_loader.dataset)
    test_loss = 0
    correct = 0
    correct_5 = 0
    count = 0
    with torch.no_grad():
        for data, target in val_loader:
            count += 1
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            _, pred = output.topk(5, 1, True, True)
            temp_1 = pred.eq(target.view(1, -1).expand_as(pred))
            temp_5 = temp_1[:5].view(-1)
            correct_5 += temp_5.sum().item()
            if count % 5000 == 0 and count != total_len:
                print('Top-1: {}/{} ({:.4f}%), Top-5: {}/{} ({:.4f}%)'.format(correct, count, 100.*(correct/count), correct_5, count, 100.*(correct_5/count)))

    test_loss /= total_len
    accuracy = correct / total_len
    accuracy_5 = correct_5 / total_len

    print('Test set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.4f}%), Top-5 Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss, correct, total_len, 100. * accuracy, correct_5, total_len, 100. * accuracy_5))
    file_object = open('./train_epoch.txt', 'a')
    file_object.write('Test set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.4f}%), Top-5 Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, total_len, 100. * accuracy, correct_5, total_len, 100. * accuracy_5))
    file_object.close()

    return accuracy, accuracy_5

# Only top-1 accuracy test (CIFAR-10)
def test_top1(model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))
    file_object = open('./train_epoch.txt', 'a')
    file_object.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))
    file_object.close()

    return accuracy

def get_dummy_input(args, device):
    if args.dataset == 'cifar10':
        dummy_input = torch.randn([args.test_batch_size, 3, 32, 32]).to(device)
    elif args.dataset == 'imagenet':
        dummy_input = torch.randn([args.test_batch_size, 3, 224, 224]).to(device)
    return dummy_input


def get_input_size(dataset):
    if dataset == 'cifar10':
        input_size = (1, 3, 32, 32)
    elif dataset == 'imagenet':
        input_size = (1, 3, 224, 224)
    return input_size


def main(args):
    cpu_or_gpu = 1 #1: cpu, 2: gpu
    # prepare dataset
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, criterion = get_data(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)
    # ResNet18 for CIFAR-10
    if args.model == 'resnet18' and args.dataset == 'cifar10':
        model = ResNet18().to(device) #VGG(depth=16).to(device)
        model.load_state_dict(torch.load('./model_trained.pth'))
    # torchvision models for ImageNet
    elif args.model == 'resnet18':
        model = models.resnet18(pretrained=True).to(device)
    elif args.model == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=True).to(device)
    elif args.model == 'mnasnet1_0':
        model = models.mnasnet1_0(pretrained=True).to(device)

    acc_requirement = args.accuracy_requirement;
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)

    def short_term_trainer(model, optimizer=optimizer, epochs=1):
        train(args, model, device, train_loader, criterion, optimizer, epochs)

    def evaluator(model):
        return test(model, device, criterion, val_loader)

    def evaluator_top1(model):
        return test_top1(model, device, criterion, val_loader)
    # ImageNet
    if args.dataset == 'imagenet':
        #accuracy, accuracy_5 = evaluator(model)
        # ResNet-18
        accuracy = 0.69758
        accuracy_5 = 0.89078
        ## MnasNet1_0
        #accuracy = 0.73456
        #accuracy_5 = 0.91510
        print('Original model - Top-1 Accuracy: %s, Top-5 Accuracy: %s' %(accuracy, accuracy_5))
    # CIFAR-10
    elif args.dataset == 'cifar10':
        accuracy = evaluator_top1(model)
        print('Original model - Top-1 Accuracy: %s' %(accuracy))
    # module types to prune, only "Conv2d" supported for channel pruning
    if args.base_algo in ['l1', 'l2', 'fpgm']:
        op_types = ['Conv2d']
    elif args.base_algo == 'level':
        op_types = ['default']

    config_list = [{
        'sparsity': args.sparsity,
        'op_types': op_types
    }]
    dummy_input = get_dummy_input(args, device)
    input_size = get_input_size(args.dataset)    
    pruner = CPruner(model, config_list, short_term_trainer=short_term_trainer, evaluator=evaluator if args.dataset == 'imagenet' else evaluator_top1, val_loader=val_loader, dummy_input=dummy_input, criterion=criterion, base_algo=args.base_algo, experiment_data_dir=args.experiment_data_dir, cpu_or_gpu=cpu_or_gpu, input_size=input_size, dataset=args.dataset, acc_requirement=acc_requirement)
    # Pruner.compress() returns the masked model
    model = pruner.compress()

    # model speed up
    if args.speed_up:
        model.load_state_dict(torch.load('./tmp_model.pth'))
        masks_file = './tmp_mask.pth'
        m_speedup = ModelSpeedup(model, dummy_input, masks_file, device)
        m_speedup.speedup_model()
    
    ################ Long-term training ################
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    best_acc = 0
    if args.dataset == 'imagenet':
        best_acc_5 = 0
    for epoch in range(args.fine_tune_epochs): # imagenet: 20, cifar10: 100
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        if args.dataset == 'imagenet':
            acc, acc_5 = evaluator(model)
            if acc_5 > best_acc_5:
                best_acc_5 = acc_5
                torch.save(model.state_dict(), os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth'))
            if acc > best_acc:
                best_acc = acc
        elif args.dataset == 'cifar10':
            acc = evaluator_top1(model)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth'))

    if args.dataset == 'imagenet':
        print('Evaluation result (Long-term): %s %s' %(best_acc, best_acc_5))
    elif args.dataset == 'cifar10':
        print('Evaluation result (Long-term): %s' %(best_acc))
    ####################################################    
    ################ Long-term tuning and compile ################
    if os.path.isfile('./model_fine_tuned.pth'):
        model.load_state_dict(torch.load(os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth')))
    arch = "arm64"
    target = "llvm -mtriple=%s-linux-android" % arch
    device_key = "android"
    log_file = "%s.log" % (device_key)
    dtype = "float32"
    use_android = True
    at_least_trials = 10
    num_per_round = 60
    model.eval()
    _, _, temp_results = count_flops_params(model, get_input_size(args.dataset))
    input_shape = get_input_size(args.dataset)
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
    #####################################
    tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
    tracker_port = int(os.environ["TVM_TRACKER_PORT"])
    ########### Extract search tasks ###########
    print("Extract tasks...")
    if cpu_or_gpu == 1:
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    else:
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target="opencl -device=mali", target_host=target)
    tune_trials = 10 * (at_least_trials + num_per_round) * len(tasks)
    print("tune_trials: " + str(tune_trials))
    ########### Tuning ###########
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=tune_trials,
        builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_android else "default"),
        runner=auto_scheduler.RPCRunner(device_key, host=tracker_host, port=tracker_port, timeout=20, number=10, repeat=2,),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        num_measures_per_round = num_per_round,
    )
    tuner.tune(tune_option)
    ########### Compile ###########
    print("Compile")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            if cpu_or_gpu == 1:
                lib = relay.build(mod, target=target, params=params)
            else:
                lib = relay.build(mod, params=params, target="opencl -device=mali", target_host=target)

    tmp = utils.tempdir()
    lib_fname = tmp.relpath("net.so")
    lib.export_library(lib_fname, ndk.create_shared)
    remote = auto_scheduler.utils.request_remote(device_key, tracker_host, tracker_port, timeout=200)
    remote.upload(lib_fname)
    rlib = remote.load_module("net.so")
    if cpu_or_gpu == 1:
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
    ##############################################################
    
 
if __name__ == '__main__':
    def str2bool(s):
        if isinstance(s, bool):
            return s
        if s.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if s.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='CTuner arguments')

    # dataset and model
    parser.add_argument('--accuracy-requirement', type=float, default=0.85,
                        help='the minimum accuracy requirement')
    parser.add_argument('--dataset', type=str, default= 'imagenet',
                        help='dataset to use, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data_fast/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model to use, resnet18, mobilenetv2, mnasnet1_0')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, #64
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--fine-tune', type=str2bool, default=True,
                        help='whether to fine-tune the pruned model')
    parser.add_argument('--fine-tune-epochs', type=int, default=20,
                        help='epochs to fine tune')
    parser.add_argument('--experiment-data-dir', type=str, default='./',
                        help='For saving experiment data')

    # pruner
    parser.add_argument('--base-algo', type=str, default='l1',
                        help='base pruning algorithm. level, l1, l2, or fpgm')
    parser.add_argument('--sparsity', type=float, default=0.1,
                        help='target overall target sparsity')

    # others
    parser.add_argument('--log-interval', type=int, default=1000, #200,
                        help='how many batches to wait before logging training status')
    # speed-up
    parser.add_argument('--speed-up', type=str2bool, default=True,
                        help='Whether to speed-up the pruned model')

    args = parser.parse_args()

    if not os.path.exists(args.experiment_data_dir):
        os.makedirs(args.experiment_data_dir)

    main(args)
