# CPrune: Compiler-Informed Model Pruning for Efficient Target-Aware DNN Execution

Our source code is based on an open deep learning compiler stack Apache TVM (https://github.com/apache/tvm) and Microsoft nni (https://github.com/microsoft/nni).

# How to set up
## Host PC side
### Install nvidia-container-runtime
1. Add package repository \
      curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add - \
      distribution=$(. /etc/os-release; echo $ID$VERSION_ID) \
      curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \ <br/>
      sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list \
      sudo apt-get update
2. Package installation \
      sudo apt-get install -y nvidia-container-runtime
3. Installation check \
      which nvidia-container-runtime-hook
### Build and run
4. Check /docker/install/ubuntu_install_python.sh to fit with 'python3.6' <br>
5. docker build -t tvm3.demo_android -f docker/Dockerfile.demo_android ./docker
6. docker run --pid=host -h tvm3 -v $PWD:/workspace -w /workspace -p 9192:9192 --name tvm3 -it --gpus all tvm3.demo_android bash

## Docker side
7. Check if the GPU driver works properly \
      nvidia-smi
### Anaconda install
8. Download the Anaconda installation script \
      wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh <br>
9. Run the script to start the installation process \
      bash /tmp/Anaconda3-2020.02-Linux-x86_64.sh \
      source ~/.bashrc
### Conda environment
10. conda create -n nni ptyhon=3.6 <br>
11. conda activate nni <br>
12. conda install -c anaconda cudnn <br>
13. conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
### Build the TVM
14. mkdir build <br>
    cd build <br>
    cmake -DUSE_LLVM=llvm-config-8 -DUSE_RPC=ON -DUSE_VULKAN=ON -DUSE_GRAPH_EXECUTOR=ON .. <br>
    make -j10
### Install Android TVM RPC - Install Gradle
15. sudo apt install curl zip vim <br>
    curl -s "https://get.sdkman.io" | bash <br>
    source "$HOME/.sdkman/bin/sdkman-init.sh" <br>
    sdk install gradle 6.8.3
### Install TVM4J - Java Frontend for TVM Runtime
16. cd /workspace <br>
    make jvmpkg <br>
    pip3 install decorator <br>
    (Optional) sh tests/scripts/task_java_unittest.sh <br>
    make jvminstall
### ~/.bashrc
17. echo 'export PYTHONPATH=/workspace/python:/workspace/vta/python:${PYTHONPATH}' >> ~/.bashrc <br>
    echo 'export ANDROID_HOME=/opt/android-sdk-linux' >> ~/.bashrc <br>
    echo 'export TVM_NDK_CC=/opt/android-toolchain-arm64/bin/aarch64-linux-android-g++' >> ~/.bashrc <br>
    echo 'export TF_CPP_MIN_LOG_LEVEL=1' >> ~.bashrc <br>
    sudo apt-get install libjemalloc1 <br>
    echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.1' >> ~/.bashrc <br>
    source ~/.bashrc <br>
    conda activate nni
### Create a standalone toolchain
18. cd /opt/android-sdk-linux/ndk/21.3.6528147/build/tools/ <br>
    ./make-standalone-toolchain.sh --platform=android-28 --use-llvm --arch=arm64 --install-dir=/opt/android-toolchain-arm64
### tvmrpc-release.apk for CPU
19-1. cd /workspace/apps/android_rpc/app/src/main/jni/ <br>
      vim ./config.mk # ADD_C_INCLUDES = /opt/adrenosdk-linux-5_0/Development/Inc (depending on the phone)
### tvmrpc-release.apk for GPU
19-2. Get libOpenCL.so file from your phone to Host PC <br>
	    adb pull /system/vendor/lib64/libOpenCL.so ./ <br>
      Put the libOpenCL.so to /workspace/apps/android_rpc/app/src/main/jni/ <br>
      mv config.mk cpu_config.mk <br>
      mv gpu_config.mk config.mk
### Build APK (to create an apk file)
20. cd /workspace/apps/android_rpc <br>
    gradle clean build <br>
    ./dev_tools/gen_keystore.sh # generate a signature <br>
    ./dev_tools/sign_apk.sh # get the signed apk file <br>
    Upload app/build/outputs/apk/release/tvmrpc-release.apk file to the Android device and install it
### Additional stuff
21. pip3 install nni colorama tornado json-tricks schema scipy PrettyTable psutil xgboost cloudpickle absl-py tensorboard tensorflow pytest
### Basic docker setup
22. exit <br>
    docker start tvm3 <br>
    docker exec -it tvm3 bash <br>
    conda activate nni
### RPC tracker    
23. python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9192
### RPC connection check
24. (new terminal) python3 -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9192


# How to execute
In tutorials/frontend, there are two core files main.py and c_pruner.py. <br>
You can select an input model and type the accuracy requirement in main.py, and run the file.
If you want to look at the CPrune algorithm code, please look at c_pruner.py.
