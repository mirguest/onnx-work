#!/bin/bash

export ONNX_WORKSPACE=/tmp/$USER/onnx-workspace

export ONNX_WORKSPACE_SCRIPT=$ONNX_WORKSPACE/work/bashrc

onnx-cd() {
    cd $ONNX_WORKSPACE
}

onnx-() {
    setup-lcg
    setup-py-local
}

onnx--() {
    source $ONNX_WORKSPACE_SCRIPT
}

setup-lcg() {
    # LCG 101
    source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh

    # protobuf 3 (onnx needs a newer protobuf)
    export PATH=/cvmfs/sft.cern.ch/lcg/releases/protobuf/3.18.1-4cc72/x86_64-centos7-gcc11-opt/bin:$PATH
    export CMAKE_PREFIX_PATH=/cvmfs/sft.cern.ch/lcg/releases/protobuf/3.18.1-4cc72/x86_64-centos7-gcc11-opt/bin:$CMAKE_PREFIX_PATH
    export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/protobuf/3.18.1-4cc72/x86_64-centos7-gcc11-opt/lib:$LD_LIBRARY_PATH
}

install-py-local() {
    pip install -t $ONNX_WORKSPACE/installed $*
}

install-py-local-() {
    install-py-local numpy protobuf==3.16.0 onnx
    install-py-local onnxruntime

}

setup-py-local() {
    export PATH=$ONNX_WORKSPACE/installed/bin:$PATH
    export PYTHONPATH=$ONNX_WORKSPACE/installed:$PYTHONPATH
}

##############################################################################
# ONNX package built from source code
##############################################################################

onnx-onnx-co() { # checkout
    pushd $ONNX_WORKSPACE
    git clone --recursive https://github.com/onnx/onnx.git
    # or using:
    #   git submodule update --init --recursive
    popd
}

onnx-onnx-build() { # build
    local blddir=onnx-build
    pushd $ONNX_WORKSPACE
    
    [ -d "$blddir" ] || mkdir $blddir

    cd $blddir

    cmake ../onnx -DCMAKE_INSTALL_PREFIX=$ONNX_WORKSPACE/installed \
                  -DONNX_USE_PROTOBUF_SHARED_LIBS=ON

    cmake --build .
    
    popd
}

##############################################################################
# ONNX model
##############################################################################

onnx-model-co() { # checkout 
    pushd $ONNX_WORKSPACE
    git clone https://github.com/onnx/models.git
    popd
}

onnx-model-co-lfs() {
    pushd $ONNX_WORKSPACE/models
    local path
    for path in $*; do
        if [ ! -f "$path" ]; then
            echo "path $path does not exist" 
            continue
        fi
        git lfs pull --include="$path" --exclude=""
    done
    popd
}

onnx-model-co-lfs-() {
    onnx-model-co-lfs vision/classification/vgg/model/vgg16-bn-7.onnx
    onnx-model-co-lfs vision/classification/vgg/model/vgg16-bn-7.tar.gz
}

