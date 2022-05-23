#!/bin/bash

export ONNX_WORKSPACE=/tmp/lint/onnx-workspace

export ONNX_WORKSPACE_SCRIPT=$ONNX_WORKSPACE/work/bashrc

onnx--() {
    source $ONNX_WORKSPACE_SCRIPT
}

setup-lcg() {
    source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh
}

install-py-local() {
    pip install -t $ONNX_WORKSPACE/installed $*
}

setup-py-local() {
    export PATH=$ONNX_WORKSPACE/installed/bin:$PATH
    export PYTHONPATH=$ONNX_WORKSPACE/installed:$PYTHONPATH
}

onnx-model-co() { # checkout 
    git clone https://github.com/onnx/onnx.git
}

onnx-model-co-lfs() {
    local path
    for path in $*; do
        if [ ! -f "$path" ]; then
            echo "path $path does not exist" 
            continue
        fi
        git lfs pull --include="$path" --exclude=""
    done
}
