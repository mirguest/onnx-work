// Description:
//   Build C++ application with ONNX Runtime
//
//   $ g++ example.cpp -L/cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/lib -lonnxruntime
//
// Reference:
// - G4 11.0 Par04
//   - https://github.com/Geant4/geant4/blob/master/examples/extended/parameterisations/Par04/include/Par04OnnxInference.hh
// - https://stackoverflow.com/questions/65379070/how-to-use-onnx-model-in-c-code-on-linux
// - https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx
// - /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/lib/libonnxruntime.so

#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include "core/session/onnxruntime_cxx_api.h"

std::string show(const std::vector<int64_t>& v) {
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int main() {
    const std::string model_dir = "/tmp/lint/onnx-workspace/models/vision/classification/vgg/model";
    const std::string model_path = model_dir + "/" + "vgg16-bn-7.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ENV");
    Ort::SessionOptions session_options;

    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    auto num_input_nodes = session.GetInputCount();
    std::cout << "num_input_nodes: " << num_input_nodes << std::endl;

    std::vector<std::string> input_node_names;

    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto name = session.GetInputName(i, allocator);
        Ort::TypeInfo type_info  = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto dims = tensor_info.GetShape();

        std::cout << "[" << i << "]"
                  << " name: " << name
                  << " ndims: " << dims.size()
                  << " dims: " << show(dims)
                  << std::endl;

    }

    // prepare the input

    // prepare the output


    // run inference


    return 0;
}
