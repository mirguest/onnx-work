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
#include <random>
#include "core/session/onnxruntime_cxx_api.h"

std::string show(const std::vector<int64_t>& v) {
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int main() {
    // random 
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);


    const std::string model_dir = "/tmp/lint/onnx-workspace/models/vision/classification/vgg/model";
    const std::string model_path = model_dir + "/" + "vgg16-bn-7.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ENV");
    Ort::SessionOptions session_options;

    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // prepare the input
    auto num_input_nodes = session.GetInputCount();
    std::cout << "num_input_nodes: " << num_input_nodes << std::endl;

    std::vector<const char*> input_node_names;
    std::vector<std::vector<int64_t>> input_node_dims;

    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto name = session.GetInputName(i, allocator);
        input_node_names.push_back(name);

        Ort::TypeInfo type_info  = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto dims = tensor_info.GetShape();
        input_node_dims.push_back(dims);

        std::cout << "[" << i << "]"
                  << " name: " << name
                  << " ndims: " << dims.size()
                  << " dims: " << show(dims)
                  << std::endl;
    }

    // prepare the output
    std::vector<int64_t> output_node_dims;
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> output_node_names(num_output_nodes);
    for(std::size_t i = 0; i < num_output_nodes; i++) {
        char* output_name              = session.GetOutputName(i, allocator);
        output_node_names[i]           = output_name;
        Ort::TypeInfo type_info        = session.GetOutputTypeInfo(i);
        auto tensor_info               = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        output_node_dims               = tensor_info.GetShape();
        std::cout << "[" << i << "]"
                  << " name: " << output_name
                  << " ndims: " << output_node_dims.size()
                  << " dims: " << show(output_node_dims)
                  << std::endl;

    }

    // input data

    std::vector<float> inputs;
    // random generate input
    for (size_t i = 0; i < input_node_names.size(); ++i) {
        const auto& dims = input_node_dims[i];

        int size = 1;
        for (auto j: dims) {
            size *= j;
        }
        std::cout << "generating " << size << " elements. " << std::endl;
        for (size_t j = 0; j < size; ++j) {
            inputs.push_back(dis(gen));
        }
    }

    const auto& dims = input_node_dims[0];

    Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

    auto input_tensor = Ort::Value::CreateTensor(info, 
                                                 inputs.data(), 
                                                 inputs.size(), 
                                                 dims.data(), 
                                                 dims.size());
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));

    // run inference
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
                                      input_node_names.data(),
                                      input_tensors.data(),
                                      input_tensors.size(),
                                      output_node_names.data(), 
                                      output_node_names.size());

    return 0;
}
