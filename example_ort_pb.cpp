// Description:
//   Build C++ application with ONNX Runtime
//
//   $ g++ example_ort_pb.cpp -DONNX_ML -DONNX_NAMESPACE=onnx -I/tmp/lint/onnx-workspace/installed/include $(pkg-config --cflags --libs protobuf) -L/tmp/lint/onnx-workspace/installed/lib64 -lonnx -lonnx_proto $(pkg-config --cflags --libs libonnxruntime)
//
// Reference:
// - G4 11.0 Par04
//   - https://github.com/Geant4/geant4/blob/master/examples/extended/parameterisations/Par04/include/Par04OnnxInference.hh
// - https://stackoverflow.com/questions/65379070/how-to-use-onnx-model-in-c-code-on-linux
// - https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx
// - /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/lib/libonnxruntime.so

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include "onnx/defs/parser.h"
#include "onnx/proto_utils.h"
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
    const std::string input_file = model_dir + "/vgg16-bn/test_data_set_0/input_0.pb";
    const std::string output_file = model_dir + "/vgg16-bn/test_data_set_0/output_0.pb";

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

    std::ifstream ss(input_file, std::ios::binary);

    ss.seekg(0, std::ios::end);
    auto size = ss.tellg();
    ss.seekg(0, std::ios::beg);

    char* buffer = new char[size];
    ss.read(buffer, size);
    ss.close();

    std::cout << "file size: " << size << std::endl;

    ::onnx::TensorProto tensor;

    ::onnx::ParseProtoFromBytes(&tensor, buffer, size);

    const auto& rawdata = tensor.raw_data();

    std::cout << "rawdata size: " << rawdata.size() << std::endl;

    // try to read the buffer as float
    char* p = const_cast<char*>(&rawdata[0]);

    float* v = reinterpret_cast<float*>(p);
    int nelems = rawdata.size() / sizeof(float);

    const auto& dims = input_node_dims[0];

    Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

    auto input_tensor = Ort::Value::CreateTensor(info, 
                                                 v, 
                                                 nelems, 
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

    // ref output
    std::ifstream ss_ref(output_file, std::ios::binary);

    ss_ref.seekg(0, std::ios::end);
    auto size_ref = ss_ref.tellg();
    ss_ref.seekg(0, std::ios::beg);

    char* buffer_ref = new char[size_ref];
    ss_ref.read(buffer_ref, size);
    ss_ref.close();

    std::cout << "ref file size: " << size_ref << std::endl;

    ::onnx::TensorProto tensor_ref;

    ::onnx::ParseProtoFromBytes(&tensor_ref, buffer_ref, size_ref);

    const auto& rawdata_ref = tensor_ref.raw_data();

    std::cout << "ref rawdata size: " << rawdata_ref.size() << std::endl;

    // try to read the buffer as float
    char* p_ref = const_cast<char*>(&rawdata_ref[0]);

    float* v_ref = reinterpret_cast<float*>(p_ref);
    int nelems_ref = rawdata_ref.size() / sizeof(float);

    std::cout << "ref rawdata nelems: " << nelems_ref << std::endl;


    // compare
    const auto& output_tensor = output_tensors[0];
    const float* v_output = output_tensor.GetTensorData<float>();

    for (int i = 0; i < 10; ++i) {
        std::cout << "[" << i << "] "
                  << v_ref[i] << " " << v_output[i]
                  << std::endl;
    }

    return 0;
}
