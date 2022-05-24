// Description:
//   Reading .pb files at the onnx model zoo
//
//   $ g++ example_pb.cpp -DONNX_ML -DONNX_NAMESPACE=onnx -I/tmp/lint/onnx-workspace/installed/include $(pkg-config --cflags --libs protobuf) -L/tmp/lint/onnx-workspace/installed/lib64 -lonnx -lonnx_proto
//                        


#include <iostream>
#include <fstream>

#include "onnx/defs/parser.h"
#include "onnx/proto_utils.h"

int main() {
    const std::string model_dir = "/tmp/lint/onnx-workspace/models/vision/classification/vgg/model";
    const std::string input_file = model_dir + "/vgg16-bn/test_data_set_0/input_0.pb";
    std::cout << "input file: " << input_file << std::endl;

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

    std::cout << "Has raw data: " << tensor.has_raw_data() << std::endl;

    std::cout << "The TensorProto data_type: " << tensor.data_type() << std::endl;

    std::cout << "The TensorProto dims_size: " << tensor.dims_size() << std::endl;

    std::cout << "The dims: ";
    for (int i = 0; i < tensor.dims_size(); ++i) {
        std::cout << tensor.dims(i) << " ";
    }
    std::cout << std::endl;

    // load the raw data
    if (tensor.has_raw_data()) {
        const auto& rawdata = tensor.raw_data();

        std::cout << "rawdata size: " << rawdata.size() << std::endl;
    }

    // in this example, float_data is empty
    std::cout << "The TensorProto float_data_size: " << tensor.float_data_size() << std::endl;
    if (tensor.float_data_size()) {
        const auto& data = tensor.float_data();
        std::cout << "type of data: " << typeid(data).name() << std::endl;
        // type of data: N6google8protobuf13RepeatedFieldIfEE
        std::cout << "size of data: " << data.size() << std::endl;

        const auto rawdata = data.data();
        std::cout << "type of rawdata: " << typeid(rawdata).name() << std::endl;
        // type of rawdata: PKf
    }

    return 0;
}
