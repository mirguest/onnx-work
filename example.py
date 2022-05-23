#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx
from onnx import numpy_helper

import os
import os.path
import glob

model_dir = "/tmp/lint/onnx-workspace/models/vision/classification/vgg/model"
model_path = os.path.join(model_dir, 'vgg16-bn-7.onnx')
print("model is %s"%model_path)

print("start loading model ...")
# model = onnx.load(model_path)
session = onnxruntime.InferenceSession(model_path, None)

input_name = session.get_inputs()[0].name  
print('Input Name:', input_name)


print("loading model done.")


test_data_dir = os.path.join(model_dir, 'vgg16-bn', 'test_data_set_0')


# Load inputs
inputs = []
inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))

print(inputs_num)

for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))

print("Load input done. ")


# Load reference outputs
ref_outputs = []
ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
for i in range(ref_outputs_num):
    output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    ref_outputs.append(numpy_helper.to_array(tensor))

print("Load output done. ")


### Running
test_data_num = 1
outputs = [session.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]


print('Predicted {} results.'.format(len(outputs)))

# Compare the results with reference outputs up to 4 decimal places
for ref_o, o in zip(ref_outputs, outputs):
    np.testing.assert_almost_equal(ref_o, o, 4)
    
print('ONNX Runtime outputs are similar to reference outputs!')
