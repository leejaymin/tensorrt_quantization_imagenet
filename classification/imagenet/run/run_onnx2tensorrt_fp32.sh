#!/bin/bash
input=$1
trap "exit" INT

declare -a ONNX_MODEL=(
"/root/hdd/models/mobilenet_v2/onnxzoo_mobilenet_v210.onnx"
"/root/hdd/models/shufflenet/model.onnx"
"/root/hdd/models/squeezenet/model.onnx"
"/root/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx"
"/root/hdd/models/mxnet_exported_resnet18.onnx"
"/root/hdd/models/resnet50/model.onnx"
)

declare -a OUTPUT=(
"../engines/mobilenet.fp32.engine"
"../engines/shufflenet.fp32.engine"
"../engines/squeezenet.fp32.engine"
"../engines/googlenet_v4_slim.fp32.engine"
"../engines/resnet18.fp32.engine"
"../engines/resnet50.fp32.engine"
)

python ../onnx_to_tensorrt.py \
    --onnx=${ONNX_MODEL[input]} \
    --explicit-batch \
    -o ${OUTPUT[input]}

