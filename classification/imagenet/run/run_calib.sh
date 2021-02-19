#!/bin/bash
input=$1
trap "exit" INT

# Path to dataset to use for calibration.
#   **Not necessary if you already have a calibration cache from a previous run.
CALIBRATION_DATA="/root/hdd/imagenet/train_processed"
#CALIBRATION_DATA="/root/hdd/imagenet/train_processed_299"

# Truncate calibration images to a random sample of this amount if more are found.
#   **Not necessary if you already have a calibration cache from a previous run.
MAX_CALIBRATION_SIZE=512

declare -a ONNX_MODEL=(
"/root/hdd/models/mobilenet_v2/onnxzoo_mobilenet_v210.onnx"
"/root/hdd/models/shufflenet/model.onnx"
"/root/hdd/models/squeezenet/model.onnx"
"/root/hdd/models/googlenet_v4_slim/googlenet_v4_slim.onnx"
"/root/hdd/models/mxnet_exported_resnet18.onnx"
"/root/hdd/models/resnet50/model.onnx"
)
declare -a CACHE_FILENAME=(
"../caches/mobilenet.cache"
"../caches/shufflenet.cache"
"../caches/squeezenet.cache"
"../caches/googlenet_v4_slim.cache"
"../caches/resnet18.cache"
"../caches/resnet50.cache"
)
declare -a pre_func=(
"preprocess_imagenet"
"preprocess_imagenet"
"preprocess_imagenet_squeezenet"
"preprocess_imagenet_googlenet"
"preprocess_imagenet_resnet18"
"preprocess_imagenet"
)
declare -a OUTPUT=(
"../engines/mobilenet.int8.engine"
"../engines/shufflenet.int8.engine"
"../engines/squeezenet.int8.engine"
"../engines/googlenet_v4_slim.int8.engine"
"../engines/resnet18.int8.engine"
"../engines/resnet50.int8.engine"
)

python3 ../onnx_to_tensorrt.py --fp16 --int8 -v \
        --max_calibration_size=${MAX_CALIBRATION_SIZE} \
        --calibration-data=${CALIBRATION_DATA} \
        --calibration-cache=${CACHE_FILENAME[input]} \
        --preprocess_func=${pre_func[input]} \
        --explicit-batch \
        --onnx ${ONNX_MODEL[input]} -o ${OUTPUT[input]}

