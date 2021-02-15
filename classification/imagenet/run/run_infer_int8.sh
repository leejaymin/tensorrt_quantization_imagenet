#!/bin/bash
input=$1
trap "exit" INT

declare -a OUTPUT=(
"../engines/mobilenet.int8.engine"
"../engines/shufflenet.int8.engine"
"../engines/squeezenet.int8.engine"
"../engines/googlenet_v4_slim.int8.engine"
"../engines/resnet18.int8.engine"
"../engines/resnet50.int8.engine"
)
declare -a RGB_MODE=(
"True"
"False"
"False"
"False"
"True"
"False"
)
declare -a pre_func=(
"preprocess_imagenet"
"preprocess_imagenet"
"preprocess_imagenet_squeezenet"
"preprocess_imagenet_googlenet"
"preprocess_imagenet_resnet18"
"preprocess_imagenet"
)
declare -a dir_name=(
"/root/hdd/imagenet/imagenet2012_processed"
"/root/hdd/imagenet/imagenet2012_processed"
"/root/hdd/imagenet/imagenet2012_processed"
"/root/hdd/imagenet/val_processed_299"
"/root/hdd/imagenet/imagenet2012_processed"
"/root/hdd/imagenet/imagenet2012_processed"
)
declare -a labeloffset=("0" "0" "0" "1" "0" "0")

python3 ../infer_tensorrt_imagenet.py \
      -d ${dir_name[input]} \
      --batch_size 1 \
      --num_classes 5 \
      --preprocess_func=${pre_func[input]} \
      --rgb_mode=${RGB_MODE[input]} \
      --label_offset=${labeloffset[input]} \
      --labels="../labels/imagenet1k_labels.txt" \
      --engine ${OUTPUT[input]}