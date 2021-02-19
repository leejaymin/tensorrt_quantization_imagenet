# ImageNet Models: ONNX -> TensorRT

This directory contains some helper scripts for creating TensorRT engines from
various ONNX classification models based on Imagenet data using the Python API. 

These scripts were last tested using the 
[NGC TensorRT Container Version 19.10-py3](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt).
You can see the corresponding framework versions for this container [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-container-release-notes/rel_19-10.html#rel_19-10).

## Quickstart

### 1. Start TensorRT Container with current directory mounted.

```bash
docker run -it --runtime=nvidia -v ${PWD}:/mnt --workdir=/mnt nvcr.io/nvidia/tensorrt:19.10-py3
```

### 2. Download Resnet50 ONNX model from [ONNX Model Zoo](https://github.com/onnx/models/tree/master/vision/classification).

```bash
wget https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz
tar -xvzf resnet50.tar.gz
```

### 3. Convert ONNX model to TensorRT

See `./onnx_to_tensorrt.py -h` for full list of command line arguments.

Also see `trtexec` if you're only interested in FP32/FP16 and not INT8 engines.

**FP32**
```bash
# OR trtexec --onnx=resnet50/model.onnx --saveEngine=resnet50.fp32.engine
./onnx_to_tensorrt.py --onnx resnet50/model.onnx -o resnet50.fp32.engine
```

**FP16**
```bash
# OR trtexec --onnx=resnet50/model.onnx --fp16 --saveEngine=resnet50.fp16.engine
./onnx_to_tensorrt.py --onnx resnet50/model.onnx -o resnet50.fp16.engine --fp16
```

**INT8**

For simplicity, we can use an existing calibration cache from [caches/resnet50.cache](caches/resnet50.cache):
```bash
./onnx_to_tensorrt.py --onnx resnet50/model.onnx -o resnet50.int8.engine --fp16 --int8 \
                      --calibration-cache="caches/resnet50.cache"
```

See the [INT8 Calibration](#int8-calibration) section below for details on calibration
using your own model or different data, where you don't have an existing calibration cache
or want to create a new one.


### 4. Infer on a sample image to quickly verify the engine.

See `./infer_tensorrt_imagenet.py -h` for full list of command line arguments.

```bash
python infer_tensorrt_imagenet.py -f test_images/mug.jpg \
                                  --batch_size 1 \
                                  --num_classes 3 \
                                  --preprocess_func=preprocess_imagenet \
                                  --engine resnet50.fp16.engine

#    Input image: test_images/mug.jpg
#        Prediction: coffee mug                     Probability: 0.83
#        Prediction: cup                            Probability: 0.16
#        Prediction: espresso                       Probability: 0.00
```

> **NOTE**: If the "Probability" for a prediction is > 1.0, this probably just means
> that there wasn't a `Softmax` layer in the original model, and I decided not to handle
> that for simplicity.


## INT8 Calibration

See [ImagenetCalibrator.py](ImagenetCalibrator.py) for a reference implementation
of TensorRT's [IInt8EntropyCalibrator2](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Int8/EntropyCalibrator2.html).

This class can be tweaked to work for other kinds of models, inputs, etc.

In the [Quickstart](#quickstart) section above, we made use of a pre-existing cache,
[caches/resnet50.cache](caches/resnet50.cache), to save time for the sake of an example.

However, to calibrate using different data or a different model, you can do so with the `--calibration-data` argument.

* This requires that you've mounted a dataset, such as Imagenet, to use for calibration.
    * Add something like `-v /imagenet:/imagenet` to your Docker command in Step (1) 
      to mount a dataset found locally at `/imagenet`.
* You can specify your own `preprocess_func` by defining it inside of `processing.py` and
  passing the function name as a `--preprocess_func` command-line argument
    * By default, `preprocess_imagenet` is used.
    * For `InceptionV1` for example, you can pass `--preprocess_func=preprocess_inception`
      instead of `preprocess_imagenet`. See [processing.py](processing.py).

```bash
# Path to dataset to use for calibration. 
#   **Not necessary if you already have a calibration cache from a previous run.
CALIBRATION_DATA="/imagenet"

# Truncate calibration images to a random sample of this amount if more are found.
#   **Not necessary if you already have a calibration cache from a previous run.
MAX_CALIBRATION_SIZE=512

# Calibration cache to be used instead of calibration data if it already exists,
# or the cache will be created from the calibration data if it doesn't exist.
CACHE_FILENAME="caches/custom.cache"

# Any function name defined in `processing.py`
PREPROCESS_FUNC="preprocess_imagenet"

# Path to ONNX model
ONNX_MODEL="resnet50/model.onnx"

# Path to write TensorRT engine to
OUTPUT="resnet50.int8.engine"

# Creates an int8 engine from your ONNX model, creating ${CACHE_FILENAME} based
# on your ${CALIBRATION_DATA}, unless ${CACHE_FILENAME} already exists, then
# it will use simply use that instead.
python3 onnx_to_tensorrt.py --fp16 --int8 -v \
        --max_calibration_size=${MAX_CALIBRATION_SIZE} \
        --calibration-data=${CALIBRATION_DATA} \
        --calibration-cache=${CACHE_FILENAME} \
        --preprocess_func=${PREPROCESS_FUNC} \
        --onnx ${ONNX_MODEL} -o ${OUTPUT}

```

### Pre-processing

In order to calibrate your model correctly, you should `pre-process` your data the same way
that you would during inference. You can pass in a `preprocess_func` to the constructor
of `ImagenetCalibrator(..., preprocess_func=<function_name>, ...)`,  where `<function_name>`
is a string, corresponding to the name of a pre-processing function defined inside of
`processing.py`. You can add your own pre-processing functions to `processing.py` and pass
the function name into the constructor accordingly.


## ONNX Models

### ONNX Model Zoo

To name a few models taken from the [ONNX Model Zoo](https://github.com/onnx/models/tree/master/vision/classification)
that should work pretty well with these scripts:
* [ResNet50](https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz)
* [MobileNetV2](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz)
* [InceptionV1](https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v1.tar.gz)
* [VGG16](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.tar.gz)

> **NOTE**: For creating FP32/FP16 engines, **it will likely be much simpler to just use `trtexec --onnx=model.onnx ...`**.
> These `python` scripts are mostly just convenient for doing `INT8` calibration, because `trtexec` currently just creates
> `INT8` engines for the sake of benchmarking, and doesn't preserve the accuracy of the model. The FP32/FP16 options just
> come along for free.

### TensorFlow Models

To play with these scripts using a TensorFlow model, you'll first need to convert the model to ONNX.

One of the best tools for doing that currently is [tf2onnx](https://github.com/onnx/tensorflow-onnx). 
Please see their documentation for more details.

Additionally, if you don't want to go from TF -> ONNX -> TRT, you can also try the built-in TF-TRT library
that comes with TensorFlow: https://github.com/tensorflow/tensorrt

### PyTorch Models

To play with these scripts using a PyTorch model, you'll first need to convert the model to ONNX.

PyTorch has built-in capabilities for exporting models to ONNX format. Please see their documentation for more details: 
https://pytorch.org/docs/stable/onnx.html

Similarly to TF-TRT, there is an ongoing effort for PyTorch here called `torch2trt`: https://github.com/NVIDIA-AI-IOT/torch2trt
