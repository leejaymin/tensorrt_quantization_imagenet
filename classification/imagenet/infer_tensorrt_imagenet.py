#!/usr/bin/env python3

# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import argparse
import PIL.Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # To automatically manage CUDA context creation and cleanup
import logging
import time

def load_normalized_test_case(test_images, pagelocked_buffer, preprocess_func):
    # Expected input dimensions
    #C, H, W = (3, 224, 224)
    # Normalize the images, concatenate them and copy to pagelocked memory.
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    data = np.asarray([preprocess_func(PIL.Image.open(img)) for img in test_images]).flatten()
    np.copyto(pagelocked_buffer, data)

# def load_normalized_test_case_299(test_images, pagelocked_buffer, preprocess_func):
#     # Expected input dimensions
#     #C, H, W = (3, 299, 299)
#     # Normalize the images, concatenate them and copy to pagelocked memory.
#     pil_logger = logging.getLogger('PIL')
#     pil_logger.setLevel(logging.INFO)
#     data = np.asarray([preprocess_func(PIL.Image.open(img)) for img in test_images]).flatten()
#     np.copyto(pagelocked_buffer, data)

class HostDeviceMem(object):
    r""" Simple helper data class that's a little nicer to use than a 2-tuple.
    """
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine: trt.ICudaEngine, batch_size: int):
    #print('Allocating buffers ...')

    inputs = []
    outputs = []
    dbindings = []

    stream = cuda.Stream()

    for binding in engine:
        size = batch_size * trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        dbindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, dbindings, stream


def infer(engine, preprocess_func, batch_size=8, input_images=[], labels=[], num_classes=3):

    # Allocate buffers and create a CUDA stream.
    inputs, outputs, dbindings, stream = allocate_buffers(engine, batch_size)

    # Contexts are used to perform inference.
    with engine.create_execution_context() as context:
        test_images = np.random.choice(input_images, size=batch_size)
        load_normalized_test_case(test_images, inputs[0].host, preprocess_func)
        #load_normalized_test_case_299(test_images, inputs[0].host, preprocess_func)

        inp = inputs[0]
        # Transfer input data to the GPU.
        cuda.memcpy_htod(inp.device, inp.host)

        # Run inference.
        context.execute(batch_size, dbindings)

        out = outputs[0]
        # Transfer predictions back to host from GPU
        cuda.memcpy_dtoh(out.host, out.device)
        out_np = np.array(out.host)

        # Split 1-D output of length N*labels into 2-D array of (N, labels)
        batch_outs = np.array(np.split(out_np, batch_size))

    return batch_outs
    #return topk_indices




def get_inputs(filename=None, directory=None, allowed_extensions=(".jpeg", ".jpg", ".png")):
    filenames = []
    if filename:
        filenames.append(filename)
    if directory:
        dir_files = [path for path in glob.iglob(os.path.join(directory, "**"), recursive=True)
                     if os.path.isfile(path) and path.lower().endswith(allowed_extensions)]
        filenames.extend(dir_files)

    if len(filenames) <= 0:
        raise ValueError("ERROR: No valid inputs given.")

    return filenames

def get_sorted_img_subdirs(validation_images_dir):
    img_dir_paths = []
    for img_dir in os.listdir(validation_images_dir):
        dir_path = os.path.join(validation_images_dir, img_dir)
        if os.path.isdir(dir_path):
            img_dir_paths.append(img_dir)
    img_dir_paths.sort()

    return img_dir_paths

def get_curr_img_paths(img_paths, img_index, batch_size):
    curr_img_paths = []

    for batch_idx in range(batch_size):
        img_path = img_paths[img_index + batch_idx]
        curr_img_paths.append(img_path)

    return curr_img_paths

# @returns two lists of the same length found in directory
# @param validation_images_dir; the first list contains paths to all images
# found, and the second list contains the corresponding labels of the image.
def get_img_paths_and_labels(validation_images_dir):
    img_subdirs = get_sorted_img_subdirs(validation_images_dir)

    # Create lists holding paths to each image to be classified and the label
    # for that image.
    img_paths = []
    img_labels = []
    curr_label_idx = 0
    for img_subdir in img_subdirs:
        img_subdir_path = os.path.join(validation_images_dir, img_subdir)
        for img in os.listdir(img_subdir_path):
            full_img_path = os.path.join(img_subdir_path, img)
            if os.path.isfile(full_img_path):
                img_paths.append(full_img_path)
                img_labels.append(curr_label_idx)
        curr_label_idx = curr_label_idx + 1
    return img_paths, img_labels

def print_topk_accuracy(total_image_count, top1_count, top5_count):
    top1_accuracy = float(top1_count) / float(total_image_count)
    top5_accuracy = float(top5_count) / float(total_image_count)
    top1_accuracy = top1_accuracy * 100
    top5_accuracy = top5_accuracy * 100
    print("\tTop-1 accuracy: " + "{0:.2f}".format(top1_accuracy))
    print("\tTop-5 accuracy: " + "{0:.2f}".format(top5_accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on TensorRT engines for Imagenet-based Classification models.')
    parser.add_argument('--engine', type=str, required=True,
                        help='Path to TensorRT engine {resnet50, vgg16, inception_v1, mobilenetv2-1.0...}')
    parser.add_argument('-f', '--file', default=None, type=str,
                        help="Path to input image.")
    parser.add_argument('-d', '--directory', default=None, type=str,
                        help="Path to directory of input images.")
    parser.add_argument("-l", "--labels", type=str, default=os.path.join("labels", "imagenet1k_labels.txt"),
                        help="Path to file containing model prediction labels.")
    parser.add_argument('-n', '--num_classes', default=3, type=int,
                        help="Top-K predictions to output.")
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        help="Number of inputs to send in parallel (up to max batch size of engine).")
    parser.add_argument("-p", "--preprocess_func", type=str, default=None,
                        help="Name of function defined in 'processing.py' to use for pre-processing calibration data.")
    parser.add_argument("-RGB", "--rgb_mode", type=bool, default=False,
                        help="RGB")
    parser.add_argument("-lo", "--label_offset", type=int, default=0,
                        help="label offset")
    parser.add_argument("-v", "--verbose", type=bool, default=False,
                        help="Print label and confidence")
    args = parser.parse_args()

    #input_images = get_inputs(args.file, args.directory)
    with open(args.labels, "r") as f:
        labels = np.array(f.read().splitlines())

    # Choose pre-processing function for inference inputs
    import processing
    if args.preprocess_func is not None:
        preprocess_func = getattr(processing, args.preprocess_func)
    else:
        preprocess_func = processing.preprocess_imagenet

    #img_paths, img_labels = get_img_paths_and_labels("/root/hdd/imagenet/imagenet2012_processed")
    #img_paths, img_labels = get_img_paths_and_labels("/root/hdd/imagenet/val_processed_299")

    #img_paths, img_labels = get_img_paths_and_labels("/root/hdd/imagenet/small_imagenet_299") # for debug
    #img_paths, img_labels = get_img_paths_and_labels("/root/hdd/imagenet/small_imagenet") # for debug
    img_paths, img_labels = get_img_paths_and_labels(args.directory) # for debug

    total_image_count = len(img_paths)
    top1_count = 0
    top5_count = 0

    with open(args.engine, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    start = time.time()
    for img_index in range(0, total_image_count):
        curr_img_paths = img_paths[img_index]
        input_images = get_inputs(curr_img_paths, args.directory)
        #print(curr_img_paths)
        expected_label = img_labels[img_index]

        top5_labels = []

        # infer(args.engine, preprocess_func, batch_size=args.batch_size, input_images=input_images,
        #       labels=labels, num_classes=args.num_classes)
        #print(input_images)
        batch_outs = infer(engine, preprocess_func, batch_size=args.batch_size, input_images=input_images,
              labels=labels, num_classes=args.num_classes)

        for batch_out in batch_outs:
            topk_indices = np.argsort(batch_out)[-1*args.num_classes:][::-1]
            #topk_indices = topk_indices-1
            #preds = labels[topk_indices]
            probs = batch_out[topk_indices]

            topk_indices = topk_indices - args.label_offset
            if args.verbose:
                print("Input image:", input_images)
                for index, prob in zip(topk_indices, probs):
                    print("\tPrediction: {:29} Probability: {:0.4f}".format(index, prob))

        index = topk_indices
        j = 1
        for i in index[:-1]:
            #print('Top-%d: class=%s(%d) ; probability=%f' % (j, labels[i], i, prob[i]))
            label = i
            top5_labels.append((int(label)))
            j += 1

        if expected_label == top5_labels[0]:
            top1_count += 1
        if expected_label in top5_labels:
            top5_count += 1

        curr_completed_count = img_index+1
        if curr_completed_count % 10 == 0:
            print("Finished image index %d out of %d" % (
                (curr_completed_count, total_image_count)))
            print("  Current Top-1/5 accuracy:")
            print_topk_accuracy(curr_completed_count, top1_count,
                                top5_count)
            current = time.time()
            hours, rem = divmod(current - start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("elapsed time: {:0>2}:{:0>2}:{:05.5f}".format(int(hours), int(minutes), seconds))
            print("avg. latency: {:05.5f}".format((current-start)/curr_completed_count))
