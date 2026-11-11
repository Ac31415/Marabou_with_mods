#!/bin/bash
curdir=$pwd
mydir="${0%/*}"
<<<<<<< HEAD
onnxdir=onnx-1.12.0
=======
onnxdir=onnx-1.15.0
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390
protobufdir=protobuf-3.19.2

# This script downloads the protobuffer file from the Onnx repo.
# There are no C++ bindings for onnx so we compile the protobuffer ourselves.
# see https://stackoverflow.com/questions/67301475/parse-an-onnx-model-using-c-extract-layers-input-and-output-shape-from-an-on
# for details.

cd $mydir

mkdir -p $onnxdir
cd $onnxdir
echo "downloading onnx proto file"
<<<<<<< HEAD
wget -q https://raw.githubusercontent.com/onnx/onnx/v1.12.0/onnx/onnx.proto3 -O onnx.proto3
=======
wget -q https://raw.githubusercontent.com/onnx/onnx/v1.15.0/onnx/onnx.proto3 -O onnx.proto3
>>>>>>> 9f69e2a1dcb3e788a9ef7cf084b2ae162e7ca390

echo "compiling the onnx proto file"
../$protobufdir/installed/bin/protoc --cpp_out=. onnx.proto3

cd $curdir
