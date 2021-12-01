#!/bin/bash

CUDA=/home/qianyue/cuda/cuda-10.0
TF=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

$CUDA/bin/nvcc -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $TF/include -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $TF/include -lcudart -L $CUDA/lib64 -O2 -I $TF/include/external/nsync/public -L $TF -ltensorflow_framework


