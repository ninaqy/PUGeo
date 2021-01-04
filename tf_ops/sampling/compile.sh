#/bin/bash
CUDA=/home/qianyue/cuda/cuda-10.0
TF=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

$CUDA/bin/nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $TF/include -I $CUDA/include -I $TF/include/external/nsync/public -lcudart -L $CUDA/lib64/ -L $TF -ltensorflow_framework -O2 #-D_GLIBCXX_USE_CXX11_ABI=0
