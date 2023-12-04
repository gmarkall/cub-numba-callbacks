CCCL_PATH := /home/gmarkall/numbadev/cccl
CUB_INCLUDE := -I$(CCCL_PATH)/cub
LIBCUDACXX_INCLUDE := -I$(CCCL_PATH)/libcudacxx/include
INCLUDES := $(CUB_INCLUDE) $(LIBCUDACXX_INCLUDE)
NVCCFLAGS := --generate-code=arch=compute_75,code=sm_75 -Xcompiler=-Werror -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wno-unused-function -rdc true

all:
	nvcc $(NVCCFLAGS) $(INCLUDES) -c example_block_reduce.cu
	nvcc $(NVCCFLAGS) $(INCLUDES) -c cpp_callback.cu
	nvcc $(NVCCFLAGS) $(INCLUDES) -ptx cpp_callback.cu
	nvcc $(NVCCFLAGS) -o example_block_reduce_cpp example_block_reduce.o cpp_callback.o
	python numba_callback.py
	ptxas -arch sm_75 -c numba_callback.ptx -o numba_callback.o
	nvcc $(NVCCFLAGS) -o example_block_reduce_numba example_block_reduce.o numba_callback.o

clean:
	rm -f example_block_reduce_numba example_block_reduce_cpp *.o *.ptx
