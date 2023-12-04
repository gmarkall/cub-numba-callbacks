CCCL_PATH := /home/gmarkall/numbadev/cccl
CUB_INCLUDE := -I$(CCCL_PATH)/cub
LIBCUDACXX_INCLUDE := -I$(CCCL_PATH)/libcudacxx/include
INCLUDES := $(CUB_INCLUDE) $(LIBCUDACXX_INCLUDE)
NVCCFLAGS := --generate-code=arch=compute_75,code=sm_75 -Xcompiler=-Werror -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wno-unused-function -rdc true

all:
	nvcc $(NVCCFLAGS) $(INCLUDES) -c example_block_reduce_numba.cu
	nvcc $(NVCCFLAGS) $(INCLUDES) -c numba_add.cu
	nvcc $(NVCCFLAGS) $(INCLUDES) -ptx numba_add.cu
	nvcc $(NVCCFLAGS) -o example_block_reduce_numba example_block_reduce_numba.o numba_add.o

