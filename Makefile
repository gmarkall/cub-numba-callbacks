CCCL_PATH := /home/gmarkall/numbadev/cccl
CUB_INCLUDE := -I$(CCCL_PATH)/cub
LIBCUDACXX_INCLUDE := -I$(CCCL_PATH)/libcudacxx/include
INCLUDES := $(CUB_INCLUDE) $(LIBCUDACXX_INCLUDE)
NVCCFLAGS := --generate-code=arch=compute_75,code=sm_75 -Xcompiler=-Werror -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wno-unused-function -rdc true

%.o: %.cu
	nvcc  $(NVCCFLAGS) $(INCLUDES) -c $<

%.ptx: %.cu
	nvcc $(NVCCFLAGS) $(INCLUDES) -ptx $<

all: example_block_reduce_cpp example_block_reduce_numba

example_block_reduce_cpp: example_block_reduce.o cpp_callback.o cpp_callback.ptx
	nvcc $(NVCCFLAGS) -o example_block_reduce_cpp example_block_reduce.o cpp_callback.o

numba_callback.ptx: numba_callback.py
	python numba_callback.py

numba_callback.o: numba_callback.ptx
	ptxas -arch sm_75 -c numba_callback.ptx -o numba_callback.o

example_block_reduce_numba: numba_callback.o example_block_reduce.o
	nvcc $(NVCCFLAGS) -o example_block_reduce_numba example_block_reduce.o numba_callback.o

clean:
	rm -f example_block_reduce_numba example_block_reduce_cpp *.o *.ptx
