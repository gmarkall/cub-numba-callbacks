CCCL_PATH := /home/gmarkall/numbadev/cccl
CUB_INCLUDE := -I$(CCCL_PATH)/cub
LIBCUDACXX_INCLUDE := -I$(CCCL_PATH)/libcudacxx/include
INCLUDES := $(CUB_INCLUDE) $(LIBCUDACXX_INCLUDE)
NVCCFLAGS := --generate-code=arch=compute_75,code=sm_75 -Xcompiler=-Werror -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wno-unused-function -rdc true -v

%.o: %.cu
	nvcc  $(NVCCFLAGS) $(INCLUDES) -c $<

%.ptx: %.cu
	nvcc $(NVCCFLAGS) $(INCLUDES) -ptx $<

all: example_block_reduce_cpp example_block_reduce_numba

example_block_reduce_cpp: example_block_reduce.o cpp_callback.o cpp_callback.ptx
	nvcc $(NVCCFLAGS) -o example_block_reduce_cpp example_block_reduce.o cpp_callback.o

numba_callback.ptx: numba_callback.py
	python numba_callback.py

numba_callback.cubin: numba_callback.ptx
	ptxas -arch sm_75 -c numba_callback.ptx -o numba_callback.cubin

numba_callback.o: numba_callback.cubin
	fatbinary -64 --image3=kind=elf,sm=75,file=numba_callback.cubin --device-c --embedded-fatbin=numba_callback.fatbin.c
	gcc -D__CUDA_ARCH__=750 -D__CUDA_ARCH_LIST__=750 -c -x c++ numba_callback.fatbin.c -o numba_callback.o "-I/usr/local/cuda-12.3/bin/../targets/x86_64-linux/include"   -m64

example_block_reduce_numba: numba_callback.o example_block_reduce.o
	nvcc $(NVCCFLAGS) -o example_block_reduce_numba example_block_reduce.o numba_callback.o

clean:
	rm -f example_block_reduce_numba example_block_reduce_cpp *.o *.ptx numba_callback.fatbin* numba_callback.cubin
