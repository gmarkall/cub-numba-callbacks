from numba import config, cuda
import os
import warnings

# Add extra options (include path, optimization flags, etc) for NVRTC here
extra_options = ['-Iother_include_path']


# Copied from numba.cuda.cudadrv.nvrtc, modified to allow extra options to be
# added.

def nvrtc_compile(src, name, cc):
    """
    Compile a CUDA C/C++ source to PTX for a given compute capability.

    :param src: The source code to compile
    :type src: str
    :param name: The filename of the source (for information only)
    :type name: str
    :param cc: A tuple ``(major, minor)`` of the compute capability
    :type cc: tuple
    :return: The compiled PTX and compilation log
    :rtype: tuple
    """
    nvrtc = cuda.cudadrv.nvrtc.NVRTC()
    program = nvrtc.create_program(src, name)

    # Compilation options:
    # - Compile for the current device's compute capability.
    # - The CUDA include path is added.
    # - Relocatable Device Code (rdc) is needed to prevent device functions
    #   being optimized away.
    major, minor = cc
    arch = f'--gpu-architecture=compute_{major}{minor}'
    include = f'-I{config.CUDA_INCLUDE_PATH}'

    cudadrv_path = os.path.dirname(os.path.abspath(__file__))
    numba_cuda_path = os.path.dirname(cudadrv_path)
    numba_include = f'-I{numba_cuda_path}'
    options = [arch, include, numba_include, '-rdc', 'true']
    options += extra_options

    # Compile the program
    compile_error = nvrtc.compile_program(program, options)

    # Get log from compilation
    log = nvrtc.get_compile_log(program)

    # If the compile failed, provide the log in an exception
    if compile_error:
        msg = (f'NVRTC Compilation failure whilst compiling {name}:\n\n{log}')
        raise cuda.cudadrv.nvrtc.NvrtcError(msg)

    # Otherwise, if there's any content in the log, present it as a warning
    if log:
        msg = (f"NVRTC log messages whilst compiling {name}:\n\n{log}")
        warnings.warn(msg)

    ptx = nvrtc.get_ptx(program)
    return ptx, log


# Monkey-patch the existing implementation
cuda.cudadrv.nvrtc.compile = nvrtc_compile


# An example kernel. When compiling example.cu, other_include_path will also be
# on the include path
@cuda.jit(link=['example.cu'])
def f():
    pass


# Launch the kernel to force compilation
f[1, 1]()
