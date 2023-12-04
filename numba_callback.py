from numba import cuda, types

int32 = types.int32
int32p = types.CPointer(types.int32)


def add(a, b):
    return a + b


def compile_operator(pyfunc, sig, abi_name):
    jitted = cuda.jit(pyfunc)

    def wrapper(this, arg0, arg1):
        return jitted(arg0[0], arg1[0])

    pointer_argtypes = [types.CPointer(arg) for arg in sig.args]
    wrapper_argtypes = [types.CPointer(types.none)] + pointer_argtypes
    wrapper_sig = sig.return_type(*wrapper_argtypes)

    abi_info = {'abi_name': abi_name}

    return cuda.compile(wrapper, wrapper_sig, abi='c', abi_info=abi_info)


# Name mangled:
#     cuda::std::__4::common_type<int&, int&>::type
#     Add::operator()<int&, int&>(int&, int&) const
abi_name = ("_ZNK3AddclIRiS1_EEN4cuda3std"
            "3__411common_typeIJT_T0_EE4typeEOS6_OS7_")

sig = int32(int32, int32)
ptx, resty = compile_operator(add, sig, abi_name)

with open('numba_callback.ptx', 'w') as f:
    f.write(ptx)
