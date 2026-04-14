# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


cpp_fused_stack_0 = async_compile.cpp_pybinding(['const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include "/mnt/public/tgy/SG-Nav/behavior/torchinductor_root/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        auto tmp0 = in_ptr0[static_cast<int64_t>(2L)];
        auto tmp4 = in_ptr0[static_cast<int64_t>(0L)];
        auto tmp8 = in_ptr0[static_cast<int64_t>(1L)];
        auto tmp1 = static_cast<float>(0.5);
        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
        auto tmp3 = std::cos(tmp2);
        auto tmp5 = decltype(tmp4)(tmp4 * tmp1);
        auto tmp6 = std::sin(tmp5);
        auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
        auto tmp9 = decltype(tmp8)(tmp8 * tmp1);
        auto tmp10 = std::cos(tmp9);
        auto tmp11 = decltype(tmp7)(tmp7 * tmp10);
        auto tmp12 = std::sin(tmp2);
        auto tmp13 = std::cos(tmp5);
        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
        auto tmp15 = std::sin(tmp9);
        auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
        auto tmp17 = decltype(tmp11)(tmp11 - tmp16);
        auto tmp18 = decltype(tmp3)(tmp3 * tmp13);
        auto tmp19 = decltype(tmp18)(tmp18 * tmp15);
        auto tmp20 = decltype(tmp12)(tmp12 * tmp6);
        auto tmp21 = decltype(tmp20)(tmp20 * tmp10);
        auto tmp22 = decltype(tmp19)(tmp19 + tmp21);
        auto tmp23 = decltype(tmp14)(tmp14 * tmp10);
        auto tmp24 = decltype(tmp7)(tmp7 * tmp15);
        auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
        auto tmp26 = decltype(tmp18)(tmp18 * tmp10);
        auto tmp27 = decltype(tmp20)(tmp20 * tmp15);
        auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
        out_ptr0[static_cast<int64_t>(0L)] = tmp17;
        out_ptr1[static_cast<int64_t>(0L)] = tmp22;
        out_ptr2[static_cast<int64_t>(0L)] = tmp25;
        out_ptr3[static_cast<int64_t>(0L)] = tmp28;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (3, ), (1, ))
    buf4 = empty_strided_cpu((4, ), (1, ), torch.float32)
    buf0 = reinterpret_tensor(buf4, (1, ), (1, ), 0)  # alias
    buf1 = reinterpret_tensor(buf4, (1, ), (1, ), 1)  # alias
    buf2 = reinterpret_tensor(buf4, (1, ), (1, ), 2)  # alias
    buf3 = reinterpret_tensor(buf4, (1, ), (1, ), 3)  # alias
    cpp_fused_stack_0(arg0_1, buf0, buf1, buf2, buf3)
    del arg0_1
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
