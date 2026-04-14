
#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_ZVECTOR) || defined(CPU_CAPABILITY_NEON) || defined(CPU_CAPABILITY_VSX)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

alignas(64) float in_out_ptr0[16] = {0.0};

extern "C" void __avx_chk_kernel() {
    auto tmp0 = at::vec::Vectorized<float>(1);
    auto tmp1 = tmp0.exp();
    tmp1.store(in_out_ptr0);
}
