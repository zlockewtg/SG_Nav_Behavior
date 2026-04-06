// Replacements for removed THC headers (PyTorch 2.x).
#pragma once

#include <c10/cuda/CUDAException.h>

#ifndef THCCeilDiv
#define THCCeilDiv(a, b) ((((a) + (b)-1) / (b)))
#endif

#ifndef THCudaCheck
#define THCudaCheck(expr) C10_CUDA_CHECK(expr)
#endif
