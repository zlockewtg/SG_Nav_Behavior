// PyTorch 1.x: THC/THCAtomics.cuh. PyTorch 2.x: ATen/cuda/Atomic.cuh (THC removed).
#pragma once
#include <torch/extension.h>
#if TORCH_VERSION_MAJOR >= 2
#include <ATen/cuda/Atomic.cuh>
#else
#include <THC/THCAtomics.cuh>
#endif
