
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

// Python bindings to call kernel():
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <sstream>
#include <cstdlib>

#ifndef _MSC_VER
#if __cplusplus < 202002L
// C++20 (earlier) code
// https://en.cppreference.com/w/cpp/language/attributes/likely
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#endif
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

// This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
// We manually link it below to workaround issues with fbcode build.
static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
    static_assert(std::is_pointer<T>::value, "arg type must be pointer or long");
    return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
}
template <> inline int64_t parse_arg<int64_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == -1 && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return result;
}
template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == reinterpret_cast<void*>(-1) && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return reinterpret_cast<uintptr_t>(result);
}



static PyObject* kernel_py(PyObject* self, PyObject* args) {
    try {
        if(unlikely(!PyTuple_CheckExact(args)))
            throw std::runtime_error("tuple args required");
        if(unlikely(PyTuple_GET_SIZE(args) != 5))
            throw std::runtime_error("requires 5 args");
        kernel(parse_arg<float*>(args, 0), parse_arg<float*>(args, 1), parse_arg<float*>(args, 2), parse_arg<float*>(args, 3), parse_arg<float*>(args, 4));Py_RETURN_NONE;
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}

static PyMethodDef py_methods[] = {
    {"kernel", kernel_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef py_module =
    {PyModuleDef_HEAD_INIT, "kernel", NULL, -1, py_methods};

PyMODINIT_FUNC PyInit_kernel(void) {
    const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
    if(!str_addr) {
        PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
        return nullptr;
    }
    std::istringstream iss(str_addr);
    uintptr_t addr = 0;
    iss >> addr;
    _torchinductor_pyobject_tensor_data_ptr =
        reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
    return PyModule_Create(&py_module);
}
