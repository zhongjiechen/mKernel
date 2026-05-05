#pragma once

#include "../pyutils/parallel_tensor.cuh"

namespace dist {

/**
 * Release-facing distributed buffer wrapper.
 *
 * This intentionally exposes the same storage semantics as the legacy
 * TKParallelTensor (CUDA tensor storage, per-GPU IPC pointers, optional
 * multicast VA), but moves operator APIs onto a dist-owned type.
 */
struct ParallelBuffer : public kittens::py::TKParallelTensor {
    using kittens::py::TKParallelTensor::TKParallelTensor;
};

}  // namespace dist

#define BIND_DIST_PARALLEL_BUFFER(m) \
    pybind11::class_<dist::ParallelBuffer>(m, "DistBuffer", pybind11::module_local()) \
        .def(pybind11::init<const at::Tensor&, int, int, bool>(), \
             pybind11::arg("tensor"), \
             pybind11::arg("local_rank"), \
             pybind11::arg("local_world_size"), \
             pybind11::arg("multicast") = false) \
        .def(pybind11::init<const std::vector<int64_t>&, const at::ScalarType&, int, int, bool>(), \
             pybind11::arg("shape"), \
             pybind11::arg("dtype"), \
             pybind11::arg("local_rank"), \
             pybind11::arg("local_world_size"), \
             pybind11::arg("multicast") = false) \
        .def("data", &dist::ParallelBuffer::data) \
        .def_readonly("data_", &dist::ParallelBuffer::data_) \
        .def_property_readonly("multicast_ptr_u64", [](const dist::ParallelBuffer &t) { \
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(t.multicast_ptr_)); \
        }) \
        .def_property_readonly("multicast_enabled", [](const dist::ParallelBuffer &t) { \
            return t.multicast_; \
        }) \
        .def_readonly("local_rank_", &dist::ParallelBuffer::local_rank_) \
        .def_readonly("local_world_size_", &dist::ParallelBuffer::local_world_size_)
