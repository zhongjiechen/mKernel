#pragma once

namespace dist {

struct coord {
    int b;
    int d;
    int r;
    int c;

    __host__ __device__ constexpr coord(int b_, int d_, int r_, int c_)
        : b(b_), d(d_), r(r_), c(c_) {}
    __host__ __device__ constexpr coord(int d_, int r_, int c_)
        : b(0), d(d_), r(r_), c(c_) {}
    __host__ __device__ constexpr coord(int r_, int c_)
        : b(0), d(0), r(r_), c(c_) {}
    __host__ __device__ constexpr coord(int c_)
        : b(0), d(0), r(0), c(c_) {}
    __host__ __device__ constexpr coord()
        : b(0), d(0), r(0), c(0) {}
};

} // namespace dist
