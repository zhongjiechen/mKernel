/**
 * @file
 * @brief dist::tma — TMA wrappers for the dist:: distributed-buffer namespace.
 *
 * Step 1 (this revision): import the kittens::tma overload set into dist::tma
 * via `using` declarations. Functionally identical to calling kittens::tma::*
 * directly, but moves the call-site spelling under the dist:: namespace so
 * subsequent steps can replace each forwarded function with our own PTX wrapper
 * one at a time.
 */

#pragma once

#include "dbuf.cuh"

namespace dist {
namespace tma {

// Bring the full overload set of each kittens::tma::* function into dist::tma.
// `using` declarations import all overloads of the named function, so braced-
// init-list coord arguments and concept-constrained templates continue to
// match exactly as they did at the kittens::tma:: call sites.
using ::kittens::tma::store_async;
using ::kittens::tma::store_add_async;
using ::kittens::tma::load_async;
using ::kittens::tma::store_async_read_wait;
using ::kittens::tma::store_async_wait;
using ::kittens::tma::expect_bytes;

} // namespace tma
} // namespace dist
