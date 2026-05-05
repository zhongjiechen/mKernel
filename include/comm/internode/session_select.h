/**
 * @file session_select.h
 * @brief Release session backend selector.
 *
 * Compile-time selection between:
 *   - libibverbs SRD (AWS EFA):              -DINTERNODE_BACKEND_EFA
 *   - libibverbs RC  (ConnectX-7 / IB):      -DINTERNODE_BACKEND_IBVERBS
 *
 * Both backends expose the same host-side API in namespace `internode`.
 */
#pragma once

#if defined(INTERNODE_BACKEND_EFA)
#include "session_efa.h"
#elif defined(INTERNODE_BACKEND_IBVERBS)
#include "session.h"
#else
#error "No internode backend selected. Pass -DINTERNODE_BACKEND_EFA (AWS EFA SRD) or -DINTERNODE_BACKEND_IBVERBS (ConnectX-7 RC)."
#endif
