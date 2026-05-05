/**
 * @file session_select.h
 * @brief Release session backend selector.
 *
 * The release build supports only the libibverbs EFA SRD backend selected by
 * -DINTERNODE_BACKEND_EFA.
 */
#pragma once

#if defined(INTERNODE_BACKEND_EFA)
#include "session_efa.h"
#else
#error "Release supports only -DINTERNODE_BACKEND_EFA"
#endif
