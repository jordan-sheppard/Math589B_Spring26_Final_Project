#pragma once

#include <cstdlib>

/// Enable with: MATH589_DEBUG=1 ./solver theta phi alpha
/// All diagnostics go to stderr so stdout stays grader-clean.
inline bool math589_solver_debug_enabled() {
    const char *e = std::getenv("MATH589_DEBUG");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}

/// Extra LM inner damping/backtrack lines on stderr (only printed when MATH589_DEBUG is also set).
inline bool math589_solver_debug_lm_verbose() {
    const char *e = std::getenv("MATH589_DEBUG_LM_VERBOSE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}
