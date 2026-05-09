#pragma once

#include "shooting.hpp"
#include "types.hpp"

namespace pendulum {

struct SheetSearchSettings {
    // Search m in [m_center - m_radius, m_center + m_radius]
    int m_radius_min = 6;
    int m_radius_max = 60;

    // Heuristic scaling: expand radius with initial speed.
    double m_radius_per_speed = 2.0;

    ShootSettings shoot{};

    // Optional continuation schedule; if nonempty, solveCostatesSingleSheetLMContinuation is used.
    // Values should be increasing horizons (e.g. [2,4,6,8,10]).
    Eigen::VectorXd T_schedule{};

    // Early exit if we find a candidate with residual <= this threshold.
    double good_enough_resid = 1e-10;

    // If true, print sheet/seed diagnostics to stderr (never stdout).
    bool debug = false;
};

struct SheetSearchResult {
    ShootResult best{};
    int best_m = 0;     // sheet index (equilibrium at 2π m)
    double theta0_shifted = 0.0;  // theta - 2π m used in solve
};

// Multi-sheet search around theta ≈ 2π m.
SheetSearchResult solveWithSheetSearch(const Params& p, const State& x0, const SheetSearchSettings& s);

}  // namespace pendulum

