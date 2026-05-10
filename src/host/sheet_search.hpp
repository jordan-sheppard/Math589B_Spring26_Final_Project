#pragma once

#include "shoot_types.hpp"

#include <vector>

namespace pendulum {

struct SheetSearchSettingsHost {
    int m_radius_min = 6;
    int m_radius_max = 60;
    double m_radius_per_speed = 2.0;
    ShootSettingsHost shoot{};
    std::vector<double> T_schedule{};
    double good_enough_resid = 1e-10;
    bool debug = false;
};

struct SheetSearchResultHost {
    ShootResultHost best{};
    int best_m = 0;
    double theta0_shifted = 0.0;
};

SheetSearchResultHost solve_with_sheet_search(const Params& p, const State& x0, const SheetSearchSettingsHost& s);

}  // namespace pendulum
