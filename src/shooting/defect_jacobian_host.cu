#include "shooting/defect_jacobian_host.hpp"

// Shooting defect F(S): for forward time, segment k integrates from knot k; interior rows require
// Phi_k(S_k) - S_{k+1} = 0 (continuity of theta,phi,l1,l2). For backward_time, segment k starts from
// knot k+1 (k < N-1) and rows require Phi_k(S_{k+1}) - S_k = 0 — the same physical chain but with
// reversed indexing of which knot is the IVP initial condition. The last segment's flow appears in
// both the (N-2,N-1) interface (if N>1) and the terminal boundary rows. VarState::M stores dPhi/d(start)
// for the segment's initial argument (4x4). Boundary block: rows pin (theta_0,phi_0) to init values and
// (theta_T,phi_T) from Phi_{N-1}(start of last segment) to goals; terminal rows use only the first two
// rows of M because only (theta,phi) are constrained at the horizon.

#include <vector>

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace {

void push_flow_jacobian_block(std::vector<Eigen::Triplet<double>> &triplets, int row0, int col0,
                              const VarState &flow_end) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            triplets.push_back(Eigen::Triplet<double>(row0 + r, col0 + c, flow_end.M(r, c)));
        }
    }
}

void push_neg_identity_rows(std::vector<Eigen::Triplet<double>> &triplets, int row0, int col0) {
    for (int i = 0; i < 4; ++i) {
        triplets.push_back(Eigen::Triplet<double>(row0 + i, col0 + i, -1.0));
    }
}

void set_interior_interface_residual(Eigen::VectorXd &F, int row0, const VarState &flow_end, int knot_offset,
                                     const std::vector<double> &node_guesses) {
    F(row0 + 0) = flow_end.theta() - node_guesses[knot_offset + 0];
    F(row0 + 1) = flow_end.phi() - node_guesses[knot_offset + 1];
    F(row0 + 2) = flow_end.l1() - node_guesses[knot_offset + 2];
    F(row0 + 3) = flow_end.l2() - node_guesses[knot_offset + 3];
}

} // namespace

/// Rows are grouped in blocks of 4 per segment interface / boundary; columns follow the same nodal layout.
// Square system: 4N equations in 4N unknowns, N = num_shooting_intervals (also the number of knots).
void build_global_system(const HDArrays &solver_arrays, const SystemParams &sys_params,
                         const IntegratorParams &int_params, SparseMat &J, VectorXd &F) {
    const int NUM_ROWS_PER_SEGMENT = 4;
    const double FINAL_THETA_DESIRED = sys_params.theta_goal;
    const double FINAL_PHI_DESIRED = sys_params.phi_goal;

    int N = sys_params.num_shooting_intervals;
    int system_size = NUM_ROWS_PER_SEGMENT * N;

    J.resize(system_size, system_size);
    F.resize(system_size);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(N * 20);

    // Interior interfaces: rows curr_row_offset + {0,1,2,3} are the four scalar continuity defects.
    for (int segment = 0; segment < N - 1; segment++) {
        int curr_row_offset = segment * NUM_ROWS_PER_SEGMENT;
        int next_row_offset = (segment + 1) * NUM_ROWS_PER_SEGMENT;

        const VarState &current_end_state = solver_arrays.h_segment_results[segment].final_state;

        if (!int_params.backward_time) {
            // Forward MS: Phi_k maps knot k -> knot k+1; Jacobian w.r.t. S_k is M, w.r.t. S_{k+1} is -I.
            set_interior_interface_residual(F, curr_row_offset, current_end_state, next_row_offset,
                                            solver_arrays.h_node_guesses);
            push_flow_jacobian_block(triplets, curr_row_offset, curr_row_offset, current_end_state);
            push_neg_identity_rows(triplets, curr_row_offset, next_row_offset);
        } else {
            // Backward MS: IVP initial data is S_{k+1}; defect is Phi_k(S_{k+1}) - S_k, so columns for
            // sensitivities attach to knot k+1 (next_row_offset) and identity to knot k (curr_row_offset).
            set_interior_interface_residual(F, curr_row_offset, current_end_state, curr_row_offset,
                                            solver_arrays.h_node_guesses);
            push_flow_jacobian_block(triplets, curr_row_offset, next_row_offset, current_end_state);
            push_neg_identity_rows(triplets, curr_row_offset, curr_row_offset);
        }
    }

    // Boundary rows (last 4 rows): two Dirichlet conditions on position at the first knot, two on
    // position at the trajectory end (costates at t=0 and co-state rows are left free here).
    int bc_row_offset = (N - 1) * NUM_ROWS_PER_SEGMENT;

    double start_theta = solver_arrays.h_node_guesses[0];
    F(bc_row_offset + 0) = start_theta - sys_params.theta_init;
    triplets.push_back(Eigen::Triplet<double>(bc_row_offset + 0, 0, 1.0));

    double start_phi = solver_arrays.h_node_guesses[1];
    F(bc_row_offset + 1) = start_phi - sys_params.phi_init;
    triplets.push_back(Eigen::Triplet<double>(bc_row_offset + 1, 1, 1.0));

    const VarState &final_end_state = solver_arrays.h_segment_results[N - 1].final_state;

    F(bc_row_offset + 2) = final_end_state.theta() - FINAL_THETA_DESIRED;
    F(bc_row_offset + 3) = final_end_state.phi() - FINAL_PHI_DESIRED;

    for (int c = 0; c < 4; c++) {
        // Terminal (theta,phi): chain rule through the last segment — derivatives w.r.t. costates at the
        // last segment's start knot live in columns bc_row_offset..bc_row_offset+3 (that knot's block).
        triplets.push_back(
            Eigen::Triplet<double>(bc_row_offset + 2, bc_row_offset + c, final_end_state.M(0, c)));

        triplets.push_back(
            Eigen::Triplet<double>(bc_row_offset + 3, bc_row_offset + c, final_end_state.M(1, c)));
    }

    J.setFromTriplets(triplets.begin(), triplets.end());
}
