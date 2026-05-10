#include "shooting/defect_jacobian_host.hpp"

#include <vector>

#include "core/manifold_seed.hpp"

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Sparse>
#include <Eigen/Dense>

// Residual layout (m = 4N + 2 = n + 2):
// Rows 0 .. 4*(N-1)-1   : continuity Φ(z_k) − z_{k+1} for k = 0..N-2 (four rows each).
// Rows 4*(N-1) .. 4*N-1 : boundary — θ₀, φ₀ on node 0; θ, φ of Φ(z_{N-1}) vs goals (legacy indexing).
// Rows 4*N .. 4*N+1    : manifold λ(ẑ) − P x(ẑ) at terminal forward state ẑ = Φ(z_{N-1}).

void build_global_system(const HDArrays &solver_arrays, const SystemParams &sys_params, SparseMat &J,
                         VectorXd &F) {
    const int NUM_ROWS_PER_SEGMENT = 4;
    const double FINAL_THETA_DESIRED = sys_params.theta_goal;
    const double FINAL_PHI_DESIRED = sys_params.phi_goal;

    int N = sys_params.num_shooting_intervals;
    int n = NUM_ROWS_PER_SEGMENT * N;
    int m = n + 2;

    double Pm[4];
    stable_manifold_P(sys_params.alpha, Pm);
    const double P11 = Pm[0];
    const double P12 = Pm[1];
    const double P21 = Pm[2];
    const double P22 = Pm[3];

    J.resize(m, n);
    F.resize(m);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(N * 24 + 16);

    for (int segment = 0; segment < N - 1; segment++) {
        int curr_row_offset = segment * NUM_ROWS_PER_SEGMENT;
        int next_row_offset = (segment + 1) * NUM_ROWS_PER_SEGMENT;

        const VarState &current_end_state = solver_arrays.h_segment_results[segment].final_state;

        F(curr_row_offset + 0) = current_end_state.theta() - solver_arrays.h_node_guesses[next_row_offset + 0];
        F(curr_row_offset + 1) = current_end_state.phi() - solver_arrays.h_node_guesses[next_row_offset + 1];
        F(curr_row_offset + 2) = current_end_state.l1() - solver_arrays.h_node_guesses[next_row_offset + 2];
        F(curr_row_offset + 3) = current_end_state.l2() - solver_arrays.h_node_guesses[next_row_offset + 3];

        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                triplets.push_back(Eigen::Triplet<double>(
                    curr_row_offset + r, curr_row_offset + c, current_end_state.M(r, c)));
            }
        }

        for (int i = 0; i < 4; i++) {
            triplets.push_back(Eigen::Triplet<double>(curr_row_offset + i, next_row_offset + i, -1.0));
        }
    }

    int bc_row_offset = (N - 1) * NUM_ROWS_PER_SEGMENT;

    F(bc_row_offset + 0) = solver_arrays.h_node_guesses[0] - sys_params.theta_init;
    triplets.push_back(Eigen::Triplet<double>(bc_row_offset + 0, 0, 1.0));

    F(bc_row_offset + 1) = solver_arrays.h_node_guesses[1] - sys_params.phi_init;
    triplets.push_back(Eigen::Triplet<double>(bc_row_offset + 1, 1, 1.0));

    const VarState &final_end_state = solver_arrays.h_segment_results[N - 1].final_state;

    F(bc_row_offset + 2) = final_end_state.theta() - FINAL_THETA_DESIRED;
    F(bc_row_offset + 3) = final_end_state.phi() - FINAL_PHI_DESIRED;

    for (int c = 0; c < 4; c++) {
        triplets.push_back(
            Eigen::Triplet<double>(bc_row_offset + 2, bc_row_offset + c, final_end_state.M(0, c)));
        triplets.push_back(
            Eigen::Triplet<double>(bc_row_offset + 3, bc_row_offset + c, final_end_state.M(1, c)));
    }

    int man1 = n;
    int man2 = n + 1;
    int col0 = (N - 1) * NUM_ROWS_PER_SEGMENT;

    const double thT = final_end_state.theta();
    const double phT = final_end_state.phi();
    const double l1T = final_end_state.l1();
    const double l2T = final_end_state.l2();

    F(man1) = l1T - (P11 * thT + P12 * phT);
    F(man2) = l2T - (P21 * thT + P22 * phT);

    // ∂/∂z_{N-1}: (e_{λ1} - P11 e_θ - P12 e_φ)^T M,  (e_{λ2} - P21 e_θ - P22 e_φ)^T M
    for (int c = 0; c < 4; c++) {
        double v0 = -P11 * final_end_state.M(0, c) - P12 * final_end_state.M(1, c) + final_end_state.M(2, c);
        double v1 = -P21 * final_end_state.M(0, c) - P22 * final_end_state.M(1, c) + final_end_state.M(3, c);
        triplets.push_back(Eigen::Triplet<double>(man1, col0 + c, v0));
        triplets.push_back(Eigen::Triplet<double>(man2, col0 + c, v1));
    }

    J.setFromTriplets(triplets.begin(), triplets.end());
}
