#include "shooting/defect_jacobian_host.hpp"

#include <vector>

#include "core/solver_types.cuh"

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Sparse>
#include <Eigen/Dense>

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

    for (int segment = 0; segment < N - 1; segment++) {
        int curr_row_offset = segment * NUM_ROWS_PER_SEGMENT;
        int next_row_offset = (segment + 1) * NUM_ROWS_PER_SEGMENT;

        const VarState &current_end_state = solver_arrays.h_segment_results[segment].final_state;

        if (!int_params.backward_time) {
            double next_theta_start_guess = solver_arrays.h_node_guesses[next_row_offset + 0];
            double next_phi_start_guess = solver_arrays.h_node_guesses[next_row_offset + 1];
            double next_l1_start_guess = solver_arrays.h_node_guesses[next_row_offset + 2];
            double next_l2_start_guess = solver_arrays.h_node_guesses[next_row_offset + 3];

            F(curr_row_offset + 0) = current_end_state.theta() - next_theta_start_guess;
            F(curr_row_offset + 1) = current_end_state.phi() - next_phi_start_guess;
            F(curr_row_offset + 2) = current_end_state.l1() - next_l1_start_guess;
            F(curr_row_offset + 3) = current_end_state.l2() - next_l2_start_guess;

            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    triplets.push_back(Eigen::Triplet<double>(
                        curr_row_offset + r, curr_row_offset + c, current_end_state.M(r, c)));
                }
            }

            for (int i = 0; i < 4; i++) {
                triplets.push_back(Eigen::Triplet<double>(curr_row_offset + i, next_row_offset + i, -1.0));
            }
        } else {
            double left_theta = solver_arrays.h_node_guesses[curr_row_offset + 0];
            double left_phi = solver_arrays.h_node_guesses[curr_row_offset + 1];
            double left_l1 = solver_arrays.h_node_guesses[curr_row_offset + 2];
            double left_l2 = solver_arrays.h_node_guesses[curr_row_offset + 3];

            F(curr_row_offset + 0) = current_end_state.theta() - left_theta;
            F(curr_row_offset + 1) = current_end_state.phi() - left_phi;
            F(curr_row_offset + 2) = current_end_state.l1() - left_l1;
            F(curr_row_offset + 3) = current_end_state.l2() - left_l2;

            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    triplets.push_back(Eigen::Triplet<double>(
                        curr_row_offset + r, next_row_offset + c, current_end_state.M(r, c)));
                }
            }

            for (int i = 0; i < 4; i++) {
                triplets.push_back(Eigen::Triplet<double>(curr_row_offset + i, curr_row_offset + i, -1.0));
            }
        }
    }

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
        triplets.push_back(
            Eigen::Triplet<double>(bc_row_offset + 2, bc_row_offset + c, final_end_state.M(0, c)));

        triplets.push_back(
            Eigen::Triplet<double>(bc_row_offset + 3, bc_row_offset + c, final_end_state.M(1, c)));
    }

    J.setFromTriplets(triplets.begin(), triplets.end());
}
