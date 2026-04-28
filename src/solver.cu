#include "solver.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>
#include <complex>

__host__ __device__
StateVec evaluate_derivatives(const StateVec& y, const double alpha) {
    // Evaluates the state/costate/cost derivatives at the current
    // controlled pendulum state/costate, and stores them in a state vector
    
    StateVec dy;

    // CUDA Hardware Optimization: Compute expensive sin & cos simultaneously!
    double sin_t, cos_t;
    sincos(y.theta, &sin_t, &cos_t);

    // Compute expensive squaring functions
    double cos2_t = cos_t * cos_t;
    double lambda_2_sq = y.lambda_2 * y.lambda_2;
    double phi_sq = y.phi * y.phi;

    // Evaluate RHS of the effective controlled pendulum dynamics
    dy.theta = y.phi;
    dy.phi = sin_t - alpha * y.phi - y.lambda_2 * cos2_t;
    dy.lambda_1 = -lambda_2_sq * cos_t * sin_t - y.lambda_2 * cos_t - sin_t;
    dy.lambda_2 = -y.phi - y.lambda_1 + alpha * y.lambda_2;
    dy.cost = 1.0 - cos_t + 0.5 * phi_sq + 0.5 * lambda_2_sq * cos2_t;

    return dy;
}

__host__ __device__
double evaluate_hamiltonian(const StateVec& y, const double alpha) {
    // CUDA Hardware Optimization: Compute expensive sin & cos simultaneously!
    double sin_t, cos_t;
    sincos(y.theta, &sin_t, &cos_t);

    // Compute expensive squaring functions
    double l2_sq = y.lambda_2 * y.lambda_2;
    double phi_sq = y.phi * y.phi;
    double cos_t_sq = cos_t * cos_t;

    // Evaluate Hamiltonian
    double hamiltonian = 1 - cos_t + 0.5*phi_sq - 0.5*l2_sq*cos_t_sq + y.lambda_1*y.phi + y.lambda_2*(sin_t - alpha*y.phi);
    return hamiltonian;
}

__host__ __device__
double wrap_theta(const double theta) {
    double sin_t, cos_t;
    sincos(theta, &sin_t, &cos_t);
    return std::atan2(sin_t, cos_t); // Wraps to (-pi, pi]
}

__host__ __device__
StateVec rk4_step(const StateVec& y, const double dt, const double alpha) {
    StateVec k1, k2, k3, k4;

    k1 = evaluate_derivatives(y, alpha);                        // Step 1: k1 = f(y)
    k2 = evaluate_derivatives(y + ((0.5 * dt) * k1), alpha);    // Step 2: k2 = f(y + dt/2 * k1)
    k3 = evaluate_derivatives(y + ((0.5 * dt) * k2), alpha);    // Step 3: k3 = f(y + dt/2 * k2)
    k4 = evaluate_derivatives(y + (dt * k3), alpha);            // Step 4: k4 = f(y + dt * k3)
    return y + ((k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0));  // Final step: y_next = y + dt/6 * (k1 + 2k2 + 2k3 + k4)
}

__host__ __device__
bool crossed_target_plane(double target_theta, double prev_theta, double curr_theta) {
    // Only trigger if pendulum is moving AWAY from origin towards target_theta
    // (backwards time -> approaching target_theta from stable equilibrium/origin)
    bool sign_flip = (target_theta - prev_theta) * (target_theta - curr_theta) < 0; // Crossed target plane theta value
    bool moving_out = std::abs(curr_theta) > std::abs(prev_theta);                  // Moving away from stable equilibrium/origin
    return sign_flip && moving_out;
}

__host__ __device__
double get_crossing_interpolation_factor(double target, double prev, double curr) {
    // f = 0.0 means crossing happened exactly at prev
    // f = 1.0 means crossing happened exactly at curr
    return (target - prev) / (curr - prev);
}

__global__
void backward_rk4_kernel(BackwardSweepParams p, BackwardsTimeDeviceArrays out) {
    // Check CUDA thread is in bounds
    int traj_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (traj_idx >= p.num_trajectories) return;

    // Get initial state for this thread
    StateVec current_state = p.seed_ring[traj_idx];
    StateVec prev_state = current_state;
    double current_t = 0.0;
    double prev_t = current_t;

    // DEBUG: Capture initial hamiltonian
    out.start_hamiltonians[traj_idx] = evaluate_hamiltonian(current_state, p.alpha);

    // Integration Loop
    for (long step = 0; step < p.num_timesteps; ++step) {
        // Take step (recall: p.dt is negative -> backwards in time).
        // Remember to wrap theta after the step is over for good usage later for distance detection.
        current_t += p.dt;
        current_state = rk4_step(current_state, p.dt, p.alpha);

        // Check if we have crossed the target-theta plane.
        if (crossed_target_plane(p.target_theta, prev_state.theta, current_state.theta)) {
            double f = get_crossing_interpolation_factor(p.target_theta, prev_state.theta, current_state.theta);
            
            StateVec hit_state = prev_state + f * (current_state - prev_state);
            double hit_time = prev_t + (f * p.dt);

            out.hit_points[traj_idx].state = hit_state;
            out.hit_points[traj_idx].time = hit_time;

            current_state = hit_state;
            current_t = hit_time;
            break;
        }
        prev_state = current_state;
        prev_t = current_t;
    }
    // DEBUG: Capture final Hamiltonian after integration loop to track drift
    out.end_hamiltonians[traj_idx] = evaluate_hamiltonian(current_state, p.alpha);
}

__global__
void forward_rk4_kernel(ForwardSweepParams p, ForwardsTimeDeviceArrays out) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x >= p.grid_size || idx_y >= p.grid_size) return;

    int tid = idx_y * p.grid_size + idx_x;

    // Get (lambda_1, lambda_2) initial guess gridpoint for this threead
    double l1 = p.l1_guess - p.search_radius + (2.0 * p.search_radius * idx_x) / (p.grid_size - 1);
    double l2 = p.l2_guess - p.search_radius + (2.0 * p.search_radius * idx_y) / (p.grid_size - 1);

    // Set up initial state of system before integrating forward in time
    StateVec current_state;
    current_state.theta = p.target_theta;
    current_state.phi = p.target_phi;
    current_state.lambda_1 = l1;
    current_state.lambda_2 = l2;
    current_state.cost = 0.0;

    // DEBUG: Capture initial hamiltonian
    out.start_hamiltonians[tid] = evaluate_hamiltonian(current_state, p.alpha);

    // Run forward in time
    for (long step = 0; step < p.num_timesteps; ++step) {
        current_state = rk4_step(current_state, p.dt, p.alpha);
    }

    // DEBUG: Capture final Hamiltonian after integration loop to track drift
    out.end_hamiltonians[tid] = evaluate_hamiltonian(current_state, p.alpha);

    // Store final state (to see how close we got to the origin)
    out.final_states[tid].state = current_state;
}

BackwardsTimeDeviceArrays allocate_device_arrays_backwards_time(int num_trajectories, std::vector<StateVec> h_seed_ring) {
    BackwardsTimeDeviceArrays d;

    // Use std::size_t for memory sizes!
    std::size_t float_array_size = (std::size_t)num_trajectories * sizeof(double);
    std::size_t trajectory_pt_array_size = (std::size_t)num_trajectories * sizeof(TrajectoryPoint);
    std::size_t seed_ring_array_size = (std::size_t)h_seed_ring.size() * sizeof(StateVec);

    gpuErrchk(cudaMalloc(&d.start_hamiltonians, float_array_size));
    gpuErrchk(cudaMalloc(&d.end_hamiltonians, float_array_size));
    gpuErrchk(cudaMalloc(&d.hit_points, trajectory_pt_array_size));
    
    gpuErrchk(cudaMalloc(&d.seed_ring, seed_ring_array_size));
    gpuErrchk(cudaMemcpy(d.seed_ring, h_seed_ring.data(), seed_ring_array_size, cudaMemcpyHostToDevice));

    return d;
}

ForwardsTimeDeviceArrays allocate_device_arrays_forwards_time(int num_trajectories) {
    ForwardsTimeDeviceArrays d;

    // Use std::size_t for memory sizes!
    std::size_t float_array_size = (std::size_t)num_trajectories * sizeof(double);
    std::size_t trajectory_pt_array_size = (std::size_t)num_trajectories * sizeof(TrajectoryPoint);

    gpuErrchk(cudaMalloc(&d.start_hamiltonians, float_array_size));
    gpuErrchk(cudaMalloc(&d.end_hamiltonians, float_array_size));
    gpuErrchk(cudaMalloc(&d.final_states, trajectory_pt_array_size));

    return d;
}

BackwardsTimeHostArrays copy_device_arrays_to_host_backwards_time(const BackwardsTimeDeviceArrays& d, int num_trajectories, long num_timesteps) {
    BackwardsTimeHostArrays h;

    std::size_t float_array_size = (std::size_t)num_trajectories * sizeof(double);
    std::size_t trajectory_pt_array_size = (std::size_t)num_trajectories * sizeof(TrajectoryPoint);
    
    // std::vector handles the CPU side allocation
    h.start_hamiltonians.resize(num_trajectories);
    h.end_hamiltonians.resize(num_trajectories);
    h.hit_points.resize(num_trajectories);

    gpuErrchk(cudaMemcpy(h.start_hamiltonians.data(), d.start_hamiltonians, float_array_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h.end_hamiltonians.data(), d.end_hamiltonians, float_array_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h.hit_points.data(), d.hit_points, trajectory_pt_array_size, cudaMemcpyDeviceToHost));
    
    return h;
}

ForwardsTimeHostArrays copy_device_arrays_to_host_forwards_time(const ForwardsTimeDeviceArrays& d, int num_trajectories) {
    ForwardsTimeHostArrays h;

    std::size_t float_array_size = (std::size_t)num_trajectories * sizeof(double);
    std::size_t trajectory_pt_array_size = (std::size_t)num_trajectories * sizeof(TrajectoryPoint);

    // std::vector handles the CPU side allocation
    h.start_hamiltonians.resize(num_trajectories);
    h.end_hamiltonians.resize(num_trajectories);
    h.final_states.resize(num_trajectories);

    gpuErrchk(cudaMemcpy(h.start_hamiltonians.data(), d.start_hamiltonians, float_array_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h.end_hamiltonians.data(), d.end_hamiltonians, float_array_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h.final_states.data(), d.final_states, trajectory_pt_array_size, cudaMemcpyDeviceToHost));

    return h;
}

void free_device_arrays_backwards_time(BackwardsTimeDeviceArrays& d) {
    // Free CUDA memory
    gpuErrchk(cudaFree(d.start_hamiltonians));
    gpuErrchk(cudaFree(d.end_hamiltonians));
    gpuErrchk(cudaFree(d.hit_points));
    gpuErrchk(cudaFree(d.seed_ring));
    
    // Nullify pointers to prevent accidental reuse
    d.start_hamiltonians = nullptr;
    d.end_hamiltonians = nullptr;
    d.hit_points = nullptr;
    d.seed_ring = nullptr;
}

void free_device_arrays_forwards_time(ForwardsTimeDeviceArrays& d) {
    // Free CUDA memory
    gpuErrchk(cudaFree(d.start_hamiltonians));
    gpuErrchk(cudaFree(d.end_hamiltonians));
    gpuErrchk(cudaFree(d.final_states));
    
    // Nullify pointers to prevent accidental reuse
    d.start_hamiltonians = nullptr;
    d.end_hamiltonians = nullptr;
    d.final_states = nullptr;
}

void compute_stable_eigenspace(const double alpha, StateVec& v1, StateVec& v2) {    
    // Build linearization matrix about origin
    Eigen::Matrix4d A;
    A << 0.0,  1.0,    0.0, 0.0,
         1.0,  -alpha, 0.0, -1.0,
	     -1.0, 0.0,    0.0, -1.0,
	     0.0,  -1.0,  -1.0, alpha;

    // Solve for eigenvalues/eigenvectors of A
    Eigen::EigenSolver<Eigen::Matrix4d> solver(A);
    Eigen::Vector4cd eigenvalues = solver.eigenvalues();
    Eigen::Matrix4cd eigenvectors = solver.eigenvectors();

    // Isolate stable manifold eigenvalues/eigenvectors
    Eigen::Matrix<std::complex<double>, 4, 2> Vs;    // Columns are stable eigenvectors
    std::size_t col = 0;
    bool is_real = true;
    for (std::size_t i = 0; i < 4; ++i) {
        if (eigenvalues(i).real() < 0 && col < 2) {
            Vs.col(col) = eigenvectors.col(i);
	        col++;
            if (std::abs(eigenvalues(i).imag()) >= 1e-9) {
                is_real = false;
            }
	    }
    }

    // Populate v1 and v2 with StateVec versions of these eigenvectors
    for (int i = 0; i < NUM_STATE_DIMS - 1; ++i) {
        v1[i] = Vs(i, 0).real();
        v2[i] = is_real ? Vs(i, 1).real() : Vs(i, 0).imag();
    }
}

std::vector<StateVec> generate_seed_ring(int num_trajectories, double r, double alpha, double center_angle, double angle_spread) {
    // Get span of stable eigenspace
    StateVec v1, v2;
    compute_stable_eigenspace(alpha, v1, v2);

    // Create seed ring from this eigenspace
    std::vector<StateVec> seed_ring(num_trajectories);
    for (int i = 0; i < num_trajectories; ++i) {
        // Map i to a specific angle within the spread
        double angle = center_angle - (angle_spread / 2.0) + (angle_spread * i) / (double)(num_trajectories - 1);
        seed_ring[i] = (v1 * std::cos(angle) + v2 * std::sin(angle)) * r;
    }
    return seed_ring;
}

TrajectoryPoint find_closest_point(const std::vector<TrajectoryPoint>& hit_points, double phi_target, int& best_idx) {
    best_idx = 0;
    double min_error = 1e18;
    for (int i = 0; i < hit_points.size(); ++i) {
        if (hit_points[i].time < 0) {
            double error = std::abs(hit_points[i].state.phi - phi_target);
            if (error < min_error) {
                min_error = error;
                best_idx = i;
            }
        }
    }
    return hit_points[best_idx];
}

void print_backwards_pass_iteration(const TrajectoryPoint& backwards_result, double angle_center, double angle_spread, int iteration) {
    std::printf("---- ITERATION %d ----\n", iteration);
    std::printf("* theta_0 = %.10f\n", backwards_result.state.theta);
    std::printf("* phi_0 = %.10f\n", backwards_result.state.phi);
    std::printf("* (lambda_1)_0 = %.10f\n", backwards_result.state.lambda_1);
    std::printf("* (lambda_2)_0 = %.10f\n", backwards_result.state.lambda_2);
    std::printf("* cost = %.10f\n\n", -backwards_result.state.cost);
    std::printf("* time of optimal trajectory = %.10f\n", -backwards_result.time);
    std::printf("* new optimal angle = %.10f\n", angle_center);
    std::printf("* new angle spread = %.10f\n", angle_spread);
}

TrajectoryPoint backwards_pass(double theta, double phi, double alpha) {
    const double DT = -0.005;             // Timestep size (negative since running in backwards time)
    const double T_MAX = -40.0;           // Max NEGATIVE time to run to
    const int NUM_TRAJECTORIES = 1000;    // Number of trajectories to shoot off at each iteration
    const double INITIAL_RADIUS = 1e-5;   // Radius of initial states about origin
    
    // Set up parameters for backwards sweep
    BackwardSweepParams p;
    p.alpha = alpha;
    p.target_theta = theta;
    p.target_phi = phi;
    p.dt = DT;
    p.num_timesteps = (long)(T_MAX/DT) + 1;
    p.num_trajectories = NUM_TRAJECTORIES;

    // Zoom parameters
    double current_center_angle = M_PI; // Start looking straight across the circle
    double current_angle_spread = 2.0 * M_PI; // Start by searching the WHOLE circle
    const int NUM_ZOOM_ITERS = 5;
    const double SHRINK_FACTOR = 0.1; // Shrink the 1D search space by 90% each iteration

    TrajectoryPoint best_point;
    int best_idx;
    for (int iter = 0; iter < NUM_ZOOM_ITERS; ++iter) {
        // 1. Generate the seed ring for the current zoom level
        std::vector<StateVec> h_seed_ring = generate_seed_ring(p.num_trajectories, INITIAL_RADIUS, p.alpha, current_center_angle, current_angle_spread);

        // 2. Allocate and run
        BackwardsTimeDeviceArrays d = allocate_device_arrays_backwards_time(p.num_trajectories, h_seed_ring);
        p.seed_ring = d.seed_ring;

        int threadsPerBlock = 256;
        int blocksPerGrid = (p.num_trajectories + threadsPerBlock - 1) / threadsPerBlock;
        backward_rk4_kernel<<<blocksPerGrid, threadsPerBlock>>>(p, d);
        gpuErrchk(cudaDeviceSynchronize());

        // 3. Download results to host
        BackwardsTimeHostArrays h = copy_device_arrays_to_host_backwards_time(d, p.num_trajectories, p.num_timesteps);
        free_device_arrays_backwards_time(d);

        // 4. Find closest point to target phi when it crosses
        best_point = find_closest_point(h.hit_points, p.target_phi, best_idx);

        // 5. Update zoom parameters to "shrink in" on the angle giving the optimal trajectory at the next iteration
        current_center_angle = current_center_angle - (current_angle_spread / 2.0) + (current_angle_spread * best_idx) / (double)(p.num_trajectories - 1);
        current_angle_spread *= SHRINK_FACTOR;

        print_backwards_pass_iteration(best_point, current_center_angle, current_angle_spread, iter + 1);
    }
    return best_point;
}

TrajectoryPoint forwards_pass(TrajectoryPoint backwards_seed, double target_theta, double target_phi, double alpha) {
    const double T_MAX = -backwards_seed.time;             // Symmery of forwards & backwards trajectories
    const double DT = 0.005;
    long num_timesteps = (long)(T_MAX/DT) + 1;

    const double GRID_SIZE = 255;                          // Number of cells to search
    const int NUM_MICROSCOPING_ITERATIONS = 10;            // Number of times to zoom in
    const double SHRINK_FACTOR = 0.5;                      // Halve the size of the box we search each time
    
    double phi_err = std::abs(backwards_seed.state.phi - target_phi);
    const double SEARCH_RADIUS = 0.05;                     // TODO: DO I MAKE THIS DEPEND ON THE ERROR IN PHI?
    
    // Configure microscoping parameters 
    ForwardSweepParams p;
    p.alpha = alpha;
    p.target_theta = target_theta;
    p.target_phi = target_phi;
    p.l1_guess = backwards_seed.state.lambda_1;
    p.l2_guess = backwards_seed.state.lambda_2;
    p.dt = DT;
    p.search_radius = SEARCH_RADIUS;
    p.grid_size = GRID_SIZE;
    p.num_timesteps = num_timesteps;

    // Allocate memory on GPU
    ForwardsTimeDeviceArrays d = allocate_device_arrays_forwards_time(p.grid_size * p.grid_size);
    ForwardsTimeHostArrays h;
    TrajectoryPoint best_point;
    for (int i = 1; i <= NUM_MICROSCOPING_ITERATIONS; ++i) {
        // 1. Run forward pass 
        dim3 threadsPerBlock(16, 16); // 256 threads total
        dim3 blocksPerGrid((p.grid_size + 15) / 16, (p.grid_size + 15) / 16);
        forward_rk4_kernel<<<blocksPerGrid, threadsPerBlock>>>(p, d);

        // 2. Sync and Check for Errors
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // 3. Download results to Host
        h = copy_device_arrays_to_host_forwards_time(d, p.grid_size * p.grid_size);

        // 4. Find "best point" (closest to origin)
        double min_score = 1e40; // Renamed from min_dist
        int best_tid = 0;

        for (int j = 0; j < (p.grid_size * p.grid_size); ++j) {
            StateVec final_s = h.final_states[j].state;
            
            // Calculate L2 distance to origin in state space
            double dist = l2_norm(final_s.theta, final_s.phi);
            
            // Calculate how badly this trajectory violates the H=0 manifold rule
            // (We evaluate at final_s, but H is constant along the true path)
            double H_penalty = std::abs(h.start_hamiltonians[j]);
            
            // Soft constraint: The trajectory must reach the origin AND maintain H=0
            double score = dist + 1e5 * H_penalty; 

            if (score < min_score) {
                min_score = score;
                best_tid = j;
            }
        }

        // 5. Recover the initial costates (L1, L2) that produced this winner
        int best_idx_x = best_tid % p.grid_size;
        int best_idx_y = best_tid / p.grid_size;
        double winner_l1 = p.l1_guess - p.search_radius + (2.0 * p.search_radius * best_idx_x) / (p.grid_size - 1);
        double winner_l2 = p.l2_guess - p.search_radius + (2.0 * p.search_radius * best_idx_y) / (p.grid_size - 1);

        // Store the full point to return later
        best_point.state = h.final_states[best_tid].state;
        best_point.state.lambda_1 = winner_l1; // Use the initial lambda, not the final one!
        best_point.state.lambda_2 = winner_l2;
        best_point.time = T_MAX;
        
        std::printf("Iteration %d: l1 = %.10f ; l2 = %.10f; |H| = %.10f\n", i, winner_l1, winner_l2, std::abs(h.start_hamiltonians[best_tid]));

        // Update for next iteration
        p.search_radius *= SHRINK_FACTOR;
        p.l1_guess = winner_l1;
        p.l2_guess = winner_l2;
    }

    // Free CUDA memory
    free_device_arrays_forwards_time(d);

    return best_point;
}

Result solve(double theta, double phi, double alpha) {
    Result res;
    
    // Run backward pass
    std::printf("=================  BACKWARDS PASS  =================\n");
    TrajectoryPoint backwards_result = backwards_pass(theta, phi, alpha);

    // Run forward pass
    std::printf("\n=================  FORWARDS PASS  =================\n");
    TrajectoryPoint forwards_result = forwards_pass(backwards_result, theta, phi, alpha);

    std::printf("\n-------- FINAL RESULT -------\n");
    std::printf("* theta_0 = %.10f\n", forwards_result.state.theta);
    std::printf("* phi_0 = %.10f\n", forwards_result.state.phi);
    std::printf("* (lambda_1)_0 = %.10f\n", forwards_result.state.lambda_1);
    std::printf("* (lambda_2)_0 = %.10f\n", forwards_result.state.lambda_2);
    std::printf("* cost = %.10f\n\n", forwards_result.state.cost);

    // Parse result
    res.l1 = forwards_result.state.lambda_1;
    res.l2 = forwards_result.state.lambda_2;
    res.cost = forwards_result.state.cost;
    return res;
}
