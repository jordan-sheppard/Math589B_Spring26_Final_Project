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
void backward_rk4_kernel(BackwardSweepParams p, DeviceArrays out) {
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

DeviceArrays allocate_device_arrays(int num_trajectories, std::vector<StateVec> h_seed_ring) {
    DeviceArrays d;

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

HostArrays copy_device_arrays_to_host(const DeviceArrays& d, int num_trajectories, long num_timesteps) {
    HostArrays h;

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

void free_device_arrays(DeviceArrays& d) {
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

std::vector<StateVec> generate_seed_ring(int num_trajectories, double r, double alpha) {
    // Get span of stable eigenspace
    StateVec v1, v2;
    compute_stable_eigenspace(alpha, v1, v2);

    // Create seed ring from this eigenspace
    std::vector<StateVec> seed_ring(num_trajectories);
    for (int i = 0; i < num_trajectories; ++i) {
        double angle = (2.0 * M_PI * i) / num_trajectories;
        seed_ring[i] = (v1 * std::cos(angle) + v2 * std::sin(angle)) * r;
    }
    return seed_ring;
}

TrajectoryPoint find_closest_point(const std::vector<TrajectoryPoint>& hit_points, double phi_target) {
    int best_idx = 0;
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


TrajectoryPoint backwards_pass(double theta, double phi, double alpha) {
    const double DT = -0.005;            // Timestep size (negative since running in backwards time)
    const double T_MAX = -5.0;           // Max NEGATIVE time to run to
    const int NUM_TRAJECTORIES = 1000;   // Number of trajectories to shoot off
    const double INITIAL_RADIUS = 1e-4;  // Radius of initial states about origin
    
    // Set up parameters for backwards sweep
    BackwardSweepParams p;
    p.alpha = alpha;
    p.target_theta = theta;
    p.target_phi = phi;
    p.dt = DT;
    p.num_timesteps = (int)(T_MAX/DT) + 1;
    p.num_trajectories = NUM_TRAJECTORIES;

    // Generate backwards sweep seed ring on CPU, and initialize it (along with storage for outputs) on GPU
    std::vector<StateVec> h_seed_ring = generate_seed_ring(p.num_trajectories, INITIAL_RADIUS, p.alpha);
    DeviceArrays d = allocate_device_arrays(p.num_trajectories, h_seed_ring);
    p.seed_ring = d.seed_ring;      // CRITICAL: Link location of seed ring memory on GPU device as parameter

    // 1. Configure and Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (p.num_trajectories + threadsPerBlock - 1) / threadsPerBlock;
    backward_rk4_kernel<<<blocksPerGrid, threadsPerBlock>>>(p, d);

    // 2. Sync and Check for Errors
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // 3. Download results to Host and free device memory
    HostArrays h = copy_device_arrays_to_host(d, p.num_trajectories, p.num_timesteps);
    free_device_arrays(d);

    // 4. Select the "Best" trajectory (closest match to target_phi)
    TrajectoryPoint closest_point = find_closest_point(h.hit_points, p.target_phi);
    return closest_point;
}



Result solve(double theta, double phi, double alpha) {
    Result res;
    
    TrajectoryPoint backwards_result = backwards_pass(theta, phi, alpha);
    std::printf("-------- BEST RESULT -------\n");
    std::printf("* theta_0 = %f\n", backwards_result.state.theta);
    std::printf("* phi_0 = %f\n", backwards_result.state.phi);
    std::printf("* (lambda_1)_0 = %f\n", backwards_result.state.lambda_1);
    std::printf("* (lambda_2)_0 = %f\n", backwards_result.state.lambda_2);
    std::printf("* cost = %f\n\n", backwards_result.state.cost);

    res.l1 = backwards_result.state.lambda_1;
    res.l2 = backwards_result.state.lambda_2;
    res.cost = -backwards_result.state.cost;
    return res;
}
