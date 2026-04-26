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

    // Unpack the current state vector (y)
    double theta    = y.data[0];
    double phi      = y.data[1];
    double lambda_1 = y.data[2];
    double lambda_2 = y.data[3];

    // Precompute expensive functions 
    double sin_t  = sin(theta);
    double cos_t  = cos(theta);
    double cos2_t = cos_t * cos_t;
    double lambda_2_sq = lambda_2 * lambda_2;
    double phi_sq = phi * phi;

    // Evaluate RHS of the effective controlled pendulum dynamics
    dy.data[0] = phi;
    dy.data[1] = sin_t - alpha * phi - lambda_2 * cos2_t;
    dy.data[2] = -lambda_2_sq * cos_t * sin_t - lambda_2 * cos_t - sin_t;
    dy.data[3] = -phi - lambda_1 + alpha * lambda_2;
    dy.data[4] = 1.0 - cos_t + 0.5 * phi_sq + 0.5 * lambda_2_sq * cos2_t;

    return dy;
}

__host__ __device__
double evaluate_hamiltonian(const StateVec& y, const double alpha) {
    double theta = y.data[0];
    double phi   = y.data[1];
    double l1    = y.data[2];
    double l2    = y.data[3];

    double sin_t = std::sin(theta);
    double cos_t = std::cos(theta);
    double l2_sq = l2 * l2;
    double phi_sq = phi * phi;
    double cos_t_sq = cos_t * cos_t;
    double hamiltonian = 1 - cos_t + 0.5*phi_sq - 0.5*l2_sq*cos_t_sq + l1*phi + l2*(sin_t - alpha*phi);
    return hamiltonian;
}

__host__ __device__
double wrap_theta(const double theta) {
    return std::atan2(std::sin(theta), std::cos(theta)); // Wraps to (-pi, pi]
}

__host__ __device__
StateVec get_initial_state(const SimulationParams& p, int i, int j) {
    // 1. Calculate the specific costate for this grid point
    double l1_init = (p.l1_init_guess - p.search_radius) + (i * p.costate_step_size);
    double l2_init = (p.l2_init_guess - p.search_radius) + (j * p.costate_step_size);

    // 2. Set the initial state (theta, phi, l1, l2, initial_cost)
    StateVec y;
    y.data[0] = p.theta_init; 
    y.data[1] = p.phi_init;   
    y.data[2] = l1_init;      
    y.data[3] = l2_init;      
    y.data[4] = 0.0;          // Cost starts at zero
    return y;
}

__host__ __device__
StateVec rk4_step(const StateVec& y, const double dt, const double alpha) {
    // Create arrays to hold intermediate/final values 
    StateVec k1, k2, k3, k4, y_next;

    // Step 1: k1 = f(y)
    k1 = evaluate_derivatives(y, alpha);

    // Step 2: k2 = f(y + dt/2 * k1)
    k2 = evaluate_derivatives(y + ((0.5 * dt) * k1), alpha);

    // Step 3: k3 = f(y + dt/2 * k2)
    k3 = evaluate_derivatives(y + ((0.5 * dt) * k2), alpha);

    // Step 4: k4 = f(y + dt * k3)
    k4 = evaluate_derivatives(y + (dt * k3), alpha);

    // Final step: y_next = y + dt/6 * (k1 + 2k2 + 2k3 + k4)
    y_next = y + ((k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0));
    return y_next;
}

__host__ __device__
StateVec run_simulation(const StateVec& y0, const SimulationParams& p) {
    // Run simulation with a given set of initial conditions
    StateVec current_state = y0;

    // Run RK4 on this set of initial conditions
    for (long i = 0; i < p.num_timesteps; ++i) {
        current_state = rk4_step(current_state, p.dt, p.alpha);
    }

    return current_state;
}

__host__ __device__
void package_pre_run_parameters(const StateVec& y0, const SimulationParams& p,
                                const int i, const int j,
                                DeviceArrays& out_arrays) {
    int out_array_idx = i * p.grid_size + j;    // Array index for results storage

    out_arrays.start_hamiltonians[out_array_idx] = evaluate_hamiltonian(y0, p.alpha);   // Initial hamiltonian value for this run
    out_arrays.l1s[out_array_idx]                = y0.data[2];                          // Initial lambda_1 value for this run
    out_arrays.l2s[out_array_idx]                = y0.data[3];                          // Initial lambda_2 value for this run
}

__host__ __device__
void package_post_run_parameters(const StateVec& y_final, const SimulationParams& p,
                                 const int i, const int j,
                                 DeviceArrays& out_arrays) {
    int out_array_idx = i * p.grid_size + j;    // Array index for results storage
    
    out_arrays.end_hamiltonians[out_array_idx] = evaluate_hamiltonian(y_final, p.alpha); // Final Hamiltonian
    out_arrays.costs[out_array_idx]           = y_final.data[4];                        // Accumulated cost
    out_arrays.thetas[out_array_idx]          = wrap_theta(y_final.data[0]);            // Final angle
    out_arrays.phis[out_array_idx]            = y_final.data[1];                        // Final angular velocity 
}

__global__
void shooting_kernel(SimulationParams p, DeviceArrays out_arrays) {
    // Calculate 2D grid coordinates for this specific thread
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // Only run kernel if we are in bounds of search grid
    if (i < p.grid_size && j < p.grid_size) {
        StateVec y = get_initial_state(p, i, j);
        package_pre_run_parameters(y, p, i, j, out_arrays);
        y = run_simulation(y, p);
        package_post_run_parameters(y, p, i, j, out_arrays);
    }
}


void compute_lqr_guess(const double theta, const double phi, const double alpha, double& l1_guess, double& l2_guess) {
    double theta_wrapped = std::atan2(std::sin(theta), std::cos(theta));  // Wraps to (-pi, pi]
    
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

    // Isolate stable manifold
    Eigen::Matrix<std::complex<double>, 4, 2> Vs;    // Columns are stable eigenvectors
    
    std::size_t col = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        if (eigenvalues(i).real() < 0 && col < 2) {
            Vs.col(col) = eigenvectors.col(i);
	        col++;
	    }
    }

    // Partition stable manifold to state/costate components 
    Eigen::Matrix2cd Vs_state = Vs.topRows<2>();
    Eigen::Matrix2cd Vs_costate = Vs.bottomRows<2>();

    // Compute projection/Ricatti matrix P mapping costates as linear combinations of states 
    Eigen::Matrix2cd P = Vs_costate * Vs_state.inverse();

    // Given initial state vector, get initial costate vector
    Eigen::Vector2d initial_states(theta_wrapped, phi);
    Eigen::Vector2cd initial_costates = P * initial_states.cast<std::complex<double>>();

    // The math guarantees imaginary parts cancel out
    l1_guess = initial_costates(0).real();
    l2_guess = initial_costates(1).real();

}

DeviceArrays allocate_device_arrays(int array_memory_size) {
    DeviceArrays d;

    gpuErrchk(cudaMalloc(&d.costs, array_memory_size));
    gpuErrchk(cudaMalloc(&d.thetas, array_memory_size));
    gpuErrchk(cudaMalloc(&d.phis, array_memory_size));
    gpuErrchk(cudaMalloc(&d.l1s, array_memory_size));
    gpuErrchk(cudaMalloc(&d.l2s, array_memory_size));
    gpuErrchk(cudaMalloc(&d.start_hamiltonians, array_memory_size));
    gpuErrchk(cudaMalloc(&d.end_hamiltonians, array_memory_size));

    return d;
}

HostArrays copy_device_arrays_to_host(const DeviceArrays& d, const int num_array_elements,
                                      const int array_memory_size) {
    HostArrays h;
    h.costs = std::vector<double>(num_array_elements);
    h.start_hamiltonians = std::vector<double>(num_array_elements);
    h.end_hamiltonians = std::vector<double>(num_array_elements);
    h.thetas = std::vector<double>(num_array_elements);
    h.phis = std::vector<double>(num_array_elements);
    h.l1s = std::vector<double>(num_array_elements);
    h.l2s = std::vector<double>(num_array_elements);

    gpuErrchk(cudaMemcpy(h.costs.data(), d.costs, array_memory_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h.start_hamiltonians.data(), d.start_hamiltonians, array_memory_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h.end_hamiltonians.data(), d.end_hamiltonians, array_memory_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h.thetas.data(), d.thetas, array_memory_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h.phis.data(), d.phis, array_memory_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h.l1s.data(), d.l1s, array_memory_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h.l2s.data(), d.l2s, array_memory_size, cudaMemcpyDeviceToHost));

    return h;
}

void free_device_arrays(DeviceArrays& d) {
    gpuErrchk(cudaFree(d.costs));
    gpuErrchk(cudaFree(d.start_hamiltonians));
    gpuErrchk(cudaFree(d.end_hamiltonians));
    gpuErrchk(cudaFree(d.thetas));
    gpuErrchk(cudaFree(d.phis));
    gpuErrchk(cudaFree(d.l1s));
    gpuErrchk(cudaFree(d.l2s));
}

ContinuationResult find_best_guess(const HostArrays& h, const int num_array_elements) {
    ContinuationResult res;
    res.r.cost = 1e100;
    double best_score = 1e100;

    for (int k = 0; k < num_array_elements; ++k) {
        double trajectory_cost = h.costs[k]; 
        double abs_H_start = std::abs(h.start_hamiltonians[k]);
        
        // Massive penalty ensures H=0 is a hard constraint at all scales
        double score = trajectory_cost + (1e9 * abs_H_start); 
        
        if (score < best_score) {
            best_score = score;
            
            res.r.cost = trajectory_cost; 
            res.r.l1 = h.l1s[k];
            res.r.l2 = h.l2s[k];
            res.min_abs_H = abs_H_start;
        }
    }
    return res;
}
ContinuationResult continuation_core(const SimulationParams& p) {
    int num_array_elements = p.grid_size * p.grid_size;
    int array_memory_size = num_array_elements * sizeof(double);

    DeviceArrays d = allocate_device_arrays(array_memory_size);

    // Configure Kernel Dimensions (Blocks and Threads)
    // We use 16x16 threads per block (256 threads total per block)
    dim3 threadsPerBlock(16, 16);
    std::size_t num_containers = (p.grid_size + 15) / 16;
    dim3 numBlocks(num_containers, num_containers);

    // Launch CUDA Kernel & run shooting method in parallel
    shooting_kernel<<<numBlocks, threadsPerBlock>>>(p, d);
    gpuErrchk(cudaPeekAtLastError());   // Check for launch errors
    gpuErrchk(cudaDeviceSynchronize()); // Check for execution errors & wait for GPU to finish

    // Copy results to host and cleanup CUDA memory
    HostArrays h = copy_device_arrays_to_host(d, num_array_elements, array_memory_size);
    free_device_arrays(d);

    // Return solution with minimum absolute value of final hamiltonian
    return find_best_guess(h, num_array_elements);
}

SimulationParams get_simulation_params(double theta_init, double phi_init, double alpha) {
    const double T_MAX = 5.0;                    // Artificial final time
    const double DT = 0.001;                      // Step size
    const std::size_t GRID_SIZE = 127;            // Number of gridpoints in 1 dimension (should be odd)

    // Set up initial simulation parameters
    SimulationParams p;

    p.theta_init = theta_init;   // Initial theta (-pi, pi]  -> TODO: Continuation uses "easier" guess
    p.phi_init = phi_init;                   // Initial phi/angular vel. -> TODO: Continuation uses "easier" guess
    p.alpha = alpha;
    p.dt = DT;
    p.num_timesteps = static_cast<long>(T_MAX / DT);
    p.grid_size = GRID_SIZE;
    p.search_radius = std::max(0.1, std::abs(p.theta_init) + std::abs(p.phi_init));	
    p.costate_step_size = (p.grid_size > 1) ? (2.0 * p.search_radius) / (p.grid_size - 1) : 0;
    
    // Compute guess for costate initial conditions using LQR (store as parameters in p)
    compute_lqr_guess(p.theta_init, p.phi_init, p.alpha, p.l1_init_guess, p.l2_init_guess);
    return p;
}

void apply_predictor_and_grid(SimulationParams& p, double prev_l1, double prev_l2, 
                              double prev_delta_l1, double prev_delta_l2, 
                              double prev_ds, double actual_ds) {
    // 1. Calculate the derivatives with respect to the path length
    double dl1_ds = (prev_ds > 0) ? (prev_delta_l1 / prev_ds) : 0.0;
    double dl2_ds = (prev_ds > 0) ? (prev_delta_l2 / prev_ds) : 0.0;
    
    double max_derivative = linfty_norm(dl1_ds, dl2_ds);
    
    // 2. Scale the search radius by the derivative to catch acceleration
    p.search_radius = std::max(0.05, 3.0 * max_derivative * actual_ds);
    p.costate_step_size = (p.grid_size > 1) ? (2.0 * p.search_radius) / (p.grid_size - 1) : 0;

    // 3. Predict the next costates using a first-order Euler step
    p.l1_init_guess = prev_l1 + (dl1_ds * actual_ds);
    p.l2_init_guess = prev_l2 + (dl2_ds * actual_ds);
}

double adapt_step_size(double current_ds, double min_abs_H, double min_step_size, bool& accepted) {
    const double SAFETY_FACTOR = 0.9;
    const double H_ACCEPTANCE_THRESHOLD = 0.05;

    if (min_abs_H > H_ACCEPTANCE_THRESHOLD) { 
        // REJECT: We fell off the stable manifold.
        accepted = false;
        double optimal_ds = current_ds * SAFETY_FACTOR * std::sqrt(0.05 / min_abs_H);
        
        // Clamp the shrinkage: don't shrink by more than 10x in a single rejection
        return std::max(min_step_size, std::max(0.1 * current_ds, optimal_ds));
    } 
    else {
        // ACCEPT: We successfully found the stable manifold.
        accepted = true;
        
        // ADAPTIVE GROWTH: Use asymptotic scaling to safely grow the step size
        if (min_abs_H < 0.005 && min_abs_H > 0.0) {
            double optimal_ds = current_ds * SAFETY_FACTOR * std::sqrt(0.05 / min_abs_H);
            return std::min(0.2, std::min(2.0 * current_ds, optimal_ds)); 
        }
        
        return current_ds;
    }
}

ContinuationResult run_microscope_refinement(SimulationParams p, double prev_delta_l1, 
                                             double prev_delta_l2, double prev_ds) {
    const int NUM_CONTINUATION_STEPS = 8;

    // Start refinement with the final, derivative-scaled search radius
    double dl1_ds = (prev_ds > 0) ? (prev_delta_l1 / prev_ds) : 0.0;
    double dl2_ds = (prev_ds > 0) ? (prev_delta_l2 / prev_ds) : 0.0;
    double final_max_derivative = linfty_norm(dl1_ds, dl2_ds);
    
    p.search_radius = std::max(0.05, 3.0 * final_max_derivative * prev_ds); 
    
    ContinuationResult current_res;

    // Zoom in 8 times, shrinking by a mathematically safe factor of 0.4 each time
    std::printf("Starting Microscope Refinement...\n");
    for (int refine = 1; refine <= NUM_CONTINUATION_STEPS; ++refine) {
        p.search_radius = p.search_radius * 0.4; 
        p.costate_step_size = (p.grid_size > 1) ? (2.0 * p.search_radius) / (p.grid_size - 1) : 0;
        
        current_res = continuation_core(p);

        std::printf("Refinement %d: Min |H| = %f\n", refine, current_res.min_abs_H);
        
        // Re-center perfectly on the newly refined best guess
        p.l1_init_guess = current_res.r.l1;
        p.l2_init_guess = current_res.r.l2;
    }

    return current_res;
}

ContinuationResult run_continuation(double target_theta, double target_phi, double alpha) {
    target_theta = wrap_theta(target_theta);
    
    double ds = 0.1; 
    const double MIN_STEP_SIZE = 1e-6; 
    
    double current_theta = 0.0;
    double current_phi   = 0.0;

    SimulationParams p = get_simulation_params(0.0, 0.0, alpha);
    
    double prev_l1 = p.l1_init_guess;
    double prev_l2 = p.l2_init_guess;
    double prev_delta_l1 = 0.0;
    double prev_delta_l2 = 0.0;
    double prev_ds = ds;

    // March forward until we physically reach the target state
    ContinuationResult current_res;
    double max_norm_distance = linfty_norm(current_theta - target_theta, current_phi - target_phi);
    while (max_norm_distance > 1e-6) {
        double dist_to_target = l2_norm(current_theta - target_theta, current_phi - target_phi);

        double actual_ds = std::min(ds, dist_to_target);    // Truncate to not overstep target, if necessary.
        double ratio = actual_ds / dist_to_target;
        
        p.theta_init = current_theta + ratio * (target_theta - current_theta);
        p.phi_init   = current_phi + ratio * (target_phi - current_phi);

        apply_predictor_and_grid(p, prev_l1, prev_l2, prev_delta_l1, prev_delta_l2, prev_ds, actual_ds); // Creates initial guess/grid mesh sizing based on how steep stable manifold is

        current_res = continuation_core(p);     // Run grid search to find points with minimal hamiltonian
        
        std::printf("L-Infinity Distance from Target: %f ; Step size ds = %f\n", max_norm_distance, actual_ds);
        std::printf("  |H| = %f\n", current_res.min_abs_H);

        // Trust Region Evaluation
        bool step_accepted = false;
        ds = adapt_step_size(ds, current_res.min_abs_H, MIN_STEP_SIZE, step_accepted);
        
        if (!step_accepted) {
            // Step rejected! Either asymptote reached, or we will retry with a smaller dt.
            if (ds <= MIN_STEP_SIZE) { 
                // Safety Net: Asymptote reached
                std::fprintf(stderr, "ERROR: Minimum step size insufficient for stability. Asymptote Reached. Exiting and returning last known good state...\n");
                break;
            } 
            std::printf("  Step size of ds=%f rejected. Retrying with smaller ds...\n", actual_ds);
            continue; // Retry with smaller ds
        } 
        
        // Step Accepted: Update our current position and state history
        std::printf("  Step size of ds=%f accepted! Moving on...\n", actual_ds);
        current_theta = p.theta_init;
        current_phi   = p.phi_init;
        
        prev_delta_l1 = current_res.r.l1 - prev_l1;
        prev_delta_l2 = current_res.r.l2 - prev_l2;
        
        prev_l1 = current_res.r.l1;
        prev_l2 = current_res.r.l2;
        prev_ds = actual_ds;

        max_norm_distance = linfty_norm(current_theta - target_theta, current_phi - target_phi);
    }    

    // Hand off to the microscope loop for the final polish
    return run_microscope_refinement(p, prev_delta_l1, prev_delta_l2, prev_ds);
}


Result solve(double theta, double phi, double alpha) {
    // Solve using continuation
    ContinuationResult res = run_continuation(theta, phi, alpha);
    return res.r;
}
