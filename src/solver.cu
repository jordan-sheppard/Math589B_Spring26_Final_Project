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
StateVec run_simulation(const StateVec& y0, const double alpha, const double dt, const long num_timesteps) {
    // Run simulation with a given set of initial conditions
    StateVec current_state = y0;
    
    // Run RK4 on this set of initial conditions
    for (long i = 0; i < num_timesteps; ++i) {
        current_state = rk4_step(current_state, dt, alpha);
    }

    return current_state;
}

__global__
void shooting_kernel(SimulationParams p, double* out_costs,
		     double* out_l1s, double* out_l2s) {
    // Calculate 2D grid coordinates for this specific thread
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if this thread is in bounds for search grid
    if (i < p.grid_size && j < p.grid_size) {
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

        // 3. Run simulation using num_timesteps directly
        y = run_simulation(y, p.alpha, p.dt, p.num_timesteps);

        // 4. Record results in flat arrays
        int idx = i * p.grid_size + j;
        out_costs[idx] = y.data[4];       // Accumulated cost
        out_l1s[idx]   = l1_init;         // Initial lambda_1 value for this run
        out_l2s[idx]   = l2_init;         // Initial lambda_2 value for this run
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

Result solve(double theta, const double phi, const double alpha) {
    const double T_MAX = 5.0;                     // Artificial final time
    const double DT = 0.01;                       // Step size
    const std::size_t GRID_SIZE = 128;            // Number of gridpoints in 1 dimension
    
    // Set up simulation parameters
    SimulationParams p;
    p.theta_init = std::atan2(std::sin(theta), std::cos(theta));   // Standardizes theta to (-pi/2, pi/2]
    p.phi_init = phi;
    p.alpha = alpha;
    p.dt = DT;
    p.num_timesteps = static_cast<long>(T_MAX / DT);
    p.grid_size = GRID_SIZE;
    p.search_radius = std::max(0.1, std::abs(p.theta_init) + std::abs(p.phi_init));	
    p.costate_step_size = (p.grid_size > 1) ? (2.0 * p.search_radius) / (p.grid_size - 1) : 0;
    
    // Compute guess for costate initial conditions using LQR (store as parameters in p)
    compute_lqr_guess(p.theta_init, p.phi_init, p.alpha, p.l1_init_guess, p.l2_init_guess);

    // Allocate results
    std::size_t num_array_elements = p.grid_size * p.grid_size;
    std::size_t array_memory_size = num_array_elements * sizeof(double);


    // 1. Allocate GPU (Device) memory
    double *d_costs, *d_l1s, *d_l2s;
    cudaMalloc(&d_costs, array_memory_size);
    cudaMalloc(&d_l1s, array_memory_size);
    cudaMalloc(&d_l2s, array_memory_size);

    // 2. Configure Kernel Dimensions (Blocks and Threads)
    // We use 16x16 threads per block (256 threads total per block)
    dim3 threadsPerBlock(16, 16);
    std::size_t num_containers = (p.grid_size + 15) / 16;
    dim3 numBlocks(num_containers, num_containers);

    // 3. Launch CUDA Kernel to run shooting method in parallel
    shooting_kernel<<<numBlocks, threadsPerBlock>>>(p, d_costs, d_l1s, d_l2s);    
    
    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // 4. Copy results back to CPU (Host) memory
    std::vector<double> h_costs(num_array_elements);
    std::vector<double> h_l1s(num_array_elements);
    std::vector<double> h_l2s(num_array_elements);

    cudaMemcpy(h_costs.data(), d_costs, array_memory_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_l1s.data(), d_l1s, array_memory_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_l2s.data(), d_l2s, array_memory_size, cudaMemcpyDeviceToHost);

    // 5. Cleanup GPU memory
    cudaFree(d_costs);
    cudaFree(d_l1s);
    cudaFree(d_l2s);

    // 6: Find minimum cost parameters and return those.
    Result r;
    r.l1 = 1e100;
    r.l2 = 1e100;
    r.cost = 1e100;
    for (int k = 0; k < num_array_elements; ++k) {
        if (h_costs[k] < r.cost) {
            r.cost = h_costs[k];
            r.l1 = h_l1s[k];
            r.l2 = h_l2s[k];
        }
    }
    return r;
}
