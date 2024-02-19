#include <casadi/casadi.hpp>
#include <iostream>
using namespace casadi;
 
// Function to solve optimization with custom x_min and x_max
void solveOptimization(double x_min, double x_max) {
    // Create an Opti instance
    Opti opti;
 
    // Define the optimization variable
    MX x = opti.variable();
    
 
    // Define the objective function
    opti.minimize(pow(x-2, 2));
 
    // Apply custom constraints
    opti.subject_to(x >= x_min);
    opti.subject_to(x <= x_max);
 
    // Set the solver and options
    opti.solver("ipopt", {{"verbose", true}});
 
    // Solve the optimization problem
    OptiSol sol = opti.solve();
 
    // Retrieve and print the optimal value
    double x_opt = static_cast<double>(sol.value(x));
    std::cout << "Optimal x with constraints x >= " << x_min
              << " and x <= " << x_max << ": " << x_opt << std::endl;
}
 
int main() {
    // Custom x_min and x_max values
    double x_min = 50;
    double x_max = 70;
 
    // Solve the optimization with custom constraints
    solveOptimization(x_min, x_max);
 
    return 0;
}
 