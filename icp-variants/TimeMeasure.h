#pragma once
#include <vector>
#include <math.h>
#include "Eigen.h"
#include "utils.h"

class TimeMeasure
{
private:
    double iterSelectionTime;
    double iterMatchingTime;
    double iterWeighingTime;
    double iterRejectionTime;
    double iterSolverTime;
    double iterConvergenceTime;
    //TODO : Suggest better term for this variable

public:

    double selectionTime;       // Step 1 (single iteration)
    double matchingTime;        // Step 2
    double weighingTime;        // Step 3
    double rejectionTime;       // Step 4
    double solverTime;          // Step 5
    double convergenceTime;     // Average time
    unsigned int* nIterations;

    TimeMeasure() {
        nIterations = 0;
        rejectionTime = 0;
        selectionTime = 0;
        weighingTime = 0;
        matchingTime = 0;
        solverTime = 0;
        convergenceTime = 0;
    };

    /**
     * Compute elapsed time for the 4 steps
     * and the total time taken by the solver
     * at each iteration.
     */
    void calculateIterationTime() {
        iterSelectionTime = selectionTime;
        iterMatchingTime = matchingTime / *nIterations;
        iterWeighingTime = weighingTime / *nIterations;
        iterRejectionTime = rejectionTime / *nIterations;
        iterSolverTime = solverTime / *nIterations;
        iterConvergenceTime = convergenceTime / *nIterations;

        std::cout << 
            "Average Convergence time = " << iterConvergenceTime << " s per iteration\n" <<
            "Time taken for each step:\n" <<
            "\t [*] Selection time = " << iterSelectionTime << " s \n" <<
            "\t [*] Matching time = " << iterMatchingTime << " s per iteration\n" <<
            "\t [*] Weighing time = " << iterWeighingTime << " s per iteration\n" <<
            "\t [*] Rejection time = " << iterRejectionTime << " s per iteration\n" <<
            "\t [*] Solver time = " << iterSolverTime << " s per iteration\n"
            ;
    }

};
