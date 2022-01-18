#pragma once

// The Google logging library (GLOG), used in Ceres, has a conflict with Windows defined constants. This definitions prevents GLOG to use the same constants
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <flann/flann.hpp>

#include "SimpleMesh.h"
#include "NearestNeighbor.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"
#include "utils.h"
#include "constraints.h"
#include "selection.h"
#include "TimeMeasure.h"
#include "weighting.h"
#include "ConvergenceMeasure.h"

/**
 * ICP optimizer - Abstract Base Class
 */
class ICPOptimizer {
public:
    ICPOptimizer() :
        metric{ 0 }, selectionMethod{0}, rejectionMethod{1}, weightingMethod{0},
        m_nIterations{ 20 }, matchingMethod{0}, maxDistance{0.0003f}, colorICP{false}{ 
   
        if(matchingMethod == 0) 
            m_nearestNeighborSearch = std::make_unique<NearestNeighborSearchFlann>();
        else
            m_nearestNeighborSearch = std::make_unique<NearestNeighborSearchProjective>();
    }

    void setMatchingMaxDistance(float maxDistance) {
        m_nearestNeighborSearch->setMatchingMaxDistance(maxDistance);
        this->maxDistance = maxDistance;
    }

    void setMetric(unsigned int metric) {
        this->metric = metric;
    }

    void enableColorICP(bool colorICP){
        this->colorICP = colorICP;
    }

    void setSelectionMethod(unsigned int selectionMethod, double proba=1.0) {
        this->selectionMethod = selectionMethod;
        this->proba = proba;
    }

    void setRejectionMethod(unsigned int rejectionMethod) {
        this->rejectionMethod = rejectionMethod;
    }

    void setWeightingMethod(unsigned int weightingMethod) {
        this->weightingMethod = weightingMethod;
    }

    void setMatchingMethod(unsigned int matchingMethod) {
        this->matchingMethod = matchingMethod;

        if(matchingMethod == 0) 
            m_nearestNeighborSearch = std::make_unique<NearestNeighborSearchFlann>();
        else
            m_nearestNeighborSearch = std::make_unique<NearestNeighborSearchProjective>();
    }

    void setCameraParamsMatchingMethod(const Eigen::Matrix3f& depthIntrinsics, const unsigned width, const unsigned height) {
        m_nearestNeighborSearch->setCameraParams(depthIntrinsics, width, height);
    }

    void setNbOfIterations(unsigned nIterations) {
        m_nIterations = nIterations;
    }

    void setTimeMeasure(TimeMeasure& timeMeasure) {
        timeMeasure.nIterations = &m_nIterations;
        m_timeMeasure = &timeMeasure;
    }

    void setConvergenceMeasure(ConvergenceMeasure& convergenMearsure) {
        m_convergenceMeasure = &convergenMearsure;
    }

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose, bool calculateRMSE = true) = 0;

protected:
    unsigned int metric;
    bool colorICP;
    unsigned int selectionMethod;
    double proba;
    unsigned int rejectionMethod;
    unsigned int weightingMethod;
    unsigned int matchingMethod;
    unsigned m_nIterations;
    TimeMeasure* m_timeMeasure;
    ConvergenceMeasure *m_convergenceMeasure;
    float maxDistance; // Sqaure distance
    std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;

    void pruneCorrespondences(const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals, std::vector<Match>& matches) {
        const unsigned nPoints = sourceNormals.size();

        /*  Rads to radians - 60 deg threshold */
        double threshold = 60 * EIGEN_PI / 180.0;

        for (unsigned i = 0; i < nPoints; i++) {
            Match& match = matches[i];
            if (match.idx >= 0) {
                const auto& sourceNormal = sourceNormals[i];
                const auto& targetNormal = targetNormals[match.idx];

                // TODO: Invalidate the match (set it to -1) if the angle between the normals is greater than 60
                if(acos(sourceNormal.dot(targetNormal) / (sourceNormal.norm() * targetNormal.norm())) > threshold)
                   match.idx = -1; 
            }
        }
    }
};


/**
 * ICP optimizer - using Ceres for optimization.
 */
class CeresICPOptimizer : public ICPOptimizer {
public:
    CeresICPOptimizer() {}

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose, bool calculateRMSE = true) override {
        clock_t step_start, step_end, start, begin, end, tot_time;

        start = clock();

        step_start = clock();
        // 1. Selection step //
        // Initialize selection step //
        auto sourceSelection = PointSelection(source, selectionMethod, proba);
        m_timeMeasure->selectionTime += double(clock() - step_start) / CLOCKS_PER_SEC;


        // Initialize weighting step //
        auto weightingStep = WeightingMethod(this->weightingMethod, this->maxDistance);

        // Initialize matching step // 

        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        if(this->colorICP)
            m_nearestNeighborSearch->buildIndex(target.getPoints(), target.getColors());
        else
            m_nearestNeighborSearch->buildIndex(target.getPoints());

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        // We optimize on the transformation in SE3 notation: 3 parameters for the axis-angle vector of the rotation (its length presents
        // the rotation angle) and 3 parameters for the translation vector. 
        double incrementArray[6];
        auto poseIncrement = PoseIncrement<double>(incrementArray);
        poseIncrement.setZero();

        for (int i = 0; i < m_nIterations; ++i) {

            // Compute the matches.
            std::cout << "Matching points ..." << std::endl;
            begin = clock();

            // 1. Selection Step // 
            // Change source to sourceSelection to do selection.
            if (selectionMethod == RANDOM_SAMPLING) // Resample each iteration
                sourceSelection.resample();
            
            auto transformedPoints = transformPoints(sourceSelection.getPoints(), estimatedPose);
            auto transformedNormals = transformNormals(sourceSelection.getNormals(), estimatedPose);
            std::cout << "Number of source points to match = " << transformedPoints.size() << std::endl;

            step_start = clock();
            //2. Matching step //
            std::vector<Match> matches; 
            if(this->colorICP)
                matches = m_nearestNeighborSearch->queryMatches(transformedPoints, sourceSelection.getColors());
            else
                matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            
            m_timeMeasure->matchingTime += double(clock() - step_start) / CLOCKS_PER_SEC;
          
            step_start = clock();
            // 3. Weighting step // 
            weightingStep.applyWeights(transformedPoints, target.getPoints(), transformedNormals, target.getNormals(), 
                                       sourceSelection.getColors(), target.getColors(), matches);
            
            m_timeMeasure->weighingTime += double(clock() - step_start) / CLOCKS_PER_SEC;
            step_start = clock();
            // 4. Rejection step //
            if (rejectionMethod == 1)
                pruneCorrespondences(transformedNormals, target.getNormals(), matches);
            m_timeMeasure->rejectionTime += double(clock() - step_start) / CLOCKS_PER_SEC;

            // TODO : What to do with this part?
            end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

            step_start = clock();

            // Prepare point-to-point and point-to-plane constraints.
            ceres::Problem problem;
            if(metric == 0)
                prepareConstraintsPointICP(transformedPoints, target.getPoints(), target.getNormals(), matches, poseIncrement, problem);
            else if(metric == 1)
                prepareConstraintsPlaneICP(transformedPoints, target.getPoints(), target.getNormals(), matches, poseIncrement, problem);
            else if(metric == 2)
                prepareConstraintsSymmetricICP(transformedPoints, target.getPoints(), transformedNormals, target.getNormals(), matches, poseIncrement, problem);

            // Configure options for the solver.
            ceres::Solver::Options options;
            configureSolver(options);

            // Run the solver (for one iteration).
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;
            //std::cout << summary.FullReport() << std::endl;

            m_timeMeasure->solverTime += double(clock() - step_start) / CLOCKS_PER_SEC;

            // Update the current pose estimate (we always update the pose from the left, using left-increment notation).
            Matrix4f matrix = PoseIncrement<double>::convertToMatrix(poseIncrement);
            estimatedPose = PoseIncrement<double>::convertToMatrix(poseIncrement) * estimatedPose;
            poseIncrement.setZero();

            std::cout << "Optimization iteration done." << std::endl;

            // RMSE compute
            if (calculateRMSE) {
                m_convergenceMeasure->recordAlignmentError(estimatedPose);
            }
        }

        m_timeMeasure->convergenceTime += double(clock() - start) / CLOCKS_PER_SEC;

        // Store result
        initialPose = estimatedPose;
    }


private:
    void configureSolver(ceres::Solver::Options& options) {
        // Ceres options.
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = 1;
        options.max_num_iterations = 10;
        options.num_threads = 8;
    }

    void prepareConstraintsPointICP(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals, const std::vector<Match> matches, const PoseIncrement<double>& poseIncrement, ceres::Problem& problem) const {
        const unsigned nPoints = sourcePoints.size();

        for (unsigned i = 0; i < nPoints; ++i) {
            const auto match = matches[i];
            if (match.idx >= 0) {
                const auto& sourcePoint = sourcePoints[i];
                const auto& targetPoint = targetPoints[match.idx];

                if (!sourcePoint.allFinite() || !targetPoint.allFinite())
                    continue;

                // TODO: Create a new point-to-point cost function and add it as constraint (i.e. residual block) 
                // to the Ceres problem.

                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
                        new PointToPointConstraint(sourcePoint, targetPoint, match.weight)
                    ),
                    nullptr, poseIncrement.getData()
                );
            }
        }
    }

    void prepareConstraintsPlaneICP(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals, const std::vector<Match> matches, const PoseIncrement<double>& poseIncrement, ceres::Problem& problem) const {
        const unsigned nPoints = sourcePoints.size();

        for (unsigned i = 0; i < nPoints; ++i) {
            const auto match = matches[i];
            if (match.idx >= 0) {
                const auto& sourcePoint = sourcePoints[i];
                const auto& targetPoint = targetPoints[match.idx];
                
                if (!sourcePoint.allFinite() || !targetPoint.allFinite())
                    continue;

                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
                        new PointToPointConstraint(sourcePoint, targetPoint, match.weight)
                    ),
                    nullptr, poseIncrement.getData()
                );
                
                const auto& targetNormal = targetNormals[match.idx];

                if (!targetNormal.allFinite())
                    continue;

                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
                        new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, match.weight)
                    ),
                    nullptr, poseIncrement.getData()
                );

            }
        }
    }

    void prepareConstraintsSymmetricICP(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals, const std::vector<Match> matches, const PoseIncrement<double>& poseIncrement, ceres::Problem& problem) const {
        const unsigned nPoints = sourcePoints.size();
        std::cout << "Symmetric ICP Non-linear" << std::endl;
        
        for (unsigned i = 0; i < nPoints; ++i) {
            const auto match = matches[i];

            if(match.idx >= 0){
                const auto& sourcePoint = sourcePoints[i];
                const auto& targetPoint = targetPoints[match.idx];
                
                if (!sourcePoint.allFinite() || !targetPoint.allFinite())
                    continue;
                
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
                        new PointToPointConstraint(sourcePoint, targetPoint, match.weight)
                    ),
                    nullptr, poseIncrement.getData()
                );

                const auto& sourceNormal = sourceNormals[i];
                const auto& targetNormal = targetNormals[match.idx];

                if (!targetNormal.allFinite() || !sourceNormal.allFinite())
                    continue;               

                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<SymmetricConstraint, 1, 6>(
                        new SymmetricConstraint(sourcePoint, targetPoint, sourceNormal, targetNormal)
                        ),
                    nullptr, poseIncrement.getData()
                );
            }
        }
    }
};


/**
 * ICP optimizer - using linear least-squares for optimization.
 */
class LinearICPOptimizer : public ICPOptimizer {
public:
    LinearICPOptimizer() {}

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose, bool calculateRMSE = true) override {
        
        clock_t start = clock();
        // 1. Selection step //
        // Initialize selection step //
        auto sourceSelection = PointSelection(source, selectionMethod, proba);

        // Initialize weightingStep step //
        auto weightingStep = WeightingMethod(this->weightingMethod, this->maxDistance);
        
        // Change PointCloud source to PointSelection source
        
        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        if(this->colorICP)
            m_nearestNeighborSearch->buildIndex(target.getPoints(), target.getColors());
        else
            m_nearestNeighborSearch->buildIndex(target.getPoints());

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ..." << std::endl;
            clock_t begin = clock();

            // 1. Selection step //
            // Change source to sourceSelection to do selection.
            if (selectionMethod == RANDOM_SAMPLING) // Resample each iteration
                sourceSelection.resample();
            
            auto transformedPoints = transformPoints(sourceSelection.getPoints(), estimatedPose);
            auto transformedNormals = transformNormals(sourceSelection.getNormals(), estimatedPose);
            std::cout << "Number of source points to match = " << transformedPoints.size() << std::endl;

            start = clock();
            //2. Matching step //
            std::vector<Match> matches; 
            if(this->colorICP)
                matches = m_nearestNeighborSearch->queryMatches(transformedPoints, sourceSelection.getColors());
            else
                matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            
            m_timeMeasure->matchingTime += double(clock() - start) / CLOCKS_PER_SEC;

            start = clock();
            // 3. Weighting step // 
            weightingStep.applyWeights(transformedPoints, target.getPoints(), transformedNormals, target.getNormals(), 
                                       sourceSelection.getColors(), target.getColors(), matches);

            m_timeMeasure->weighingTime += double(clock() - start) / CLOCKS_PER_SEC;
            start = clock();
            // 4. Rejection step //
            if (rejectionMethod == 1)
                pruneCorrespondences(transformedNormals, target.getNormals(), matches);
            m_timeMeasure->rejectionTime += double(clock() - start) / CLOCKS_PER_SEC;

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;
            //TODO : Check if this measurement point is correct
            m_timeMeasure->convergenceTime += double(end - begin) / CLOCKS_PER_SEC;

            std::vector<Vector3f> sourcePoints;
            std::vector<Vector3f> targetPoints;
            std::vector<Vector3f> sourceNormals;
            std::vector<Vector3f> targetNormals;

            // Add all matches to the sourcePoints and targetPoints vector,
            // so that the sourcePoints[i] matches targetPoints[i]. For every source point,
            // the matches vector holds the index of the matching target point.
            for (int j = 0; j < transformedPoints.size(); j++) {
                const auto& match = matches[j];
                if (match.idx >= 0) {
                    sourcePoints.push_back(transformedPoints[j]);
                    targetPoints.push_back(target.getPoints()[match.idx]);
                    sourceNormals.push_back(transformedNormals[j]);
                    targetNormals.push_back(target.getNormals()[match.idx]);
                }
            }

            // Estimate the new pose
            if (metric == 1) {
                estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, targetNormals) * estimatedPose;
            }
            else if(metric == 0) {
                estimatedPose = estimatePosePointToPoint(sourcePoints, targetPoints) * estimatedPose;
            }
            else if(metric == 2) {
                estimatedPose = estimatePoseSymmetricICP(sourcePoints, targetPoints, sourceNormals, targetNormals) * estimatedPose;
            }

            std::cout << "Optimization iteration done." << std::endl;

            // RMSE compute
            if (calculateRMSE) {
                m_convergenceMeasure->recordAlignmentError(estimatedPose);
            }
        }

        // Store result
        initialPose = estimatedPose;
    }

private:
    Matrix4f estimatePosePointToPoint(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, bool calculateRMSE = true) {
        ProcrustesAligner procrustAligner;
        Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);

        return estimatedPose;
    }

    Matrix4f estimatePosePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals) {
        const unsigned nPoints = sourcePoints.size();

        // Build the system
        MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
        VectorXf b = VectorXf::Zero(4 * nPoints);

        for (unsigned i = 0; i < nPoints; i++) {
            const auto& s = sourcePoints[i];
            const auto& d = targetPoints[i];
            const auto& n = targetNormals[i];

            /* Use advance initialization to fill rows        */ 
            /* Use temporary eigen row vector                 */
            RowVectorXf planeConstraintRow(6);

            /* Derived from the paper - linear point-to-plane */
            planeConstraintRow << n[2]*s[1] - n[1]*s[2],
                                  n[0]*s[2] - n[2]*s[0],
                                  n[1]*s[0] - n[0]*s[1],
                                  n[0],
                                  n[1],
                                  n[2];
            
            /* Fix system */
            A.row(4*i) = planeConstraintRow;

            b(4*i) = n[0]*d[0] + n[1]*d[1] + n[2]*d[2] 
                     -
                     (n[0]*s[0] + n[1]*s[1] + n[2]*s[2]);

            /*  Ms = d -> find unkowns and free vars like in */
            /*  in the paper expansion                       */
            /*  So, add three rows. 1 per coordianate        */

            /* Second row */
            RowVectorXf pointConstraintRow(6);
            pointConstraintRow << 0, s[2], -s[1], 1.0, 0.0, 0.0; // a, b, g, tx, ty, tz
            
            A.row(4*i + 1) = pointConstraintRow;
            b(4*i + 1) = d[0] - s[0];

            /* Third row */
            pointConstraintRow << -s[2], 0, s[0], 0, 1.0, 0.0;
            
            A.row(4*i + 2) = pointConstraintRow;
            b(4*i + 2) = d[1] - s[1];

            /* Fourth row */
            pointConstraintRow << s[1], -s[0], 0.0, 0.0, 0.0, 1.0;
            
            A.row(4*i + 3) = pointConstraintRow;
            b(4*i + 3) = d[2] - s[2];
            
            float LAMBDA_POINT = 1.0f;
            float LAMBDA_PLANE = 1.0f;
            
            A.row(4*i) *= LAMBDA_PLANE;
            b(4*i) *= LAMBDA_PLANE;
        
            A.row(4*i + 1) *= LAMBDA_POINT;
            b(4*i + 1) *= LAMBDA_POINT;
            
            A.row(4*i + 2) *= LAMBDA_POINT;
            b(4*i + 2) *= LAMBDA_POINT;

            A.row(4*i + 3) *= LAMBDA_POINT;
            b(4*i + 3) *= LAMBDA_POINT;
        }

        VectorXf x(6);
     
        char linearSolver = 1;
        if(linearSolver == 1){ 
            JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
            x = svd.solve(b);
        }
        if(linearSolver == 2){
            x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b); 
        }
        if(linearSolver == 3){
            CompleteOrthogonalDecomposition<MatrixXf> cod(A);
            x = cod.solve(b);
        }

        float alpha = x(0), beta = x(1), gamma = x(2);

        // Build the pose matrix
        Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
            AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
            AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

        Vector3f translation = x.tail(3);

        Matrix4f estimatedPose = Matrix4f::Identity();
        estimatedPose.block(0, 0, 3, 3) = rotation;
        estimatedPose.block(0, 3, 3, 1) = translation;
    
        return estimatedPose;
    }

    Matrix4f estimatePoseColorICP(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals) {
        Matrix4f estimatedPose = Matrix4f::Identity();
    
        return estimatedPose;
    }

    Matrix4f estimatePoseSymmetricICP(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, 
            const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals) {
        const unsigned nPoints = sourcePoints.size();

        // Build the system
        MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
        VectorXf b = VectorXf::Zero(4 * nPoints);

        // Normalize source and target points to center (0,0,0)
        // Todo Compute mean
        Vector3f meanSource = computeMean(sourcePoints);
        Vector3f meanTarget = computeMean(targetPoints);

        for (unsigned i = 0; i < nPoints; i++) {
            const auto& s = sourcePoints[i];
            const auto& d = targetPoints[i];
            const auto& n = targetNormals[i];

            // FIXME Verify this
            Vector3f s_normalized = s - meanSource;
            Vector3f d_normalized = d - meanTarget;

            Vector3f normal_sum = targetNormals[i] + sourceNormals[i];
            // b
            b(4 * i) = (d_normalized - s_normalized).dot(normal_sum);

            // Add the Symmetric constraints to the system
            A.row(4 * i).segment(0, 3) = (s_normalized + d_normalized).cross(normal_sum);
            A.row(4 * i).segment(3, 3) = normal_sum;
            
            // Add point-to-point constraints //
            // Second row 
            RowVectorXf pointConstraintRow(6);
            pointConstraintRow << 0, s_normalized[2], -s_normalized[1], 1.0, 0.0, 0.0; // a, b, g, tx, ty, tz
            
            A.row(4*i + 1) = pointConstraintRow;
            b(4*i + 1) = d_normalized[0] - s_normalized[0];

            // Third row 
            pointConstraintRow << -s_normalized[2], 0, s_normalized[0], 0, 1.0, 0.0;
            
            A.row(4*i + 2) = pointConstraintRow;
            b(4*i + 2) = d_normalized[1] - s_normalized[1];

            // Fourth row 
            pointConstraintRow << s_normalized[1], -s_normalized[0], 0.0, 0.0, 0.0, 1.0;
            
            A.row(4*i + 3) = pointConstraintRow;
            b(4*i + 3) = d_normalized[2] - s_normalized[2];
        }

        // Solve the system
        // Option 1: Using LU solver
        VectorXf x(6);
        MatrixXf m_systemMatrix = A.transpose() * A;
		VectorXf m_rhs = A.transpose() * b;

		// Optionally: regularizer -> smoother surface
		// pushes the coefficients to zero
		float lambda = 0.0001;
		m_systemMatrix.diagonal() += lambda * lambda * VectorXf::Ones(6);

        FullPivLU<Matrix<float, Dynamic, Dynamic>> LU(m_systemMatrix);
	    // VectorXf m_coefficents;
		x = LU.solve(m_rhs);

        // Build the pose matrix using the rotation and translation matrices
        // Using symmtric formula
        // a_tilde = a * tan(theta) (||a|| = 1) => ||a_tilde|| = tan(theta)
        // a = a_tilde / ||a_tilde||
        // t = t_tilde * cos(theta)
        // theta < pi (180)
        Vector3f a_tilde = x.head(3);
        Vector3f t_tilde = x.tail(3);
        float tan_theta = a_tilde.norm(); // Assure a_tilde > 0
        Vector3f a = a_tilde / tan_theta;
        //std::cout << "a length " << a.norm() <<  "tan_theta" << tan_theta << "\n";

        // TODO compute angle theta; or its cos sin from a_tilde
        // Sin, cos is positive or negative
        float sin_theta = tan_theta / std::sqrt(1.0 + tan_theta * tan_theta);
        float cos_theta = sin_theta / tan_theta; 
        //std::cout << "Cos theta: " << cos_theta << " - Sin theta: " << sin_theta << "\n";
        Vector3f t = t_tilde * cos_theta; // Look good

        // TODO Fix this
        Matrix4f rodriguesMatrix =  Matrix4f::Identity();
        rodriguesMatrix.block(0, 0, 3, 3) = getRodriguesMatrix(a, sin_theta, cos_theta);

        Matrix4f estimatedPose = Matrix4f::Identity();
        // TODO Verify
        estimatedPose = gettranslationMatrix(meanTarget) * rodriguesMatrix * gettranslationMatrix(t) 
                * rodriguesMatrix * gettranslationMatrix(-meanSource);
    
        return estimatedPose;
    }
};
