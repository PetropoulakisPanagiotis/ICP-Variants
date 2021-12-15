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

/**
 * ICP optimizer - Abstract Base Class
 */
class ICPOptimizer {
public:
    ICPOptimizer() :
        metric{ 0 },
        m_nIterations{ 20 }, matchingMethod{0}{ 
   
        if(matchingMethod == 0) 
            m_nearestNeighborSearch = std::make_unique<NearestNeighborSearchFlann>();
        else
            m_nearestNeighborSearch = std::make_unique<NearestNeighborSearchProjective>();
    }

    void setMatchingMaxDistance(float maxDistance) {
        m_nearestNeighborSearch->setMatchingMaxDistance(maxDistance);
    }

    void setMetric(unsigned int metric) {
        this->metric = metric;
    }

    void setMatchingMethod(unsigned int matchingMethod) {
        this->matchingMethod = matchingMethod;
    }

    void setNbOfIterations(unsigned nIterations) {
        m_nIterations = nIterations;
    }

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) = 0;

protected:
    unsigned int metric;
    unsigned int matchingMethod;
    unsigned m_nIterations;
    std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;

    std::vector<Vector3f> transformPoints(const std::vector<Vector3f>& sourcePoints, const Matrix4f& pose) {
        std::vector<Vector3f> transformedPoints;
        transformedPoints.reserve(sourcePoints.size());

        const auto rotation = pose.block(0, 0, 3, 3);
        const auto translation = pose.block(0, 3, 3, 1);

        for (const auto& point : sourcePoints) {
            transformedPoints.push_back(rotation * point + translation);
        }

        return transformedPoints;
    }

    std::vector<Vector3f> transformNormals(const std::vector<Vector3f>& sourceNormals, const Matrix4f& pose) {
        std::vector<Vector3f> transformedNormals;
        transformedNormals.reserve(sourceNormals.size());

        const auto rotation = pose.block(0, 0, 3, 3);

        for (const auto& normal : sourceNormals) {
            transformedNormals.push_back(rotation.inverse().transpose() * normal);
        }

        return transformedNormals;
    }

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

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) override {
        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
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
            clock_t begin = clock();

            auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

            auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            pruneCorrespondences(transformedNormals, target.getNormals(), matches);

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

            // Prepare point-to-point and point-to-plane constraints.
            ceres::Problem problem;
            if(metric == 0)
                prepareConstraintsPointICP(transformedPoints, target.getPoints(), target.getNormals(), matches, poseIncrement, problem);
            else if(metric == 1)
                prepareConstraintsPlaneICP(transformedPoints, target.getPoints(), target.getNormals(), matches, poseIncrement, problem);

            // Configure options for the solver.
            ceres::Solver::Options options;
            configureSolver(options);

            // Run the solver (for one iteration).
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;
            //std::cout << summary.FullReport() << std::endl;

            // Update the current pose estimate (we always update the pose from the left, using left-increment notation).
            Matrix4f matrix = PoseIncrement<double>::convertToMatrix(poseIncrement);
            estimatedPose = PoseIncrement<double>::convertToMatrix(poseIncrement) * estimatedPose;
            poseIncrement.setZero();

            std::cout << "Optimization iteration done." << std::endl;
        }

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

                // TODO: Create a new point-to-point cost function and add it as constraint (i.e. residual block) 
                // to the Ceres problem.

                const auto& targetNormal = targetNormals[match.idx];

                if (!targetNormal.allFinite())
                    continue;

                // TODO: Create a new point-to-plane cost function and add it as constraint (i.e. residual block) 
                // to the Ceres problem.
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
                        new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, match.weight)
                    ),
                    nullptr, poseIncrement.getData()
                );

            }
        }
    }

    void prepareConstraintsColorICP(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals, const std::vector<Match> matches, const PoseIncrement<double>& poseIncrement, ceres::Problem& problem) const {
        const unsigned nPoints = sourcePoints.size();
        /*
        for (unsigned i = 0; i < nPoints; ++i) {
            const auto match = matches[i];
            if (match.idx >= 0) {
                const auto& sourcePoint = sourcePoints[i];
                const auto& targetPoint = targetPoints[match.idx];

                if (!sourcePoint.allFinite() || !targetPoint.allFinite())
                    continue;

                // TODO: Create a new point-to-point cost function and add it as constraint (i.e. residual block) 
                // to the Ceres problem.

                const auto& targetNormal = targetNormals[match.idx];

                if (!targetNormal.allFinite())
                    continue;

                // TODO: Create a new point-to-plane cost function and add it as constraint (i.e. residual block) 
                // to the Ceres problem.
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
                        new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, match.weight)
                    ),
                    nullptr, poseIncrement.getData()
                );

            }
        }
        */
    }

    void prepareConstraintsSymmetricICP(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals, const std::vector<Match> matches, const PoseIncrement<double>& poseIncrement, ceres::Problem& problem) const {
        const unsigned nPoints = sourcePoints.size();
        /*
        for (unsigned i = 0; i < nPoints; ++i) {
            const auto match = matches[i];
            if (match.idx >= 0) {
                const auto& sourcePoint = sourcePoints[i];
                const auto& targetPoint = targetPoints[match.idx];

                if (!sourcePoint.allFinite() || !targetPoint.allFinite())
                    continue;

                // TODO: Create a new point-to-point cost function and add it as constraint (i.e. residual block) 
                // to the Ceres problem.

                const auto& targetNormal = targetNormals[match.idx];

                if (!targetNormal.allFinite())
                    continue;

                // TODO: Create a new point-to-plane cost function and add it as constraint (i.e. residual block) 
                // to the Ceres problem.
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
                        new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, match.weight)
                    ),
                    nullptr, poseIncrement.getData()
                );

            }
        }
        */
    }

};


/**
 * ICP optimizer - using linear least-squares for optimization.
 */
class LinearICPOptimizer : public ICPOptimizer {
public:
    LinearICPOptimizer() {}

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) override {
        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        m_nearestNeighborSearch->buildIndex(target.getPoints());

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ..." << std::endl;
            clock_t begin = clock();

            auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

            auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            pruneCorrespondences(transformedNormals, target.getNormals(), matches);

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

            std::vector<Vector3f> sourcePoints;
            std::vector<Vector3f> targetPoints;

            // Add all matches to the sourcePoints and targetPoints vector,
            // so that the sourcePoints[i] matches targetPoints[i]. For every source point,
            // the matches vector holds the index of the matching target point.
            for (int j = 0; j < transformedPoints.size(); j++) {
                const auto& match = matches[j];
                if (match.idx >= 0) {
                    sourcePoints.push_back(transformedPoints[j]);
                    targetPoints.push_back(target.getPoints()[match.idx]);
                }
            }

            // Estimate the new pose
            if (metric == 1) {
                estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, target.getNormals()) * estimatedPose;
            }
            else if(metric == 0) {
                estimatedPose = estimatePosePointToPoint(sourcePoints, targetPoints) * estimatedPose;
            }

            std::cout << "Optimization iteration done." << std::endl;
        }

        // Store result
        initialPose = estimatedPose;
    }

private:
    Matrix4f estimatePosePointToPoint(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints) {
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

            // TODO: Add the point-to-plane constraints to the system

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

            // TODO: Add the point-to-point constraints to the system
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
            
            //TODO: Optionally, apply a higher weight to point-to-plane correspondences
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

        // TODO: Solve the system
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

        // TODO: Build the pose matrix using the rotation and translation matrices
        Matrix4f estimatedPose = Matrix4f::Identity();
        estimatedPose.block(0, 0, 3, 3) = rotation;
        estimatedPose.block(0, 3, 3, 1) = translation;
    
        return estimatedPose;
    }

    Matrix4f estimatePoseColorICP(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals) {
        Matrix4f estimatedPose = Matrix4f::Identity();
    
        return estimatedPose;
    }

    Matrix4f estimatePoseSymmetricICP(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals) {
        Matrix4f estimatedPose = Matrix4f::Identity();
    
        return estimatedPose;
    }
};
