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

#define MULTI_RESOLUTION_MINIMUM_POINTS 100 // 1-> enable 


/**
 * ICP optimizer - Abstract Base Class
 */
class ICPOptimizer {
public:
    ICPOptimizer() :
        metric{ 0 }, selectionMethod{0}, rejectionMethod{1}, weightingMethod{0},
        m_nIterations{ 20 }, matchingMethod{0}, maxDistance{0.0003f}, colorICP{false}, multiResolutionICP{false}{ 
   
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

    void enableMultiResolution(bool enableMultiResolution) {
        this->multiResolutionICP = enableMultiResolution;
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
    
    void printICPConfiguration(){
        std::cout << "\n\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n";
        std::cout << "Starting ICP with the following configuration:\n";

        if(colorICP)
            std::cout << "Color-ICP enabled\n";

        if(multiResolutionICP)
            std::cout << "Multi-Resolution ICP enabled\n";

        if(selectionMethod == SELECT_ALL)
            std::cout << "1. Selection: all\n";
        else if(selectionMethod == RANDOM_SAMPLING)
            std::cout << "1. Selection: random\n";

        if(matchingMethod == 1)
            std::cout << "2. Matching: projective (max distance " << maxDistance << " m)\n";
        else if(matchingMethod == 0)
            std::cout << "2. Matching: k-nn (max distance " << maxDistance << " m)\n";
   
        if(weightingMethod == 0)
            std::cout << "3. Weighting: constant\n";
        else if(weightingMethod == 1)
            std::cout << "3. Weighting: point distances\n";
        else if(weightingMethod == 2)
            std::cout << "3. Weighting: normals\n";
        else if(weightingMethod == 3)
            std::cout << "3. Weighting: colors\n";
    
        if(rejectionMethod == 1)
            std::cout << "4. Rejection: angle of normals\n";
        else
            std::cout << "4. Rejection: keep all\n";
        
        if (metric == 0) 
            std::cout << "5. Metric: Point to Point\n";
        else if (metric == 1) 
            std::cout << "5. Metric: Point to Plane\n";
        else if (metric == 2) 
            std::cout << "5. Metric: Symmetric\n";
        std::cout << "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n\n";
    }

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose, bool calculateRMSE = true) = 0;

protected:
    unsigned int metric;
    bool colorICP;
    bool multiResolutionICP;
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
        clock_t step_start, start, tot_time; 
        double iter_time; 
        // Use step_start as tmp var for every step
        // iter_time print time needed per iteration 
        
        printICPConfiguration();
        
        start = clock(); // Measure the whole iteration 

        int currentResolution = 1; 
        int originalSize = source.getPoints().size();
        if(this->multiResolutionICP){
            // Find the lowest resolution                        //
            // e.g. step 1 -> 32, step 2 -> 16, step 3 -> 4      //
            // step 4 -> 2, step 5 -> 1 (aka original size)      //
            // Lowest resolution should have at least 300 points //
            while(1){
                originalSize = originalSize / 2;
                if(originalSize < MULTI_RESOLUTION_MINIMUM_POINTS)
                    break;
                currentResolution *= 2;
            } // End while
        }
        
        // Initialize selection step //
        PointSelection sourceSelection;
        if(this->multiResolutionICP){
            PointCloud coarseCloud = source.getCoarseResolution(currentResolution);
            sourceSelection = PointSelection(coarseCloud, selectionMethod, proba);
        }
        else
            sourceSelection = PointSelection(source, selectionMethod, proba);

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

        for (int i = 0; i < m_nIterations || this->multiResolutionICP; ++i) {
            std::cout << std::endl << "--- Running iteration " << i << std::endl;
            
            if(this->multiResolutionICP)
                std::cout << "Current resolution: " << currentResolution << std::endl;

            // 1. Selection Step // 
            step_start = clock();
            // Change source to sourceSelection to do selection.
            if (selectionMethod == RANDOM_SAMPLING) // Resample each iteration
                sourceSelection.resample();
            m_timeMeasure->selectionTime += double(clock() - step_start) / CLOCKS_PER_SEC;
            
            auto transformedPoints = transformPoints(sourceSelection.getPoints(), estimatedPose);
            auto transformedNormals = transformNormals(sourceSelection.getNormals(), estimatedPose);
            std::cout << "Number of source points to match = " << transformedPoints.size() << std::endl;

            // 2. Matching step //
            // Compute the matches.
            std::vector<Match> matches; 
            
            std::cout << "Matching points ..." << std::endl;
            
            step_start = clock();
            if(this->colorICP)
                matches = m_nearestNeighborSearch->queryMatches(transformedPoints, sourceSelection.getColors());
            else
                matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            
            m_timeMeasure->matchingTime += double(clock() - step_start) / CLOCKS_PER_SEC;
          
            // 3. Weighting step // 
            step_start = clock();
            weightingStep.applyWeights(transformedPoints, target.getPoints(), transformedNormals, target.getNormals(), 
                                       sourceSelection.getColors(), target.getColors(), matches);
            
            m_timeMeasure->weighingTime += double(clock() - step_start) / CLOCKS_PER_SEC;
            
            // 4. Rejection step //
            step_start = clock();
            if (rejectionMethod == 1)
                pruneCorrespondences(transformedNormals, target.getNormals(), matches);
            m_timeMeasure->rejectionTime += double(clock() - step_start) / CLOCKS_PER_SEC;

            // 5. Select error metric //            
            ceres::Problem problem;
            ceres::Solver::Summary summary;
            ceres::Solver::Options options;

            // Configure options for the solver.
            configureSolver(options);

            step_start = clock();
            if(metric == 0)
                prepareConstraintsPointICP(transformedPoints, target.getPoints(), target.getNormals(), matches, poseIncrement, problem);
            else if(metric == 1)
                prepareConstraintsPlaneICP(transformedPoints, target.getPoints(), target.getNormals(), matches, poseIncrement, problem);
            else if(metric == 2)
                prepareConstraintsSymmetricICP(transformedPoints, target.getPoints(), transformedNormals, target.getNormals(), matches, poseIncrement, problem);

            // Run the solver (for one iteration).
            ceres::Solve(options, &problem, &summary);
           
            iter_time = double(clock() - step_start) / CLOCKS_PER_SEC;
            m_timeMeasure->solverTime += iter_time; 
            
            std::cout << summary.BriefReport() << std::endl;
            //std::cout << summary.FullReport() << std::endl;

            // Update the current pose estimate (we always update the pose from the left, using left-increment notation).
            Matrix4f matrix = PoseIncrement<double>::convertToMatrix(poseIncrement);
            estimatedPose = PoseIncrement<double>::convertToMatrix(poseIncrement) * estimatedPose;
            poseIncrement.setZero();

            std::cout << "Optimization iteration done (in " << iter_time << "s)"  << std::endl;
            
            // RMSE compute
            if (calculateRMSE) {
                m_convergenceMeasure->recordAlignmentError(estimatedPose);
            }

            // Increase resolution //
            if(multiResolutionICP){
               
                // Reached max resolution and max iterations //
                if(currentResolution == 1 && i >= m_nIterations - 1)
                    break;

                // Reached max resolution but not max iterations                //
                // Continue - especially for small sets we need this condition  //
                // In the original paper they stop when resolution == 1         //
                // For bunny, we can not reduce a lot the resolution and hence, //
                // we perfom less than 5 ICP steps without this condition and   // 
                // we have not found a descent pose                             // 
                if(currentResolution == 1)
                    continue;

                currentResolution /= 2;
                
                PointCloud coarseCloud = source.getCoarseResolution(currentResolution);
                sourceSelection = PointSelection(coarseCloud, selectionMethod, proba);
            }
        }

        // Store result
        initialPose = estimatedPose;

        // Measure the whole actual time //
        m_timeMeasure->convergenceTime += double(clock() - start) / CLOCKS_PER_SEC;
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
        
        std::cout << "Preparing Point-to-Point ICP Non-linear" << std::endl;

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

        std::cout << "Preparing Point-to-Plane ICP Non-linear" << std::endl;
        
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
        
        std::cout << "Preparing Symmetric ICP Non-linear" << std::endl;
        
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
        clock_t start, step_start, tot_time;
        double iter_time; 
        // Use step_start as tmp var for every step
        // iter_time print time needed per iteration 
    
        printICPConfiguration();
        
        start = clock();

        int currentResolution = 1; 
        int originalSize = source.getPoints().size();
        if(this->multiResolutionICP){
            // Find the lowest resolution                        //
            // e.g. step 1 -> 32, step 2 -> 16, step 3 -> 4      //
            // step 4 -> 2, step 5 -> 1 (aka original size)      //
            // Lowest resolution should have at least 300 points //
            while(1){
                originalSize = originalSize / 2;
                if(originalSize < MULTI_RESOLUTION_MINIMUM_POINTS)
                    break;
                currentResolution *= 2;
            } // End while
        }
        
        // Initialize selection step //
        PointSelection sourceSelection;
        if(this->multiResolutionICP){
            PointCloud coarseCloud = source.getCoarseResolution(currentResolution);
            sourceSelection = PointSelection(coarseCloud, selectionMethod, proba);
        }
        else
            sourceSelection = PointSelection(source, selectionMethod, proba);

        // Initialize weightingStep step //
        auto weightingStep = WeightingMethod(this->weightingMethod, this->maxDistance);
        
        // Initialize matching step // 
        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        if(this->colorICP)
            m_nearestNeighborSearch->buildIndex(target.getPoints(), target.getColors());
        else
            m_nearestNeighborSearch->buildIndex(target.getPoints());

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        for (int i = 0; i < m_nIterations || this->multiResolutionICP; ++i) {
            std::cout << std::endl << "--- Running iteration " << i << std::endl;

            if(this->multiResolutionICP)
                std::cout << "Current resolution: " << currentResolution << std::endl;

            // 1. Selection step //
            step_start = clock();
            // Change source to sourceSelection to do selection.
            if (selectionMethod == RANDOM_SAMPLING) // Resample each iteration
                sourceSelection.resample();
            m_timeMeasure->selectionTime += double(clock() - step_start) / CLOCKS_PER_SEC;
            
            auto transformedPoints = transformPoints(sourceSelection.getPoints(), estimatedPose);
            auto transformedNormals = transformNormals(sourceSelection.getNormals(), estimatedPose);
            std::cout << "Number of source points to match = " << transformedPoints.size() << std::endl;

            //2. Matching step //
            std::vector<Match> matches; 
            std::cout << "Matching points ..." << std::endl;
            
            step_start = clock();
            if(this->colorICP)
                matches = m_nearestNeighborSearch->queryMatches(transformedPoints, sourceSelection.getColors());
            else
                matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            
            m_timeMeasure->matchingTime += double(clock() - step_start) / CLOCKS_PER_SEC;

            // 3. Weighting step // 
            step_start = clock();
            weightingStep.applyWeights(transformedPoints, target.getPoints(), transformedNormals, target.getNormals(), 
                                       sourceSelection.getColors(), target.getColors(), matches);

            m_timeMeasure->weighingTime += double(clock() - step_start) / CLOCKS_PER_SEC;
            
            // 4. Rejection step //
            step_start = clock();
            if (rejectionMethod == 1)
                pruneCorrespondences(transformedNormals, target.getNormals(), matches);
            m_timeMeasure->rejectionTime += double(clock() - step_start) / CLOCKS_PER_SEC;

            // 5. Select error metric //
            std::vector<Vector3f> sourcePoints;
            std::vector<Vector3f> targetPoints;
            std::vector<Vector3f> sourceNormals;
            std::vector<Vector3f> targetNormals;
            
            step_start = clock();
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

            iter_time = double(clock() - step_start) / CLOCKS_PER_SEC;
            m_timeMeasure->solverTime += iter_time; 

            std::cout << "Optimization iteration done (in " << iter_time << "s)"  << std::endl;

            // RMSE compute
            if (calculateRMSE) {
                m_convergenceMeasure->recordAlignmentError(estimatedPose);
            }

            // Increase resolution //
            if(multiResolutionICP){

                // Reached max resolution and max iterations //
                if(currentResolution == 1 && i >= m_nIterations - 1)
                    break;

                // Reached max resolution but not max iterations                //
                // Continue - especially for small sets we need this condition  //
                // In the original paper they stop when resolution == 1         //
                // For bunny, we can not reduce a lot the resolution and hence, //
                // we perfom less than 5 ICP steps without this condition and   // 
                // we have not found a descent pose                             // 
                if(currentResolution == 1)
                    continue;

                currentResolution /= 2;

                PointCloud coarseCloud = source.getCoarseResolution(currentResolution);
                sourceSelection = PointSelection(coarseCloud, selectionMethod, proba);
            }
        }

        // Store result
        initialPose = estimatedPose;

        // Measure the whole actual time //
        m_timeMeasure->convergenceTime += double(clock() - start) / CLOCKS_PER_SEC;
    }

private:
    Matrix4f estimatePosePointToPoint(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, bool calculateRMSE = true) {
        std::cout << "Preparing Point-to-Point ICP Linear" << std::endl;
        
        ProcrustesAligner procrustAligner;
        Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);

        return estimatedPose;
    }

    Matrix4f estimatePosePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals) {
        std::cout << "Preparing Point-to-Plane ICP Linear" << std::endl;
        
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

    Matrix4f estimatePoseSymmetricICP(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, 
            const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals) {
        
        std::cout << "Preparing Point-to-Plane ICP Linear" << std::endl;
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

        // compute angle theta; or its cos sin from a_tilde
        // Sin, cos is positive or negative
        float sin_theta = tan_theta / std::sqrt(1.0 + tan_theta * tan_theta);
        float cos_theta = sin_theta / tan_theta; 
        //std::cout << "Cos theta: " << cos_theta << " - Sin theta: " << sin_theta << "\n";
        Vector3f t = t_tilde * cos_theta; // Look good

        Matrix4f rodriguesMatrix =  Matrix4f::Identity();
        rodriguesMatrix.block(0, 0, 3, 3) = getRodriguesMatrix(a, sin_theta, cos_theta);

        Matrix4f estimatedPose = Matrix4f::Identity();
        
        estimatedPose = gettranslationMatrix(meanTarget) * rodriguesMatrix * gettranslationMatrix(t) 
                * rodriguesMatrix * gettranslationMatrix(-meanSource);
    
        return estimatedPose;
    }
};
