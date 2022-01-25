#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <assert.h>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/point_cloud.h>
#include "PointCloud.h"

#include "Eigen.h"
#include "utils.h"

class ConvergenceMeasure
{
private:
    /* data */
    std::vector<Vector3f> m_sourcePoints;
    std::vector<Vector3f> m_unchangedPoints;
    std::vector<float> iterationErrorsRMSE;
    std::vector<float> iterationErrorsBenchmark;
    int numCorrspondeces;
    bool m_runBenchmark;

public:
    ConvergenceMeasure() {
        numCorrspondeces = 0;
        m_runBenchmark = false;
    };

    ConvergenceMeasure(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& unchangedPoints, const bool runBenchmark = false) 
    {       
        ASSERT(sourcePoints.size() == unchangedPoints.size() &&  sourcePoints.size() > 0
                && "The number of points must be the same and > 0.");
        m_sourcePoints = sourcePoints;
        m_unchangedPoints = unchangedPoints;
        numCorrspondeces = m_sourcePoints.size();
        m_runBenchmark = runBenchmark;
    };

    ~ConvergenceMeasure() {};

    /**
     * Computing rmse Alignment error between two Poincloud
     * With known correspondences index
     * Note that although source was usually transformed, 
     * indexes of points inside it remain unchange.
     */
    float rmseAlignmentError(const Matrix4f& pose) {
        ASSERT(numCorrspondeces > 0
                && "The number of correspondences must be > 0.");
        float rmse = 0.0;
        auto transformedPoints = transformPoints(m_sourcePoints, pose);
        for (int i=0; i < numCorrspondeces; ++i) {
            // Compute error
            rmse += (transformedPoints[i] - m_unchangedPoints[i]).squaredNorm();
        }
        rmse /= numCorrspondeces;
        return std::sqrt(rmse);
    };

    /* Record Alignment Error*/
    void recordAlignmentError(const Matrix4f& pose) {
        float rmse_err = rmseAlignmentError(pose);
        std::cout << "RMSE Alignment errors: " << rmse_err << "\n";
        iterationErrorsRMSE.push_back(rmse_err);
        if (m_runBenchmark) {
            float benchmark_err = benchmarkError(pose);
            std::cout << "Benchmark errors: " << benchmark_err << "\n";
            iterationErrorsBenchmark.push_back(benchmark_err);
        }
    };

    /* Print Alignment Errors*/
    void outputAlignmentError() {
        if (iterationErrorsRMSE.size() == 0) {
            std::cout << "No recorded alignment error.\n";
            return;
        }
        std::cout << "Recorded RMSE Alginment Error!\n";
        std::cout << "\tIter \t RMSE Error\n";
        for (int i=0; i< iterationErrorsRMSE.size(); i++) {
            printf ("\t%02d \t %01.6f\n", i, iterationErrorsRMSE[i]);
        }
        if (m_runBenchmark) {
            if (iterationErrorsBenchmark.size() == 0) {
                std::cout << "No recorded alignment error for benchmark.\n";
                return;
            }
            std::cout << "Recorded benchmark Alginment Error!\n";
            std::cout << "\tIter \t Benchmark Error\n";
            for (int i = 0; i < iterationErrorsBenchmark.size(); i++) {
                printf("\t%02d \t %01.6f\n", i, iterationErrorsBenchmark[i]);
            }
        }
    };
    
    double benchmarkError(const Matrix4f& pose) {
        ASSERT(numCorrspondeces > 0
            && "The number of points must be > 0.");

        auto transformedPoints = transformPoints(m_sourcePoints, pose);

        pcl::PointCloud<pcl::PointXYZ> pcl_sourcePoints;
        pcl_sourcePoints.width = 1;
        pcl_sourcePoints.height = transformedPoints.size();
        pcl_sourcePoints.points.resize(transformedPoints.size());
        for (int i = 0; i < transformedPoints.size(); i++)
        {
            pcl_sourcePoints.points[i].x = transformedPoints[i].x();
            pcl_sourcePoints.points[i].y = transformedPoints[i].y();
            pcl_sourcePoints.points[i].z = transformedPoints[i].z();
        }
        pcl::PointCloud<pcl::PointXYZ> pcl_unchangedPoints;
        pcl_unchangedPoints.width = 1;
        pcl_unchangedPoints.height = m_unchangedPoints.size();
        pcl_unchangedPoints.points.resize(m_unchangedPoints.size());
        for (int i = 0; i < m_unchangedPoints.size(); i++)
        {
            pcl_unchangedPoints.points[i].x = m_unchangedPoints[i].x();
            pcl_unchangedPoints.points[i].y = m_unchangedPoints[i].y();
            pcl_unchangedPoints.points[i].z = m_unchangedPoints[i].z();
        }
        return calculate_error(pcl_sourcePoints.makeShared(), pcl_unchangedPoints.makeShared());
    };

    double calculate_error(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2) {
        assert(cloud1->size() == cloud2->size());
        
        double error = 0;
        Eigen::Vector4d centroid_v;
        
        pcl::compute3DCentroid(*cloud1, centroid_v);
        pcl::PointXYZ centroid(centroid_v[0], centroid_v[1], centroid_v[2]);
        
        for (int i = 0; i < cloud1->size(); i++) {
            double centroid_distance = pcl::euclideanDistance(cloud1->at(i), centroid);

            error += pcl::euclideanDistance(cloud1->at(i), cloud2->at(i)) / centroid_distance;
        }

        error /= cloud1->size();

        return error;
    };

    void writeRMSEToFile(std::string nameFile){
        std::ofstream newFile;

        newFile.open(nameFile);

        for(unsigned int i = 0; i < this->iterationErrorsRMSE.size(); i++){
            newFile << iterationErrorsRMSE[i] << std::endl;
        }

        newFile.close();
    }

    void writeBenchmarkToFile(std::string nameFile) {
        std::ofstream newFile;

        newFile.open(nameFile);

        for (unsigned int i = 0; i < this->iterationErrorsBenchmark.size(); i++) {
            newFile << iterationErrorsBenchmark[i] << std::endl;
        }

        newFile.close();
    }

    float getFinalErrorRMSE() {
        return iterationErrorsRMSE.back();
    }

    float getFinalErrorBenchmark() {
        return iterationErrorsBenchmark.back();
    }
};
