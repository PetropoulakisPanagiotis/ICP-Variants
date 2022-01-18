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
    std::vector<Vector3f> m_sourceCorrespondences;
    std::vector<Vector3f> m_targetCorrespondences;
    std::vector<float> iterationErrors;
    int numCorrspondeces;

public:
    ConvergenceMeasure() {
        numCorrspondeces = 0;
    };

    ConvergenceMeasure(const std::vector<Vector3f>& sourceCorrespondences, const std::vector<Vector3f>& targetCorrespondences) 
    {       
        ASSERT(sourceCorrespondences.size() == targetCorrespondences.size() &&  sourceCorrespondences.size() > 0
                && "The number of source and target correspondences must be the same and > 0.");
        m_sourceCorrespondences = sourceCorrespondences;
        m_targetCorrespondences = targetCorrespondences;
        numCorrspondeces = m_sourceCorrespondences.size();
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
        auto transformedPoints = transformPoints(m_sourceCorrespondences, pose);
        for (int i=0; i < numCorrspondeces; ++i) {
            // Compute error
            rmse += (transformedPoints[i] - m_targetCorrespondences[i]).squaredNorm();
        }
        rmse /= numCorrspondeces;
        return std::sqrt(rmse);
    };

    /* Record Alignment Error*/
    void recordAlignmentError(const Matrix4f& pose) {
        float rmse_err = rmseAlignmentError(pose);
        std::cout << "RMSE Alignment errors: " << rmse_err << "\n";
        iterationErrors.push_back(rmse_err);
    };

    /* Print Alignment Errors*/
    void outputAlignmentError() {
        if (iterationErrors.size() == 0) {
            std::cout << "No recorded alignment error.\n";
            return;
        }
        std::cout << "Recorded RMSE Alginment Error!\n";
        std::cout << "\tIter \t RMSE Error\n";
        for (int i=0; i< iterationErrors.size(); i++) {
            printf ("\t%02d \t %01.6f\n", i, iterationErrors[i]);
        }
    };
  
    static double calculate_error(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2) {
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

    void writeToFile(std::string nameFile){
        std::ofstream newFile;

        newFile.open(nameFile);

        for(unsigned int i = 0; i < this->iterationErrors.size(); i++){
            newFile << iterationErrors[i] << std::endl;
        }

        newFile.close();
    }
};
