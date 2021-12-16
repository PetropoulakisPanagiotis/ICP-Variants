#pragma once
#include <vector>
#include <math.h>
#include "PointCloud.h"
#include "Eigen.h"
#include "utils.h"

class ConvergenceMeasure
{
private:
    /* data */
    std::vector<Vector3f> m_sourceCorrespondences;
    std::vector<Vector3f> m_targetCorrespondences;
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
};
