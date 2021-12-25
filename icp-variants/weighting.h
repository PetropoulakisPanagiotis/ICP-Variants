#pragma once
#include <vector>
#include "PointCloud.h"
#include "NearestNeighbor.h"

enum weighting_methods {CONSTANT_WEIGHTING=0, DISTANCES_WEIGHTING, NORMALS_WEIGHTING, COLORS_WEIGHTING, HYBRID_WEIGHTING};

// When applying all methods, add an additional weighting factor to each of them //
struct HybridWeights{
    float distancesWeight; // E.g. 0.4
    float normalsWeight;   //      0.4 
    float colorsWeight;    //      0.2
};

// Weighitng class for correspondences // 
class WeightingMethod{
    private:
        int method;
        HybridWeights hybridWeights; 
        float maxDistance;

        float calculateDistancesWeight(const Vector3f& sourcePoint, const Vector3f& targetPoint){
            Vector3f diff = sourcePoint - targetPoint;

            return 1 - ((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) / this->maxDistance);
        }

        float calculateNormalsWeight(const Vector3f& sourceNormal, const Vector3f& targetNormal){
            return 1.0;
        }

        float calculateColorsWeight(const Vector3uc& sourceColor, const Vector3uc& targetColor){
            return 1.0;
        }

    public:
        WeightingMethod(int method = DISTANCES_WEIGHTING, float maxDistance = 0.0003f, HybridWeights hybridWeights = {1.0, 1.0, 1.0}){
            this->method = method;
            this->hybridWeights = hybridWeights;
            this->maxDistance = maxDistance;
        }

        // Apply a weighting method to the correspondences //  
        void applyWeights(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, 
                const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals,
                const std::vector<Vector3uc>& sourceColors, const std::vector<Vector3uc>& targetColors,
                std::vector<Match> &matches){

            if(method == CONSTANT_WEIGHTING)
                return;
        
            unsigned int nPoints = matches.size();

            for(unsigned int i = 0; i < nPoints; i++){

                if(matches[i].idx < 0)
                    continue;
                
                float matchNewWeight = 0.0;

                if(this->method == DISTANCES_WEIGHTING || this->method == HYBRID_WEIGHTING){
                
                    float distancesWeight = calculateDistancesWeight(sourcePoints[i], targetPoints[matches[i].idx]);

                    matchNewWeight += this->hybridWeights.distancesWeight * distancesWeight;
                }

                if(this->method == NORMALS_WEIGHTING || this->method == HYBRID_WEIGHTING){
                
                    float normalsWeight = calculateNormalsWeight(sourceNormals[i], targetNormals[matches[i].idx]);

                    matchNewWeight += this->hybridWeights.normalsWeight * normalsWeight;
                }

                if(this->method == COLORS_WEIGHTING || this->method == HYBRID_WEIGHTING){
                    float colorsWeight = calculateColorsWeight(sourceColors[i], targetColors[matches[i].idx]);

                    matchNewWeight += this->hybridWeights.colorsWeight * colorsWeight;
                }
          
                // Fix weight of current correspondence // 
                matches[i].weight = matchNewWeight;

                /* Debug
                if(matchNewWeight > 1 || matchNewWeight < 0){
                    std::cout << "ops\n";
                }
                std::cout << matchNewWeight << std::endl;
                */

            } // End for
        }
};

