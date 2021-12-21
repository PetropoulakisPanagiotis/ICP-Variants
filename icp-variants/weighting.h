#pragma once
#include <vector>
#include "PointCloud.h"
#include "NearestNeighbor.h"

enum weighting_methods {CONSTANT_WEIGHTING=0, DISTANCE_WEIGHTING, NORMALS_WEIGHTING, COLOR_WEIGHTING, HYBRID_WEIGHTING};

// Apply all method and add an additional weighting factor to each method //
struct HybridWeights{
    float distanceWeight; // E.g. 0.4
    float normalsWeight;    //      0.4 
    float colorWeight;    //      0.2
};

// Weighitng class for correspondences // 
class WeightingMethod{
    private:
        int method;
        HybridWeights hybridWeights; 
        float maxDistance;

        float calculateDistanceWeight(const Vector3f& sourcePoint, const Vector3f& targetPoint){
            Vector3f diff = sourcePoint - targetPoint;
           
            return 1 - (diff.norm() / this->maxDistance);
        }

        float calculateNormalsWeight(const Vector3f& sourceNormal, const Vector3f& targetNormal){
            return 1.0;
        }

        float calculateColorWeight(const Vector3f& sourceColor, const Vector3f& targetColor){
            return 1.0;
        }

    public:
        WeightingMethod(int method = DISTANCE_WEIGHTING, float maxDistance = 0.0003f, HybridWeights hybridWeights = {1.0, 1.0, 1.0}){
            this->method = method;
            this->hybridWeights = hybridWeights;
            this->maxDistance = maxDistance;
        }

        // Apply a weighting method to correspondences //  
        void applyWeights(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, 
                const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals,
                std::vector<Match> &matches){

            if(method == CONSTANT_WEIGHTING)
                return;
        
            unsigned int nPoints = matches.size();

            for(unsigned int i = 0; i < nPoints; i++){

                if(matches[i].idx < 0)
                    continue;
                
                float matchNewWeight = 0.0;

                if(this->method == DISTANCE_WEIGHTING || this->method == HYBRID_WEIGHTING){
                
                    float distanceWeight = calculateDistanceWeight(sourcePoints[i], targetPoints[matches[i].idx]);

                    matchNewWeight += this->hybridWeights.distanceWeight * distanceWeight;
                }

                if(this->method == NORMALS_WEIGHTING || this->method == HYBRID_WEIGHTING){
                
                    float normalsWeight = calculateNormalsWeight(sourceNormals[i], targetNormals[matches[i].idx]);

                    matchNewWeight += this->hybridWeights.normalsWeight * normalsWeight;
                }

                if(this->method == COLOR_WEIGHTING || this->method == HYBRID_WEIGHTING){
                    matchNewWeight += 0.0;
                }
          
                // Fix weight of current correspondence // 
                matches[i].weight = matchNewWeight;
                std::cout << matches[i].weight << std::endl;
            } // End for
        }
};



