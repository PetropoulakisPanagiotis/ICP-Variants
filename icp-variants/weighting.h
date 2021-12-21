#pragma once
#include "PointCloud.h"
#include "NearestNeighbor.h"

enum weighting_methods {CONSTANT_WEIGHTING=0, DISTANCE_WEIGHTING, NORMALS_WEIGHTING, COLOR_WEIGHTING, HYBRID_WEIGHTING};

// Apply all method and add an additional weighting factor to each method //
struct HybridWeights{
    float distanceWeight; // E.g. 0.4
    float planeWeight;    //      0.4 
    float colorWeight;    //      0.2
};

// Weighitng class for correspondences // 
class WeightingMethod{
    private:
        int method;
        HybridWeights hybridWeights; 

        float calculateDistanceWeight(Vector3f& sourcePoint, Vector3f& targetPoint){
            Vector3f diff = sourcePoint - targetPoint;

            return 1 - (diff.norm() / MAX_DISTANCE);
        }

        float calculateNormalsWeight(Vector3f& sourceNormal, Vector3f& targetNormal){
            return 1.0;
        }

        float calculateColorWeight(Vector3f& sourceColor, Vector3f& targetColor){
            return 1.0;
        }

    public:
        WeightingMethod(int method = DISTANCE_WEIGHTING, HybridWeights& hybridWeights = {1.0, 1.0, 1.0}){
            this-method = method;
            this->hybridWeights = hybridWeights
        }

        // Apply a weighting method to correspondences //  
        void applyWeights(const PointCloud& source, const PointCloud& target, std::vector<Match> &matches){

            if(method == CONSTANT_WEIGHTING)
                return;
        
            unsigned int nPoints = matches.size();

            for(unsigned int i = 0; i < nPoints; i++){

                if(matches[i].idx < 0)
                    continue;
                
                float currentWeight = 0.0;

                if(this->method == DISTANCE_WEIGHTING || this->method == HYBRID_WEIGHTING){
                    const auto& sourcePoint = source.getPoints()[i];
                    const auto& targetPoint = target.getPoints()[match.idx];
                
                    currentWeight += this->hybridWeights.distanceWeight * calculateDistanceWeight(sourcePoint, targetPoint);
                }

                if(this->method == NORMALS_WEIGHTING || this->method == HYBRID_WEIGHTING){
                    const auto& sourceNormal = source.getNormals()[i];
                    const auto& targetNormal = target.getNormals()[match.idx];
                
                    currentWeight += this->hybridWeights.distanceWeight * calculateNormalsWeight(sourceNormal, targetNormal);
                }

                if(this->method == COLOR_WEIGHTING || this->method == HYBRID_WEIGHTING){
                    //const auto& sourceNormal = source.getNormals()[i];
                    //const auto& targetNormal = target.getNormals()[match.idx];
                
                    currentWeight += 0.0;
                }
          
                // Fix weight of current correspondence // 
                matches[i].weight = currentWeight;
            } // End for
        }
};



