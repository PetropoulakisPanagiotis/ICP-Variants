#pragma once
#include <vector>
#include "PointCloud.h"
#include "NearestNeighbor.h"

#define MAX_COLOR_DIFFERENCE 195075 

enum weighting_methods {CONSTANT_WEIGHTING=0, DISTANCES_WEIGHTING, NORMALS_WEIGHTING, COLORS_WEIGHTING};

// Weighitng class for correspondences // 
class WeightingMethod{
    private:
        int method;
        float maxDistance;

        float calculateDistancesWeight(const Vector3f& sourcePoint, const Vector3f& targetPoint){
            Vector3f diff = sourcePoint - targetPoint;

            return 1 - ((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) / this->maxDistance);
        }

        float calculateNormalsWeight(const Vector3f& sourceNormal, const Vector3f& targetNormal){
            
            return sourceNormal.dot(targetNormal);

        }

        float calculateColorsWeight(const Vector4uc& sourceColor, const Vector4uc& targetColor){
            Vector4uc diff = sourceColor - targetColor;
            
            return 1 - ((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] + diff[3] * diff[3]) / MAX_COLOR_DIFFERENCE);
        }

    public:
        WeightingMethod(int method = DISTANCES_WEIGHTING, float maxDistance = 0.0003f){
            this->method = method;
            this->maxDistance = maxDistance;
        }

        // Apply a weighting method to the correspondences //  
        void applyWeights(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, 
                const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals,
                const std::vector<Vector4uc>& sourceColors, const std::vector<Vector4uc>& targetColors,
                std::vector<Match> &matches){

            if(method == CONSTANT_WEIGHTING)
                return;
        
            unsigned int nPoints = matches.size();

            for(unsigned int i = 0; i < nPoints; i++){

                if(matches[i].idx < 0)
                    continue;
                
                float matchNewWeight = 0.0;

                if(this->method == DISTANCES_WEIGHTING || this->method == COLORS_WEIGHTING){
                    
                    if(!sourcePoints[i].allFinite() || !targetPoints[matches[i].idx].allFinite())
                        matchNewWeight += 0.0;
                    
                    else {
                     
                        float distancesWeight = calculateDistancesWeight(sourcePoints[i], targetPoints[matches[i].idx]);

                        matchNewWeight += distancesWeight;
                    }

                }

                if(this->method == NORMALS_WEIGHTING){
                    
                    if(!sourceNormals[i].allFinite() || !targetNormals[matches[i].idx].allFinite())
                        matchNewWeight += 0.0;
                
                    else {
                        float normalsWeight = calculateNormalsWeight(sourceNormals[i], targetNormals[matches[i].idx]);
                    
                        matchNewWeight += normalsWeight;
                    }
                }

                if(this->method == COLORS_WEIGHTING){
                    
                    float colorsWeight = calculateColorsWeight(sourceColors[i], targetColors[matches[i].idx]);
                    
                    matchNewWeight *= colorsWeight;
                }
          
                // Fix weight of current correspondence // 
                matches[i].weight = matchNewWeight;

                /*   
                if(matchNewWeight > 1 || matchNewWeight < 0){
                    std::cout << "ops\n";
                }
                std::cout << matchNewWeight << std::endl;
                */
            } // End for
        }
};

