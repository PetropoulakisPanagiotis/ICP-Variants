#pragma once

#include <vector>
#include "PointCloud.h"
#include <iostream>

enum selection_methods {SELECT_ALL, UNIFORM_SAMPLING, RANDOM_SAMPLING, STABLE_SAMPLING};


class PointSelection {
    public:
        PointSelection(const PointCloud& source, unsigned int selectionMode=SELECT_ALL, float selectionProba=0.5f)
            : m_source {source} {
            std::cout << "Point selection with mode " << selectionMode << std::endl;
            numPoints = source.getPoints().size();
            m_selectionMode = selectionMode;
            m_selectionProba = selectionProba;
            selectPointIndexes();
        }

        // Get points associated with selected indexes
        std::vector<Vector3f>& getPoints() {
            // If select all
            if (m_selectionMode == SELECT_ALL) 
                return m_source.getPoints();

            auto points = m_source.getPoints();
            // Resample pointIndexes
            if (m_selectionMode == RANDOM_SAMPLING) {
                selectRandomPointIndexes(m_selectionProba);
            }

            std::vector<Vector3f> selectedPoints;
            for (auto idx: selectedPointIndexes) {
                selectedPoints.push_back(points[idx]);
            }
            return selectedPoints;
        }


        // Get normals associated with selected indexes
        std::vector<Vector3f>& getNormals() {
            // If select all
            if (m_selectionMode == SELECT_ALL) 
                return m_source.getNormals();;

            auto normals = m_source.getNormals();
            std::vector<Vector3f> selectedNormals;
            for (auto idx: selectedPointIndexes) {
                selectedNormals.push_back(normals[idx]);
            }
            return selectedNormals;
        }

    private:
        PointCloud m_source;
        std::vector<int> selectedPointIndexes;
        unsigned int numPoints;
        unsigned int m_selectionMode;
        float m_selectionProba;

        // Fill indexes
        void selectPointIndexes() {
            if (m_selectionMode == SELECT_ALL) 
                selectAllPointIndexes();
        };

        // Fill all indexes
        void selectAllPointIndexes() {
            // for (size_t i=0; i < numPoints; ++i) {
            //     selectedPointIndexes.push_back(i);
            // }
            return;
        }

        void selectRandomPointIndexes(float prob) {
        }

        void selectRandomPointIndexes(int numSamples) {
        }
};