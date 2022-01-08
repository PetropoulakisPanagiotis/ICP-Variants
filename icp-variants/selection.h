#pragma once

#include <vector>
#include "PointCloud.h"
#include <iostream>
#include <random>

enum selection_methods {SELECT_ALL=0, RANDOM_SAMPLING, STABLE_SAMPLING, UNIFORM_SAMPLING};
typedef std::mt19937 MyRNG;  // the Mersenne Twister with a popular choice of parameters


class PointSelection {
    public:
        PointSelection(const PointCloud& source, unsigned int selectionMode=SELECT_ALL, float selectionProba=0.5f)
            : m_source {source} {
            std::cout << "Point selection with mode " << selectionMode << std::endl;
            numPoints = source.getPoints().size();
            m_selectionMode = selectionMode;
            m_selectionProba = selectionProba;
            if (m_selectionMode > SELECT_ALL)
                initSampler();
        }

        // Get points associated with selected indexes
        const std::vector<Vector3f>& getPoints() {
            // If select all
            if (m_selectionMode == SELECT_ALL) 
                return m_source.getPoints();

            if (m_selectionMode == RANDOM_SAMPLING)
                return m_points;
        }


        // Get normals associated with selected indexes
        const std::vector<Vector3f>& getNormals() {
            // If select all
            if (m_selectionMode == SELECT_ALL) 
                return m_source.getNormals();;

            if (m_selectionMode == RANDOM_SAMPLING)
                return m_normals;
        }

        void resample() {
            std::cout << "Resample points.\n";
            sampleRandomPoints(m_selectionProba);
        }

    private:
        PointCloud m_source;
        std::vector<int> selectedPointIndexes;
        unsigned int numPoints;
        unsigned int numSelectedPoints;
        unsigned int m_selectionMode;
        double m_selectionProba;
        std::vector<Vector3f> m_points;
        std::vector<Vector3f> m_normals;
        MyRNG rng;    


        // Init rng
        void initializeRandomGenerator() {
            std::random_device rd;
            rng.seed(rd());
        }

        // Init sampler
        void initSampler() {
            if (m_selectionMode == RANDOM_SAMPLING) 
                initializeRandomGenerator();
        };

        // Sample random points and populate m_points and m_normals;
        void sampleRandomPoints(double prob) {
            std::uniform_real_distribution<double> ureal(0.0, 1.0);
            numSelectedPoints = 0;
            m_points.clear();
            m_normals.clear();

            // Sample
            for (size_t i=0; i < numPoints; i++) {
                auto rnumber = ureal(rng);
                if (rnumber < prob) {
                    // std::cout << i << " ";
                    m_points.push_back(m_source.getPoints()[i]);
                    m_normals.push_back(m_source.getNormals()[i]);
                    numSelectedPoints++;
                }
            }
            std::cout << "\tNumber points samples " << m_points.size() << "\n";
        }
};
