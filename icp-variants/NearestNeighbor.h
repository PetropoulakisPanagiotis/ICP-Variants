#pragma once
#include <flann/flann.hpp>
#include "Eigen.h"

#define MAX_DISTANCE 0.005f

struct Match {
	int idx;
	float weight;
};

class NearestNeighborSearch {
public:
	virtual ~NearestNeighborSearch() {}

	virtual void setMatchingMaxDistance(float maxDistance) {
		m_maxDistance = maxDistance; // Squared
	}

    float getMatchingMaxDistance(float maxDistance) {
		return m_maxDistance; // Squared
	}

	virtual void buildIndex(const std::vector<Eigen::Vector3f>& targetPoints) = 0;
	virtual std::vector<Match> queryMatches(const std::vector<Vector3f>& transformedPoints) = 0;
	
    virtual void setCameraParams(const Eigen::Matrix3f& depthIntrinsics, const unsigned width, const unsigned height) = 0;

protected:
	float m_maxDistance;

	NearestNeighborSearch() : m_maxDistance{ MAX_DISTANCE } {}
};


/**
 * Brute-force nearest neighbor search.
 */
class NearestNeighborSearchBruteForce : public NearestNeighborSearch {
public:
	NearestNeighborSearchBruteForce() : NearestNeighborSearch() {}

	void buildIndex(const std::vector<Eigen::Vector3f>& targetPoints) {
		m_points = targetPoints;
	}

	std::vector<Match> queryMatches(const std::vector<Vector3f>& transformedPoints) {
		const unsigned nMatches = transformedPoints.size();
		std::vector<Match> matches(nMatches);
		const unsigned nTargetPoints = m_points.size();
		std::cout << "nMatches: " << nMatches << std::endl;
		std::cout << "nTargetPoints: " << nTargetPoints << std::endl;

		#pragma omp parallel for
		for (int i = 0; i < nMatches; i++) {
			matches[i] = getClosestPoint(transformedPoints[i]);
		}

		return matches;
	}
 
    void setCameraParams(const Eigen::Matrix3f& depthIntrinsics, const unsigned width, const unsigned height){
        return;
    }


private:
	std::vector<Eigen::Vector3f> m_points;

	Match getClosestPoint(const Vector3f& p) {
		int idx = -1;

		float minDist = std::numeric_limits<float>::max();
		for (unsigned int i = 0; i < m_points.size(); ++i) {
			float dist = (p - m_points[i]).norm();
			if (minDist > dist) {
				idx = i;
				minDist = dist;
			}
		}

		if (minDist <= m_maxDistance)
			return Match{ idx, 1.f };
		else
			return Match{ -1, 0.f };
	}
};


/**
 * Nearest neighbor search using FLANN.
 */
class NearestNeighborSearchFlann : public NearestNeighborSearch {
public:
	NearestNeighborSearchFlann() :
		NearestNeighborSearch(),
		m_nTrees{ 1 },
		m_index{ nullptr },
		m_flatPoints{ nullptr }
	{ }

	~NearestNeighborSearchFlann() {
		if (m_index) {
			delete m_flatPoints;
			delete m_index;
			m_flatPoints = nullptr;
			m_index = nullptr;
		}
	}

	void buildIndex(const std::vector<Eigen::Vector3f>& targetPoints) {
		std::cout << "Initializing FLANN index with " << targetPoints.size() << " points." << std::endl;

		// FLANN requires that all the points be flat. Therefore we copy the points to a separate flat array.
		m_flatPoints = new float[targetPoints.size() * 3];
		for (size_t pointIndex = 0; pointIndex < targetPoints.size(); pointIndex++) {
			for (size_t dim = 0; dim < 3; dim++) {
				m_flatPoints[pointIndex * 3 + dim] = targetPoints[pointIndex][dim];
			}
		}

		flann::Matrix<float> dataset(m_flatPoints, targetPoints.size(), 3);

		// Building the index takes some time.
		m_index = new flann::Index<flann::L2<float>>(dataset, flann::KDTreeIndexParams(m_nTrees));
		m_index->buildIndex();

		std::cout << "FLANN index created." << std::endl;
	    this->type = 0; // Only points
    }

	std::vector<Match> queryMatches(const std::vector<Vector3f>& transformedPoints) {
        if (!m_index) {
			std::cout << "FLANN index needs to be build before querying any matches." << std::endl;
			return {};
		}

        if (this->type ==  1) {
			std::cout << "Please call queryMatches with colors." << std::endl;
			return {};
		}

		// FLANN requires that all the points be flat. Therefore we copy the points to a separate flat array.
		float* queryPoints = new float[transformedPoints.size() * 3];
		for (size_t pointIndex = 0; pointIndex < transformedPoints.size(); pointIndex++) {
			for (size_t dim = 0; dim < 3; dim++) {
				queryPoints[pointIndex * 3 + dim] = transformedPoints[pointIndex][dim];
			}
		}

		flann::Matrix<float> query(queryPoints, transformedPoints.size(), 3);
		flann::Matrix<int> indices(new int[query.rows * 1], query.rows, 1);
		flann::Matrix<float> distances(new float[query.rows * 1], query.rows, 1);

        //for (size_t i = 0; i < transformedPoints.size(); i++)
        //{
        //    std::cout << "Query: " << *query[i] << std::endl;
        //}
		
		// Do a knn search, searching for 1 nearest point and using 16 checks.
		flann::SearchParams searchParams{ 16 };
		searchParams.cores = 0;
		m_index->knnSearch(query, indices, distances, 1, searchParams);

		// Filter the matches.
		const unsigned nMatches = transformedPoints.size();
		std::vector<Match> matches;
		matches.reserve(nMatches);

        std::cout << m_maxDistance << std::endl;
		for (int i = 0; i < nMatches; ++i) {
			if (*distances[i] <= m_maxDistance)        
				matches.push_back(Match{ *indices[i], 1.f });
			else
				matches.push_back(Match{ -1, 0.f });
		}

        //for (int i = 0; i < nMatches; i++)
        //{
        //    std::cout << "Query: " << *query[i] << std::endl;
        //    std::cout << "Indices: " << *indices[i] << std::endl;
        //    std::cout << "Distances: " << *distances[i] << std::endl;
        //    std::cout << "Matches: " << matches[i].idx << std::endl;
        //}

		// Release the memory.
		delete[] query.ptr();
		delete[] indices.ptr();
		delete[] distances.ptr();

        //for (int i = 0; i < nMatches; i++)
        //{
        //    std::cout << "Matches: " << matches[i].idx << std::endl;
        //}

		return matches;
	}

    void setCameraParams(const Eigen::Matrix3f& depthIntrinsics, const unsigned width, const unsigned height){
        return;
    }

private:
	int m_nTrees;
	flann::Index<flann::L2<float>>* m_index;
	float* m_flatPoints;
    int type; // 0-> points. 1-> with colors 
};

// Project query source points to image target plane and find their closest neighbor by using a small search window // 
class NearestNeighborSearchProjective : public NearestNeighborSearch {
public:
	NearestNeighborSearchProjective(): searchWindow(5), height(0) {}

	~NearestNeighborSearchProjective() {
	}

	void buildIndex(const std::vector<Eigen::Vector3f>& targetPoints) {
		
        std::cout << "Initializing Projective index with " << targetPoints.size() << " points." << std::endl;
		
        m_points = targetPoints;
		
        std::cout << "Projective index created." << std::endl;
    }

	std::vector<Match> queryMatches(const std::vector<Vector3f>& transformedPoints) {

        if (this->m_points.size() == 0) {
			std::cout << "Projective index needs to be build before querying any matches." << std::endl;
			return {};
        }

        // No available camera params //
        if(this->height == 0){
			std::cout << "Set camera params before querying any matches." << std::endl;
            return {};
        }

        if(this->m_points.size() != (this->width * this->height)){
            std::cout << "Invalid size of target points." << std::endl;
            return {};
        }

        const unsigned nMatches = transformedPoints.size();
		const unsigned nTargetPoints = m_points.size();
		std::vector<Match> matches(nMatches);

		std::cout << "nTargetPoints: " << nTargetPoints << std::endl;
        std::cout << "nMatches: " << nMatches << std::endl;

        float fx, fy, mx, my; // Depth intrinsics
        int counterValid = 0; // Num of valid matches

        // Get depth intrinsics //
        fx = this->depthIntrinsics(0,0);
        fy = this->depthIntrinsics(1,1);
        mx = this->depthIntrinsics(0,2);
        my = this->depthIntrinsics(1,2);

        // For each source point find its closest neighbor //
        #pragma omp parallel for
		for (unsigned int i = 0; i < nMatches; i++) {

            // Invalid point //
            if(transformedPoints[i].x() == MINF)
                continue;

            unsigned int uPoint, vPoint; // Pixel coordinates
            
            // Tranfsorm camera coodinates to image coordinates of the current point // 
            uPoint = std::round(((transformedPoints[i].x() * fx) / transformedPoints[i].z()) + mx);
            vPoint = std::round(((transformedPoints[i].y() * fy) / transformedPoints[i].z()) + my);
       
            float minDist = std::numeric_limits<float>::max();
            unsigned int idx = -1; // Neighrest neighbor index

            // Scan neighbors and find the closest one //  
            for(unsigned int v = vPoint - this->searchWindow; (v >= 0 && v < this->height && v <= vPoint + this->searchWindow); v++){
                for(unsigned int u = uPoint - this->searchWindow; (u >= 0 && u < this->width && u <= uPoint + this->searchWindow); u++){

                    // Index of current neighbor // 
                    unsigned int neighborIndex = this->width * v + u;
                    
                    // Invalid neighbor point //
                    if(this->m_points[neighborIndex].x() == MINF)
                        continue;
                    
                    // Use squared distance // 
                    float dist = (transformedPoints[i] - this->m_points[neighborIndex]).squaredNorm();
                    
                    // Closest neighbor found //
                    if (minDist > dist) {
                        idx = neighborIndex;
                        minDist = dist;
                    }
                } // End for u
            } // End for v  
            
            // Add nearest neighbor for current (query) point //
            if (minDist <= m_maxDistance){
                matches[i].idx = idx;
                matches[i].weight = 1.f;
                counterValid++; 
            }
            else{
                matches[i].idx = -1;
                matches[i].weight = 0.f; 
            }
        } // End for i - Scan transformedPoints (query points)
       
		std::cout << "nValid matches: " << counterValid << std::endl;

        return matches;
	}
    
    // Parse camera parameters - required for querying points //
    void setCameraParams(const Eigen::Matrix3f& depthIntrinsics, const unsigned width, const unsigned height){
        this->depthIntrinsics = depthIntrinsics;
        this->width = width;
        this->height = height;
    }

private:
	std::vector<Eigen::Vector3f> m_points; // Target points
    unsigned searchWindow; // How many pixels to take into account during search
    unsigned width; // Img width
    unsigned height;
    Eigen::Matrix3f depthIntrinsics;
};
