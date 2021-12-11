#pragma once
//#include <nanoflann.hpp>
#include <flann/flann.hpp>

#include "Eigen.h"

struct Match {
	int idx;
	float weight;
};

class NearestNeighborSearch {
public:
	virtual ~NearestNeighborSearch() {}

	virtual void setMatchingMaxDistance(float maxDistance) {
		m_maxDistance = maxDistance;
	}

	virtual void buildIndex(const std::vector<Eigen::Vector3f>& targetPoints) = 0;
	virtual std::vector<Match> queryMatches(const std::vector<Vector3f>& transformedPoints) = 0;

protected:
	float m_maxDistance;

	NearestNeighborSearch() : m_maxDistance{ 0.005f } {}
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
	}

	std::vector<Match> queryMatches(const std::vector<Vector3f>& transformedPoints) {
        if (!m_index) {
			std::cout << "FLANN index needs to be build before querying any matches." << std::endl;
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

private:
	int m_nTrees;
	flann::Index<flann::L2<float>>* m_index;
	float* m_flatPoints;
};


