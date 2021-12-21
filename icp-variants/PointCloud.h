#pragma once
#include "SimpleMesh.h"
#include "Eigen.h"

class PointCloud {
public:
    PointCloud() {}

    PointCloud(const SimpleMesh& mesh) {
        const auto& vertices = mesh.getVertices();
        const auto& triangles = mesh.getTriangles();
        const unsigned nVertices = vertices.size();
        const unsigned nTriangles = triangles.size();

        // Copy vertices.
        m_points.reserve(nVertices);
        for (const auto& vertex : vertices) {
            m_points.push_back(Vector3f{ vertex.position.x(), vertex.position.y(), vertex.position.z() });
        }

        // Compute normals (as an average of triangle normals).
        m_normals = std::vector<Vector3f>(nVertices, Vector3f::Zero());
        for (size_t i = 0; i < nTriangles; i++) {
            const auto& triangle = triangles[i];
            Vector3f faceNormal = (m_points[triangle.idx1] - m_points[triangle.idx0]).cross(m_points[triangle.idx2] - m_points[triangle.idx0]);

            m_normals[triangle.idx0] += faceNormal;
            m_normals[triangle.idx1] += faceNormal;
            m_normals[triangle.idx2] += faceNormal;
        }
        for (size_t i = 0; i < nVertices; i++) {
            m_normals[i].normalize();
        }
    }

    PointCloud(float* depthMap, const Matrix3f& depthIntrinsics, const Matrix4f& depthExtrinsics, const unsigned width, const unsigned height, unsigned downsampleFactor = 1, float maxDistance = 0.1f) {
        // Get depth intrinsics.
        float fovX = depthIntrinsics(0, 0);
        float fovY = depthIntrinsics(1, 1);
        float cX = depthIntrinsics(0, 2);
        float cY = depthIntrinsics(1, 2);
        const float maxDistanceHalved = maxDistance / 2.f;

        // Compute inverse depth extrinsics.
        Matrix4f depthExtrinsicsInv = depthExtrinsics.inverse();
        Matrix3f rotationInv = depthExtrinsicsInv.block(0, 0, 3, 3);
        Vector3f translationInv = depthExtrinsicsInv.block(0, 3, 3, 1);

        // Back-project the pixel depths into the camera space.
        std::vector<Vector3f> pointsTmp(width * height);

        // For every pixel row.
#pragma omp parallel for
        for (int v = 0; v < height; ++v) {
            // For every pixel in a row.
            for (int u = 0; u < width; ++u) {
                unsigned int idx = v * width + u; // linearized index
                float depth = depthMap[idx];
                if (depth == MINF) {
                    pointsTmp[idx] = Vector3f(MINF, MINF, MINF);
                }
                else {
                    // Back-projection to camera space.
                    pointsTmp[idx] = rotationInv * Vector3f((u - cX) / fovX * depth, (v - cY) / fovY * depth, depth) + translationInv;
                }
            }
        }

        // We need to compute derivatives and then the normalized normal vector (for valid pixels).
        std::vector<Vector3f> normalsTmp(width * height);

#pragma omp parallel for
        for (int v = 1; v < height - 1; ++v) {
            for (int u = 1; u < width - 1; ++u) {
                unsigned int idx = v * width + u; // linearized index

                const float du = 0.5f * (depthMap[idx + 1] - depthMap[idx - 1]);
                const float dv = 0.5f * (depthMap[idx + width] - depthMap[idx - width]);
                if (!std::isfinite(du) || !std::isfinite(dv) || abs(du) > maxDistanceHalved || abs(dv) > maxDistanceHalved) {
                    normalsTmp[idx] = Vector3f(MINF, MINF, MINF);
                    continue;
                }

                // TODO: Compute the normals using central differences.

                /* depthMap[x,y] -> normal = (1,0,ddepthMap/dx) x (0,1,ddepthMap/dy) */
                normalsTmp[idx] = Vector3f(-du, -dv, 1); // Needs to be replaced.
                normalsTmp[idx].normalize();
            }
        }

        // We set invalid normals for border regions.
        for (int u = 0; u < width; ++u) {
            normalsTmp[u] = Vector3f(MINF, MINF, MINF);
            normalsTmp[u + (height - 1) * width] = Vector3f(MINF, MINF, MINF);
        }
        for (int v = 0; v < height; ++v) {
            normalsTmp[v * width] = Vector3f(MINF, MINF, MINF);
            normalsTmp[(width - 1) + v * width] = Vector3f(MINF, MINF, MINF);
        }

        // We filter out measurements where either point or normal is invalid.
        const unsigned nPoints = pointsTmp.size();
        m_points.reserve(std::floor(float(nPoints) / downsampleFactor));
        m_normals.reserve(std::floor(float(nPoints) / downsampleFactor));

        for (int i = 0; i < nPoints; i = i + downsampleFactor) {
            const auto& point = pointsTmp[i];
            const auto& normal = normalsTmp[i];

            if (point.allFinite() && normal.allFinite()) {
                m_points.push_back(point);
                m_normals.push_back(normal);
            }
        }
    }

    bool readFromFile(const std::string& filename) {
        std::ifstream is(filename, std::ios::in | std::ios::binary);
        if (!is.is_open()) {
            std::cout << "ERROR: unable to read input file!" << std::endl;
            return false;
        }

        char nBytes;
        is.read(&nBytes, sizeof(char));

        unsigned int n;
        is.read((char*)&n, sizeof(unsigned int));

        if (nBytes == sizeof(float)) {
            float* ps = new float[3 * n];

            is.read((char*)ps, 3 * sizeof(float) * n);

            for (unsigned int i = 0; i < n; i++) {
                Eigen::Vector3f p(ps[3 * i + 0], ps[3 * i + 1], ps[3 * i + 2]);
                m_points.push_back(p);
            }

            is.read((char*)ps, 3 * sizeof(float) * n);
            for (unsigned int i = 0; i < n; i++) {
                Eigen::Vector3f p(ps[3 * i + 0], ps[3 * i + 1], ps[3 * i + 2]);
                m_normals.push_back(p);
            }

            delete ps;
        }
        else {
            double* ps = new double[3 * n];

            is.read((char*)ps, 3 * sizeof(double) * n);

            for (unsigned int i = 0; i < n; i++) {
                Eigen::Vector3f p((float)ps[3 * i + 0], (float)ps[3 * i + 1], (float)ps[3 * i + 2]);
                m_points.push_back(p);
            }

            is.read((char*)ps, 3 * sizeof(double) * n);

            for (unsigned int i = 0; i < n; i++) {
                Eigen::Vector3f p((float)ps[3 * i + 0], (float)ps[3 * i + 1], (float)ps[3 * i + 2]);
                m_normals.push_back(p);
            }

            delete ps;
        }


        //std::ofstream file("pointcloud.off");
        //file << "OFF" << std::endl;
        //file << m_points.size() << " 0 0" << std::endl;
        //for(unsigned int i=0; i<m_points.size(); ++i)
        //	file << m_points[i].x() << " " << m_points[i].y() << " " << m_points[i].z() << std::endl;
        //file.close();

        return true;
    }

    std::vector<Vector3f>& getPoints() {
        return m_points;
    }

    const std::vector<Vector3f>& getPoints() const {
        return m_points;
    }

    std::vector<Vector3f>& getNormals() {
        return m_normals;
    }

    const std::vector<Vector3f>& getNormals() const {
        return m_normals;
    }
    
    std::vector<Vector3uc>& getColors() {
        return m_colors;
    }

    const std::vector<Vector3uc>& getColors() const {
        return m_colors;
    }

    unsigned int getClosestPoint(Vector3f& p) {
        unsigned int idx = 0;

        float min_dist = std::numeric_limits<float>::max();
        for (unsigned int i = 0; i < m_points.size(); ++i) {
            float dist = (p - m_points[i]).norm();
            if (min_dist > dist) {
                idx = i;
                min_dist = dist;
            }
        }

        return idx;
    }

private:
    std::vector<Vector3f> m_points;
    std::vector<Vector3f> m_normals;
    std::vector<Vector3uc> m_colors;

};
