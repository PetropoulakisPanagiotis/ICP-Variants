#pragma once
#include "Eigen.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

/**
 * Helper methods for writing Ceres cost functions.
 */
template <typename T>
static inline void fillVector(const Vector3f& input, T* output) {
    output[0] = T(input[0]);
    output[1] = T(input[1]);
    output[2] = T(input[2]);
}


/**
 * Pose increment is only an interface to the underlying array (in constructor, no copy
 * of the input array is made).
 * Important: Input array needs to have a size of at least 6.
 */
// array -> rx, ry, rz, tx, ty, tz
// conver to matrix -> matrix
// apply  
template <typename T>
class PoseIncrement {
public:
    explicit PoseIncrement(T* const array) : m_array{ array } { }

    void setZero() {
        for (int i = 0; i < 6; ++i)
            m_array[i] = T(0);
    }

    T* getData() const {
        return m_array;
    }

    /**
     * Applies the pose increment onto the input point and produces transformed output point.
     * Important: The memory for both 3D points (input and output) needs to be reserved (i.e. on the stack)
     * beforehand).
     */
    void apply(T* inputPoint, T* outputPoint) const {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        const T* rotation = m_array;
        const T* translation = m_array + 3;

        T temp[3];
        ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

        outputPoint[0] = temp[0] + translation[0];
        outputPoint[1] = temp[1] + translation[1];
        outputPoint[2] = temp[2] + translation[2];
    }

    /**
     * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
     * transformation 4x4 matrix.
     */
    static Matrix4f convertToMatrix(const PoseIncrement<double>& poseIncrement) {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        double* pose = poseIncrement.getData();
        double* rotation = pose;
        double* translation = pose + 3;

        // Convert the rotation from SO3 to matrix notation (with column-major storage).
        double rotationMatrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

        // Create the 4x4 transformation matrix.
        Matrix4f matrix;
        matrix.setIdentity();
        matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
        matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
        matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);

        return matrix;
    }

private:
    T* m_array;
};

/* Transform points given a pose
*/
std::vector<Vector3f> transformPoints(const std::vector<Vector3f>& sourcePoints, const Matrix4f& pose) {
    std::vector<Vector3f> transformedPoints;
    transformedPoints.reserve(sourcePoints.size());

    const auto rotation = pose.block(0, 0, 3, 3);
    const auto translation = pose.block(0, 3, 3, 1);

    for (const auto& point : sourcePoints) {
        transformedPoints.push_back(rotation * point + translation);
    }

    return transformedPoints;
}

/* Transfrom normals given a pose
*/
std::vector<Vector3f> transformNormals(const std::vector<Vector3f>& sourceNormals, const Matrix4f& pose) {
    std::vector<Vector3f> transformedNormals;
    transformedNormals.reserve(sourceNormals.size());

    const auto rotation = pose.block(0, 0, 3, 3);

    for (const auto& normal : sourceNormals) {
        transformedNormals.push_back(rotation.inverse().transpose() * normal);
    }

    return transformedNormals;
}

// Compute mean of points
Vector3f computeMean(const std::vector<Vector3f>& points) {
    // TODO: Assert num points > 0
    ASSERT(points.size() > 0 && "Number of points must be positive.");
    Vector3f mean = Vector3f::Zero();
    for (auto point: points) {
        mean += point;
    }
    mean /= points.size();
    return mean;
}

// Derive translation matrix 
// T * (x 1) = (x+t 1)
Matrix4f gettranslationMatrix(const Vector3f& translation) {
    // Start with identity 4x4 matrix and fill in last column
    Matrix4f matrix = Matrix4f::Identity();
    matrix(0,3) = translation[0];
    matrix(1,3) = translation[1];
    matrix(2,3) = translation[2];
    return matrix;
}

// Derive cross product matrix
// aka. k x v = K * v
Matrix3f crossProductMatrix(const Vector3f& k) {
    Matrix3f matrix = Matrix3f::Zero(); // Not Identity but Zero
    // Fill in value
                        matrix(0,1) = -k.z(); matrix(0,2) = k.y();
    matrix(1,0) = k.z();                        matrix(1,2) = -k.x();
    matrix(2,0) = -k.y(); matrix(2,1) = k.x();
    return matrix;
}

// Derive rodrigues matrix from axis and rotation angle
// R = I + sin(theta) * K + (1 - cos(theta)) * K * K
Matrix3f getRodriguesMatrix(const Vector3f& axis, const float& sin_theta, const float& cos_theta) {
    Matrix3f matrix = Matrix3f::Identity();
    Matrix3f K = crossProductMatrix(axis);
    matrix += sin_theta * K + (1 - cos_theta) * K * K;
    return matrix;
}