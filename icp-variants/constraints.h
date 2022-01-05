#pragma once
#include <ceres/ceres.h>
#include "Eigen.h"
#include "utils.h"

/**
 * Optimization constraints.
 */
class PointToPointConstraint {
public:
    PointToPointConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint, const float weight) :
        m_sourcePoint{ sourcePoint },
        m_targetPoint{ targetPoint },
        m_weight{ weight }
    { }

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // TODO: Implemented the point-to-point cost function.
        // The resulting 3D residual should be stored in residuals array. To apply the pose 
        // increment (pose parameters) to the source point, you can use the PoseIncrement
        // class.
        // Important: Ceres automatically squares the cost function.

        /*  Use poseIncrement to apply transformation to source point */
        PoseIncrement<T> poseIncrement = PoseIncrement<T>((T* const)pose);

        /* Use casting for m_sourcePoint and apply transformation */
        T m_sourcePointTemp[3] = {(T)this->m_sourcePoint[0], (T)this->m_sourcePoint[1], (T)this->m_sourcePoint[2]}; 
        T m_sourcePointTransformed[3] = {(T)0.0, (T)0.0, (T)0.0}; 

        poseIncrement.apply(m_sourcePointTemp, m_sourcePointTransformed);

        residuals[0] = (T)this-> LAMBDA * (T)this->m_weight * (m_sourcePointTransformed[0] - (T)this->m_targetPoint[0]);
		residuals[1] = (T)this-> LAMBDA * (T)this->m_weight * (m_sourcePointTransformed[1] - (T)this->m_targetPoint[1]);
		residuals[2] = (T)this-> LAMBDA * (T)this->m_weight * (m_sourcePointTransformed[2] - (T)this->m_targetPoint[2]);

        return true;
    }

    static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector3f& targetPoint, const float weight) {
        return new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
            new PointToPointConstraint(sourcePoint, targetPoint, weight)
            );
    }

protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const float m_weight;
    const float LAMBDA = 0.1f;
};

class PointToPlaneConstraint {
public:
    PointToPlaneConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint, const Vector3f& targetNormal, const float weight) :
        m_sourcePoint{ sourcePoint },
        m_targetPoint{ targetPoint },
        m_targetNormal{ targetNormal },
        m_weight{ weight }
    { }

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // TODO: Implemented the point-to-plane cost function.
        // The resulting 1D residual should be stored in residuals array. To apply the pose 
        // increment (pose parameters) to the source point, you can use the PoseIncrement
        // class.
        // Important: Ceres automatically squares the cost function.

        /*  Use poseIncrement to apply transformation to source point */
        PoseIncrement<T> poseIncrement = PoseIncrement<T>((T* const)pose);

        /* Use casting for m_sourcePoint and apply transformation */
        T m_sourcePointTemp[3] = {(T)this->m_sourcePoint[0], (T)this->m_sourcePoint[1], (T)this->m_sourcePoint[2]}; 
        T m_sourcePointTransformed[3] = {(T)0.0, (T)0.0, (T)0.0}; 

        /* Use casting for m_sourcePoint and apply transformation */
        poseIncrement.apply(m_sourcePointTemp, m_sourcePointTransformed);

        T x_component = (T)this->m_targetNormal[0] * (m_sourcePointTransformed[0] - (T)this->m_targetPoint[0]);
        T y_component = (T)this->m_targetNormal[1] * (m_sourcePointTransformed[1] - (T)this->m_targetPoint[1]);
        T z_component = (T)this->m_targetNormal[2] * (m_sourcePointTransformed[2] - (T)this->m_targetPoint[2]);

        residuals[0] = (T)this->LAMBDA * (T)this->m_weight * (x_component + y_component + z_component);

        return true;
    }

    static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector3f& targetPoint, const Vector3f& targetNormal, const float weight) {
        return new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
            new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, weight)
            );
    }

protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const Vector3f m_targetNormal;
    const float m_weight;
    const float LAMBDA = 1.0f;
};


class SymmetricConstraint{
public:
    SymmetricConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint, const Vector3f& sourceNormal, const Vector3f& targetNormal) :
        m_sourcePoint{ sourcePoint },
        m_targetPoint{ targetPoint },
        m_sourceNormal{ sourceNormal },
        m_targetNormal{ targetNormal }
    { }

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {

        /*  Use poseIncrement to apply transformation to source point */
        PoseIncrement<T> poseIncrement = PoseIncrement<T>((T* const)pose);

        /* Use casting for m_sourcePoint and apply transformation */
        T m_sourcePointTemp[3] = { (T)this->m_sourcePoint[0], (T)this->m_sourcePoint[1], (T)this->m_sourcePoint[2] };
        T m_targetPointTemp[3] = { (T)this->m_targetPoint[0], (T)this->m_targetPoint[1], (T)this->m_targetPoint[2] };
        T m_sourcePointTransformed[3] = { (T)0.0, (T)0.0, (T)0.0 };
        T m_targetPointTransformed[3] = { (T)0.0, (T)0.0, (T)0.0 };

        /* Use casting for m_sourcePoint and apply transformation */
        poseIncrement.apply(m_sourcePointTemp, m_sourcePointTransformed);
        poseIncrement.apply_inv_rotation(m_targetPointTemp, m_targetPointTransformed);

        T x_component = ((T)this->m_targetNormal[0] + (T)this->m_sourceNormal[0]) * (m_sourcePointTransformed[0] - m_targetPointTransformed[0]);
        T y_component = ((T)this->m_targetNormal[1] + (T)this->m_sourceNormal[1]) * (m_sourcePointTransformed[1] - m_targetPointTransformed[1]);
        T z_component = ((T)this->m_targetNormal[2] + (T)this->m_sourceNormal[2]) * (m_sourcePointTransformed[2] - m_targetPointTransformed[2]);

        residuals[0] = (T)this->LAMBDA * (x_component + y_component + z_component);

        return true;
    }

    static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector3f& targetPoint, const Vector3f& sourceNormal, const Vector3f& targetNormal) {
        return new ceres::AutoDiffCostFunction<SymmetricConstraint, 1, 6>(
            new SymmetricConstraint(sourcePoint, targetPoint, sourceNormal, targetNormal)
            );
    }

protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const Vector3f m_sourceNormal;
    const Vector3f m_targetNormal;
    const float LAMBDA = 1.0f;
};

