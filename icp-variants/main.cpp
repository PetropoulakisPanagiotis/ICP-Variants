#include <iostream>
#include <fstream>


#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
#include "BunnyDataLoader.h"
#include "ETHDataLoader.h"
#include "ConvergenceMeasure.h"
#include "selection.h"
#include "TimeMeasure.h"

#define SHOW_BUNNY_CORRESPONDENCES 1

#define MATCHING_METHOD     1 // 1 -> projective, 0 -> knn
#define SELECTION_METHOD    0 // 0 -> all, 1 -> random
#define WEIGHTING_METHOD    2 // 0 -> constant, 1 -> point distances, 2 -> normals, 3 -> colors, 4-> hybrid

#define USE_LINEAR_ICP		0 // Optimization method

#define USE_POINT_TO_PLANE	0 // Objectives - Set only one to true 
#define USE_SYMMETRIC	    1

#define RUN_SHAPE_ICP		1
#define RUN_SEQUENCE_ICP	0
#define RUN_ETH_ICP			0

int alignBunnyWithICP() {
	// Load the source and target mesh.
	BunnyDataLoader bunny_data_loader{};

	// Estimate the pose from source to target mesh with ICP optimization.
	ICPOptimizer* optimizer = nullptr;
	
    // 5. Set minimization method //
    if (USE_LINEAR_ICP) {
		optimizer = new LinearICPOptimizer();
	}
	else {
		optimizer = new CeresICPOptimizer();
	}
	

    // 6. Set objective // 
	if (USE_POINT_TO_PLANE) {
		optimizer->setMetric(1);
		optimizer->setNbOfIterations(20);
	}
	else if (USE_SYMMETRIC) {
		optimizer->setMetric(2);
		optimizer->setNbOfIterations(20);
	}
	else {
		optimizer->setMetric(0);
		optimizer->setNbOfIterations(20);
	}

    // 1. Set matching set  //
    // Always knn for bunny //
    optimizer->setMatchingMethod(0);
	optimizer->setMatchingMaxDistance(0.0003f);

    // 2. Set selection method //
    if(SELECTION_METHOD)
	    optimizer->setSelectionMethod(RANDOM_SAMPLING);
    else
	    optimizer->setSelectionMethod(SELECT_ALL);

    // 3. Set weighting method //
    if(WEIGHTING_METHOD == 1)
        optimizer->setWeightingMethod(DISTANCES_WEIGHTING);
    else if(WEIGHTING_METHOD == 2)
        optimizer->setWeightingMethod(NORMALS_WEIGHTING);
    else if(WEIGHTING_METHOD == 3)
        optimizer->setWeightingMethod(COLORS_WEIGHTING);
    else if(WEIGHTING_METHOD == 4)
        optimizer->setWeightingMethod(HYBRID_WEIGHTING);
    else
        optimizer->setWeightingMethod(CONSTANT_WEIGHTING);

	// load the sample
	Sample input = bunny_data_loader.getItem(0);
    Matrix4f estimatedPose = Matrix4f::Identity();

	// Example ground truth correspondences
	// Fill in the matched points: sourcePoints[i] is matched with targetPoints[i].
	// 215 294 -0.0512466 0.0956544 0.0436517 -0.051901 0.095458 0.043938
	// 424 258 -0.0161308 0.0873282 0.056647 -0.016683 0.087267 0.056741
	// 640 1238 0.0429302 0.045474 0.0291547 0.042297 0.045436 0.029041
	// 1023 1310 -0.00232282 0.0349611 0.0453906 -0.002826 0.034885 0.045611
	std::vector<Vector3f> gtSourcePoints;
	gtSourcePoints.push_back(input.source.getPoints()[215]); // left ear
	gtSourcePoints.push_back(input.source.getPoints()[424]); // left ear
	gtSourcePoints.push_back(input.source.getPoints()[640]); // left ear
	gtSourcePoints.push_back(input.source.getPoints()[1023]); // left ear

	std::vector<Vector3f> gtTargetPoints;
	gtTargetPoints.push_back(input.target.getPoints()[294]); // left ear
	gtTargetPoints.push_back(input.target.getPoints()[258]); // left ear
	gtTargetPoints.push_back(input.target.getPoints()[1238]); // left ear
	gtTargetPoints.push_back(input.target.getPoints()[1310]); // left ear

	// Create a Convergence Measure
	auto convergenMearsure = ConvergenceMeasure(gtSourcePoints, gtTargetPoints);
	optimizer->setConvergenceMeasure(convergenMearsure);

	// Create a Time Profiler
	auto timeMeasure = TimeMeasure();
	optimizer->setTimeMeasure(timeMeasure);
	
	// Estimate pose
	optimizer->estimatePose(input.source, input.target, estimatedPose);
	
	// Calculate convergence measure
	auto alignmentError = convergenMearsure.rmseAlignmentError(estimatedPose);
	std::cout << "RMSE Alignment error of Final transform: " << alignmentError << std::endl;

	// Calculate time
	timeMeasure.calculateIterationTime();

	// Print out RMSE errors of each iteration
	convergenMearsure.outputAlignmentError();
	
	// Visualize the resulting joined mesh. We add triangulated spheres for point matches.
	SimpleMesh resultingMesh = SimpleMesh::joinMeshes(bunny_data_loader.getSourceMesh(), bunny_data_loader.getTargetMesh(), estimatedPose);
	if (SHOW_BUNNY_CORRESPONDENCES) {
		for (const auto& sourcePoint : input.source.getPoints()) {
			resultingMesh = SimpleMesh::joinMeshes(SimpleMesh::sphere(sourcePoint, 0.001f), resultingMesh, estimatedPose);
		}
		for (const auto& targetPoint : input.target.getPoints()) {
			resultingMesh = SimpleMesh::joinMeshes(SimpleMesh::sphere(targetPoint, 0.001f, Vector4uc(255, 255, 255, 255)), resultingMesh, Matrix4f::Identity());
		}

		// Show ground truth correspondences
		for (const auto& sourcePoint : gtSourcePoints) {
			resultingMesh = SimpleMesh::joinMeshes(SimpleMesh::sphere(sourcePoint, 0.003f, Vector4uc(0, 255, 0, 255)), resultingMesh, estimatedPose);
		}
		for (const auto& targetPoint : gtTargetPoints) {
			resultingMesh = SimpleMesh::joinMeshes(SimpleMesh::sphere(targetPoint, 0.003f, Vector4uc(255, 0, 255, 0)), resultingMesh, Matrix4f::Identity());
		}
	}
	resultingMesh.writeMesh(std::string("bunny_icp.off"));
	std::cout << "Resulting mesh written." << std::endl;

	delete optimizer;

	return 0;
}

int reconstructRoom() {
	std::string filenameIn = std::string("../../Data/rgbd_dataset_freiburg1_xyz/");
	std::string filenameBaseOut = std::string("mesh_");

	// Load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.init(filenameIn)) {
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

	// We store a first frame as a reference frame. All next frames are tracked relatively to the first frame.
	sensor.processNextFrame();
	PointCloud target{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), true};

    // Setup the optimizer.
	ICPOptimizer* optimizer = nullptr;

    // 5. Set minimization method //
	if (USE_LINEAR_ICP) {
		optimizer = new LinearICPOptimizer();
	}
	else {
		optimizer = new CeresICPOptimizer();
	}

    // 6. Set objective //
    if (USE_POINT_TO_PLANE) {
		optimizer->setMetric(1);
		optimizer->setNbOfIterations(10);
	}
	else {
		optimizer->setMetric(0);
		optimizer->setNbOfIterations(20);
	}

    // 1. Set matching step //
    if(MATCHING_METHOD){
        optimizer->setMatchingMethod(1);
        optimizer->setCameraParamsMatchingMethod(sensor.getDepthIntrinsics(),sensor.getDepthImageWidth(), sensor.getDepthImageHeight());
    }

    optimizer->setMatchingMaxDistance(0.1f);

    // 2. Set selection method //
    if(SELECTION_METHOD)
	    optimizer->setSelectionMethod(RANDOM_SAMPLING);
    else
	    optimizer->setSelectionMethod(SELECT_ALL);

    // 3. Set weighting method //
    if(WEIGHTING_METHOD == 1)
        optimizer->setWeightingMethod(DISTANCES_WEIGHTING);
    else if(WEIGHTING_METHOD == 2)
        optimizer->setWeightingMethod(NORMALS_WEIGHTING);
    else if(WEIGHTING_METHOD == 3)
        optimizer->setWeightingMethod(COLORS_WEIGHTING);
    else if(WEIGHTING_METHOD == 4)
        optimizer->setWeightingMethod(HYBRID_WEIGHTING);
    else
        optimizer->setWeightingMethod(CONSTANT_WEIGHTING);

    // Create a Time Profiler
	auto timeMeasure = TimeMeasure();
	optimizer->setTimeMeasure(timeMeasure);

    // We store the estimated camera poses.
	std::vector<Matrix4f> estimatedPoses;
	Matrix4f currentCameraToWorld = Matrix4f::Identity();
	estimatedPoses.push_back(currentCameraToWorld.inverse());

	int i = 0;
	const int iMax = 50; //50
	while (sensor.processNextFrame() && i <= iMax) {
		float* depthMap = sensor.getDepth();
		Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();
		Matrix4f depthExtrinsics = sensor.getDepthExtrinsics();

		// Estimate the current camera pose from source to target mesh with ICP optimization.
		// We downsample the source image to speed up the correspondence matching.
		PointCloud source{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), false, 8 };
        optimizer->estimatePose(source, target, currentCameraToWorld);
		
		// Invert the transformation matrix to get the current camera pose.
		Matrix4f currentCameraPose = currentCameraToWorld.inverse();
		std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;
		estimatedPoses.push_back(currentCameraPose);

		if (i % 5 == 0) {
			// We write out the mesh to file for debugging.
			SimpleMesh currentDepthMesh{ sensor, currentCameraPose, 0.1f };
			SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraPose, 0.0015f);
			SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

			std::stringstream ss;
			ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
			std::cout << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off" << std::endl;
			if (!resultingMesh.writeMesh(ss.str())) {
				std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
				return -1;
			}
		}
		
		i++;
	}

	delete optimizer;

	return 0;
}

int alignETH() {
	// Load the source and target mesh.
	ETHDataLoader eth_data_loader{};
	Sample s = eth_data_loader.getItem(0);
	s.source.writeToFile("source.ply");
	s.target.writeToFile("target.ply");
	return 0;
}

int main() {
	int result = 0;
	if (RUN_SHAPE_ICP)
		result += alignBunnyWithICP();
	if (RUN_SEQUENCE_ICP)
		result += reconstructRoom();
	if (RUN_ETH_ICP)
		result += alignETH();

	return result;
}
