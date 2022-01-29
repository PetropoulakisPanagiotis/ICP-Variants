#include <iostream>
#include <fstream>
#include <limits>


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

int alignBunnyWithICP(unsigned useLinear, unsigned useMetric, unsigned matchingMethod, unsigned selectionMethod, unsigned weightingMethod, 
		unsigned useMultiresolution, unsigned numIterations=20, float maxMatchingDist=0.01f, float samplingProba=0.5f, std::string expName="bunny") {
    // ASSERT arguments
	printf("Running Bunny ICP with useLinear %d, useMetric %d, matchingMethod %d, selectionMethod %d, weightingMethod %d, useMultiresolution %d, numIterations %d, maxMatchingDist %f, samplingProba %f, expName %s.\n", 
			useLinear , useMetric , matchingMethod , selectionMethod , weightingMethod , useMultiresolution , numIterations , maxMatchingDist, samplingProba, expName.c_str());
	// ASSERT(useLinear < 2 &&  useMetric < 3 && matchingMethod < 2  && selectionMethod < 2  && weightingMethod < 4  && useMultiresolution < 2  && "Config unsupported.");

	// Load the source and target mesh.
	BunnyDataLoader bunny_data_loader{};

	// Estimate the pose from source to target mesh with ICP optimization.
	ICPOptimizer* optimizer = nullptr;
	
    // 5. Set minimization method //
    if (useLinear) 
		optimizer = new LinearICPOptimizer();
	else 
		optimizer = new CeresICPOptimizer();

    // 6. Set objective // 
    optimizer->setMetric(useMetric);
    optimizer->setNbOfIterations(numIterations);

    // 1. Set matching set  //
    // Always knn for bunny //
    optimizer->setMatchingMethod(0);
	optimizer->setMatchingMaxDistance(maxMatchingDist);

    // 2. Set selection method //
    optimizer->setSelectionMethod(selectionMethod, samplingProba);

    // 3. Set weighting method //
    optimizer->setWeightingMethod(weightingMethod);

    if(useMultiresolution)
        optimizer->enableMultiResolution(true);

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
	gtSourcePoints.push_back(input.source.getPoints()[215]); 
	gtSourcePoints.push_back(input.source.getPoints()[424]); 
	gtSourcePoints.push_back(input.source.getPoints()[640]); 
	gtSourcePoints.push_back(input.source.getPoints()[1023]); 

	std::vector<Vector3f> gtTargetPoints;
	gtTargetPoints.push_back(input.target.getPoints()[294]); 
	gtTargetPoints.push_back(input.target.getPoints()[258]); 
	gtTargetPoints.push_back(input.target.getPoints()[1238]); 
	gtTargetPoints.push_back(input.target.getPoints()[1310]); 

	// Create a Convergence Measure
	auto convergenMearsure = ConvergenceMeasure(gtSourcePoints, gtTargetPoints);
	optimizer->setConvergenceMeasure(convergenMearsure);

	// Create a Time Profiler
	auto timeMeasure = TimeMeasure();
	optimizer->setTimeMeasure(timeMeasure);
	
	// Estimate pose
	std::cout << "num points source:" << input.source.getPoints().size() << std::endl;
	std::cout << "num points target:" << input.target.getPoints().size() << std::endl;
	optimizer->estimatePose(input.source, input.target, estimatedPose);
	
	// Calculate convergence measure
	auto alignmentError = convergenMearsure.rmseAlignmentError(estimatedPose);
	std::cout << "RMSE Alignment error of Final transform: " << alignmentError << std::endl;

	// Calculate time
	timeMeasure.calculateIterationTime();

	std::cout << "estimatedPose:\n" << estimatedPose << std::endl;

	input.source.writeToFile(expName + "_bunny_source.ply");
	input.target.writeToFile(expName + "_bunny_target.ply");
	PointCloud transformed_source = input.source.copy_point_cloud();
	transformed_source.change_pose(estimatedPose);
	transformed_source.writeToFile(expName + "_bunny_final_source.ply");
  
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

    // std::string outFile = "bunny_icp_" + std::to_string(useLinear) + std::to_string(useMetric) + ".off";
	resultingMesh.writeMesh(expName + std::string("_bunny_icp.off"));
	std::cout << "Resulting mesh written." << std::endl;

    // saving iteration errors to file //
    convergenMearsure.writeRMSEToFile(expName + std::string("_RMSE.txt"));

	delete optimizer;

	return 0;
}

int reconstructRoom(unsigned useLinear, unsigned useMetric, unsigned matchingMethod, unsigned selectionMethod, unsigned weightingMethod, 
        unsigned useMultiresolution, unsigned numIterations=20, float maxMatchingDist=0.01f, float samplingProba=0.5f, std::string expName="room") {
	printf("Running Room ICP with useLinear %d, useMetric %d, matchingMethod %d, selectionMethod %d, weightingMethod %d, useMultiresolution %d, numIterations %d, maxMatchingDist %f, samplingProba %f, expName %s.\n", 
			useLinear , useMetric , matchingMethod , selectionMethod , weightingMethod , useMultiresolution , numIterations , maxMatchingDist, samplingProba, expName.c_str());
	// ASSERT arguments
	// ASSERT(useLinear < 2 &&  useMetric <3 && matchingMethod <2  && selectionMethod < 2  && weightingMethod < 4  && useMultiresolution < 2  && samplingProba <= 1 && "Config not supported.");

	std::string filenameIn = std::string("../../../Data/rgbd_dataset_freiburg1_xyz/");
	std::string filenameBaseOut = std::string(expName + "_mesh_");

	// Load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	int frameStep = 10;
	VirtualSensor sensor = VirtualSensor(frameStep); // Increase step
	if (!sensor.init(filenameIn)) {
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

	// We store a first frame as a reference frame. All next frames are tracked relatively to the first frame.
	// sensor.processNextFrame();
	sensor.processFrameIndex(0);

    // For projective search keep the whole target point cloud //
    // even the invalid points                                 //
    bool keepOriginalSize = false;
    if(matchingMethod)
        keepOriginalSize = true;

	PointCloud target{ sensor.getDepth(), sensor.getColorRGBX(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), keepOriginalSize};
	Matrix4f targetTrajectory = sensor.getTrajectory();
	// std::cout << "Target trajectory:\n" << targetTrajectory << "\n";

    // Estimate the pose from source to target mesh with ICP optimization.
	ICPOptimizer* optimizer = nullptr;
	
    // 5. Set minimization method //
    if (useLinear) 
		optimizer = new LinearICPOptimizer();
	else 
		optimizer = new CeresICPOptimizer();

    // 6. Set objective // 
    optimizer->setMetric(useMetric);
    optimizer->setNbOfIterations(numIterations);

    // 1. Set matching set  //
    // Always knn for bunny //
    optimizer->setMatchingMethod(0);
	optimizer->setMatchingMaxDistance(maxMatchingDist);

    // 2. Set selection method //
    optimizer->setSelectionMethod(selectionMethod);

    // 3. Set weighting method //
    optimizer->setWeightingMethod(weightingMethod);

    if(useMultiresolution)
        optimizer->enableMultiResolution(true);

    // Create a Time Profiler
	auto timeMeasure = TimeMeasure();
	optimizer->setTimeMeasure(timeMeasure);

    // We store the estimated camera poses.
	std::vector<Matrix4f> estimatedPoses;
	Matrix4f currentCameraToWorld = Matrix4f::Identity();
	estimatedPoses.push_back(currentCameraToWorld.inverse());

	// Save target
	if (saveRoomToFile(sensor, currentCameraToWorld.inverse(), filenameBaseOut) == -1)
				return -1;

	// std::vector<float> errorsFinalIteration;
	int i = 0;
	const int iMax = 10; //50
	// while (sensor.processNextFrame() && i <= iMax) {
	while (sensor.processFrameIndex((i + 1) * frameStep) && i <= iMax) {
        float* depthMap = sensor.getDepth();
		Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();
		Matrix4f depthExtrinsics = sensor.getDepthExtrinsics();

		// Estimate the current camera pose from source to target mesh with ICP optimization.
	    PointCloud source; 	
        // For multiresolution keep all the points //
        if(useMultiresolution)
            source = PointCloud(sensor.getDepth(), sensor.getColorRGBX(),sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), true, 1 );
		else
            source = PointCloud(sensor.getDepth(), sensor.getColorRGBX(),sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), false, 8 );
        
		// Get transform from current to frame 0 coordinate as ground truth
		Matrix4f trajectoryInv = sensor.getTrajectory().inverse(); // Inverse to world coordinate
		Matrix4f currentToZeroCoordinates = targetTrajectory * trajectoryInv;
		auto gtTargetPoints = transformPoints(source.getPoints(), currentToZeroCoordinates);
		std::cout << "Ground Truth current trajectory to target trajectory transform:\n" << currentToZeroCoordinates << "\n";

		// Covergence Measure RMSE with ground truth correspondences
		auto convergenMearsure = ConvergenceMeasure(source.getPoints(), gtTargetPoints, false);
		optimizer->setConvergenceMeasure(convergenMearsure);
		auto initial_rmse = convergenMearsure.rmseAlignmentError(currentCameraToWorld);

        // Apply ICP //        
        optimizer->estimatePose(source, target, currentCameraToWorld, true);
        // ICP Finished //        

		std::cout << "Initial RMSE:" << initial_rmse << std::endl;
        std::cout << "Final RMSE:" << convergenMearsure.rmseAlignmentError(currentCameraToWorld) << std::endl;

		// Print out RMSE and benchmark errors of each iteration
		convergenMearsure.outputAlignmentError();

		// saving iteration errors to file //
		convergenMearsure.writeRMSEToFile(expName + "_RMSE" + std::to_string(i)+ ".txt");
		
        // Invert the transformation matrix to get the current camera pose.
		Matrix4f currentCameraPose = currentCameraToWorld.inverse();
		std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;
		estimatedPoses.push_back(currentCameraPose);

		if (i % 5 == 0) {
			// We write out the mesh to file for debugging.
			if (saveRoomToFile(sensor, currentCameraPose, filenameBaseOut) == -1)
				return -1;
		}
		
		i++;
	}

	delete optimizer;

	return 0;
}

int alignETH(unsigned useLinear, unsigned useMetric, unsigned matchingMethod, unsigned selectionMethod, unsigned weightingMethod, 
        unsigned useMultiresolution, unsigned numIterations=20, float maxMatchingDist=0.1f, float samplingProba=0.01f, std::string expName="benchmark") {
	printf("Running Benchmark ICP with useLinear %d, useMetric %d, matchingMethod %d, selectionMethod %d, weightingMethod %d, useMultiresolution %d, numIterations %d, maxMatchingDist %f, samplingProba %f, expName %s.\n", useLinear , useMetric , matchingMethod , selectionMethod , weightingMethod , useMultiresolution , numIterations , maxMatchingDist, samplingProba, expName.c_str());
	// ASSERT arguments
	// ASSERT(useLinear < 2 &&  useMetric <3 && matchingMethod <2  && selectionMethod < 2  && weightingMethod < 4  && useMultiresolution < 2  && samplingProba <= 1 && "Config not supported.");

	ICPOptimizer* optimizer = nullptr;
	// 5. Set minimization method //
    if (useLinear) 
		optimizer = new LinearICPOptimizer();
	else 
		optimizer = new CeresICPOptimizer();

    // 6. Set objective // 
    optimizer->setMetric(useMetric);
    optimizer->setNbOfIterations(numIterations);

    // 1. Set matching set  //
    // Always knn for bunny //
    optimizer->setMatchingMethod(0);
	optimizer->setMatchingMaxDistance(maxMatchingDist);

    // 2. Set selection method //
    optimizer->setSelectionMethod(selectionMethod);

    // 3. Set weighting method //
    optimizer->setWeightingMethod(weightingMethod);

    if(useMultiresolution)
        optimizer->enableMultiResolution(true);


	// Create the dataloader
	std::string fileName = "apartment_global.csv";
	ETHDataLoader eth_data_loader(fileName);
	
    double min_error = std::numeric_limits<double>::max();
	int index_min_error = -1;
	double min_relative_error = 1;
	int index_min_relative_error = -1;

	std::vector<float> errorsFinalIteration;

	for (int index = 0; index < eth_data_loader.getLength(); index++) {
		std::cout << "\n----Processing index: " << index << "\n";
		// Load the source and target mesh
		Sample input = eth_data_loader.getItem(index);

		Matrix4f estimatedPose = Matrix4f::Identity();
		PointCloud original_source = input.source.copy_point_cloud();
		
        // Apply initial transform to source point cloud
		input.source.change_pose(input.pose);
		// Caculate distance between source and tartget centroids after transform
		auto meanSourceTf = computeMean(input.source.getPoints());
		auto meanSource = computeMean(original_source.getPoints());
		auto meanTarget = computeMean(input.target.getPoints());
		std::cout << "Distance between mean source transform vs target: " << (meanSourceTf - meanTarget).norm() << "\n";
		std::cout << "Distance between mean source transform vs original: " << (meanSourceTf - meanSource).norm() << "\n";
		
		// Create a Convergence Measure
		auto convergenMearsure = ConvergenceMeasure(input.source.getPoints(), original_source.getPoints(), true);
		optimizer->setConvergenceMeasure(convergenMearsure);

        // Create a Time Profiler
		auto timeMeasure = TimeMeasure();
		optimizer->setTimeMeasure(timeMeasure);

		// Estimate pose
		std::cout << "num points source:" << input.source.getPoints().size() << std::endl;
		std::cout << "num points target:" << input.target.getPoints().size() << std::endl;
		

		double initial_error = convergenMearsure.benchmarkError(estimatedPose);
		auto initial_rmse = convergenMearsure.rmseAlignmentError(estimatedPose);
        std::cout << "Initial error:" << initial_error << std::endl;
        std::cout << "Initial RMSE:" << initial_rmse << std::endl;

        // Apply ICP //
        optimizer->estimatePose(input.source, input.target, estimatedPose, true);
        
        // Calculate time
		timeMeasure.calculateIterationTime();
		
		
        // Calculate error after ICP //
		//input.source.change_pose(estimatedPose);
		double final_error = convergenMearsure.getFinalErrorBenchmark();
		errorsFinalIteration.push_back(final_error);
		
        std::cout << "initial error:" << initial_error << std::endl;
		std::cout << "final error:" << final_error << std::endl;

        std::cout << "Initial RMSE:" << initial_rmse << std::endl;
        std::cout << "Final RMSE:" << convergenMearsure.rmseAlignmentError(estimatedPose) << std::endl;


		// Print out RMSE and benchmark errors of each iteration
		convergenMearsure.outputAlignmentError();

		// saving iteration errors to file //
		convergenMearsure.writeRMSEToFile(expName + "_RMSE" + std::to_string(index)+ ".txt");
		convergenMearsure.writeBenchmarkToFile(expName + "_Benchmark" + std::to_string(index) + ".txt");

		// This code can be used to save the point clouds to disk
		//original_source.writeToFile("original_source.ply");
		// input.source.writeToFile("transformed_source.ply");
		// input.target.writeToFile("target.ply");
		// input.source.change_pose(estimatedPose);
		// input.source.writeToFile("final_source.ply");

		// Compute best index to find nice examples
		if (final_error < min_error) {
			index_min_error = index;
			min_error = final_error;
		}
		if (final_error / initial_error < min_relative_error) {
			index_min_relative_error = index;
			min_relative_error = final_error / initial_error;
		}
	}

	std::cout << "The minimum error is " << min_error << " for index " << index_min_error << std::endl;
	std::cout << "The minimum relative error is " << min_relative_error << " for index " << index_min_relative_error << std::endl;

	std::ofstream newFile;
	newFile.open(expName + "_benchmark_error.txt"); // TODO Rename

	for (unsigned int i = 0; i < errorsFinalIteration.size(); i++) {
		newFile << errorsFinalIteration[i] << std::endl;
	}
	newFile.close();

	delete optimizer;

	return 0;
}

int main(int argc, char *argv[]) {
    std::string filename = "experiment.csv";
	if (argc >=2)
    	filename = argv[1]; // default experiment.

    // Read from file and run experiment
    int result = 0;
    CSVReader reader("../../Data/" + filename);
    auto configs = reader.getData();
	// Ignore first line (header)
    for (int configIdx=1; configIdx < configs.size(); configIdx++) {
		auto cf = configs[configIdx];
        std::string expName = cf[0];
        std::string expType = cf[1];
        unsigned useLinear = atoi(cf[2].c_str());
        unsigned useMetric = atoi(cf[3].c_str());
        unsigned matchingMethod = atoi(cf[4].c_str());
        unsigned selectionMethod = atoi(cf[5].c_str());
        unsigned weightingMethod = atoi(cf[6].c_str());
        unsigned useMultiresolution = atoi(cf[7].c_str());
        unsigned numIterations = atoi(cf[8].c_str()); 
        float maxMatchingDist = atof(cf[9].c_str());
        float samplingProba = atof(cf[10].c_str());
        std::cout << "\n*****Running experiment: " << expName << "\n";
        if (expType == "bunny")
            result += alignBunnyWithICP(useLinear, useMetric, matchingMethod, selectionMethod, 
                    weightingMethod, useMultiresolution, numIterations, maxMatchingDist, samplingProba, expName);
        else if (expType == "room")
            result += reconstructRoom(useLinear, useMetric, matchingMethod, selectionMethod, 
                    weightingMethod, useMultiresolution, numIterations, maxMatchingDist, samplingProba, expName);
        else if (expType == "eth")
            result += alignETH(useLinear, useMetric, matchingMethod, selectionMethod, 
                    weightingMethod, useMultiresolution, numIterations, maxMatchingDist, samplingProba, expName);
    }

    std::cout << "Run total of " << result << " experiments! Finished!";
	return result;
}
