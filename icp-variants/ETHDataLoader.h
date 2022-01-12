#pragma once
#include <stdexcept>
#include <string> 
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "PointCloud.h"
#include "DataLoader.h"
#include "CSVReader.h"


class ETHDataLoader : public DataLoader {
public:
	ETHDataLoader() {
		// CSVWriter is not our work!
		// source: https://thispointer.com/how-to-read-data-from-a-csv-file-in-c/
        // Create an object of CSVWriter
		CSVReader reader("../../Data/apartment_local.csv");

		// Get the data from CSV File
		poseList = reader.getData();
	}

	int getLength() {
		return 3000;
	}

	Sample getItem(int index) {
		
        if (index >= 3000) {
			throw std::runtime_error("index out of range, only 3000 samples available");
		}
		Sample data;
		// Look for the correct row for the pose
		std::vector<std::string> vec = poseList[index + 1];
		std::cout << "Current index:" << index << std::endl;
		std::cout << "Current id:" << vec[0] << std::endl;
		std::cout << "Current source:" << vec[1] << std::endl;
		std::cout << "Current target:" << vec[2] << std::endl;
		data.pose <<
			std::stof(vec[4]), std::stof(vec[5]), std::stof(vec[6]), std::stof(vec[7]),
			std::stof(vec[8]), std::stof(vec[9]), std::stof(vec[10]), std::stof(vec[11]),
			std::stof(vec[12]), std::stof(vec[13]), std::stof(vec[14]), std::stof(vec[15]),
			0, 0, 0, 1;

		// Load the correct source cloud
		std::cout << "Starting data loader for source point cloud" << std::endl;
		const std::string filenameSource = std::string("../../Data/apartment/" + vec[1]);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(filenameSource, *cloud_source) == -1) {
			PCL_ERROR("Couldn't read source point cloud for apartment data");
			throw std::runtime_error("Could not open file");
		}
		std::cout << "Loaded "
			<< cloud_source->width * cloud_source->height
			<< " data points from source point cloud"
			<< std::endl;

		// Parse source pcl cloud to PointCloud //
		data.source = PointCloud(cloud_source);

		// Load the correct target point cloud
		std::cout << "Starting data loader for target point cloud" << std::endl;
        const std::string filenameTarget = std::string("../../Data/apartment/" + vec[2]);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(filenameTarget, *cloud_target) == -1) {
			PCL_ERROR("Couldn't read target point cloud for apartment data");
			throw std::runtime_error("Could not open file");
		}
        std::cout << "Loaded "
			<< cloud_target->width * cloud_target->height
			<< " data points from target point cloud"
			<< std::endl;
        // Parse target pcl cloud to PointCloud //
		data.target = PointCloud(cloud_target);
		return data;
	}

private:
	std::vector<std::vector<std::string>> poseList;
};
