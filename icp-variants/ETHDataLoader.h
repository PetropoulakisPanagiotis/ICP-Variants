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
		CSVReader reader("../../Data/pose_scanner_leica.csv");

		// Get the data from CSV File
		poseList = reader.getData();

		// only load the source file here
		// Load the target based on the index in getItem()
        std::cout << "Starting data loader for source point cloud" << std::endl;
		
        const std::string filenameSource = std::string("../../Data/apartment/PointCloud0.pcd");
		
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
		
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(filenameSource, *cloud_source) == -1) {
			PCL_ERROR("Couldn't read source point cloud for apartment data");
			throw std::runtime_error("Could not open file");
		}
		
        std::cout << "Loaded "
			<< cloud_source->width * cloud_source->height
			<< " data points from source point cloud"
			<< std::endl;

        // Save source to our custom pointcloud structure (aka not pcl cloud) //
		source_pc = PointCloud(cloud_source);
	}

	int getLength() {
		return 44;
	}

	Sample getItem(int index) {
		
        if (index >= 44) {
			throw std::runtime_error("index out of range, only 44 samples available");
		}
		
        // Use the same dource point cloud every time
		Sample data;
		data.source = source_pc;

		// Load the correct target point cloud
		std::cout << "Starting data loader for target point cloud" << std::endl;
		
        const std::string filenameTarget = std::string("../../Data/apartment/PointCloud" + std::to_string(index + 1) + ".pcd");
		
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

		// Look for the correct row for the pose
		for (std::vector<std::string> vec : poseList)
		{
			if (vec[0] == std::to_string(index+1)) {
				std::cout << "Found correct poseId with timestamp " << vec[1] << std::endl;
				data.pose <<
					std::stof(vec[2]), std::stof(vec[3]), std::stof(vec[4]), std::stof(vec[5]),
					std::stof(vec[6]), std::stof(vec[7]), std::stof(vec[8]), std::stof(vec[9]),
					std::stof(vec[10]), std::stof(vec[11]), std::stof(vec[12]), std::stof(vec[13]),
					std::stof(vec[14]), std::stof(vec[15]), std::stof(vec[16]), std::stof(vec[17]);
				return data;
			}
		}
		
        throw std::runtime_error(std::to_string(index + 1) + " not found as poseId in csv file");
		
	}

private:
	PointCloud source_pc;
	std::vector<std::vector<std::string>> poseList;
};
