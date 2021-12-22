#pragma once
#include <stdexcept>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "PointCloud.h"
#include "DataLoader.h"


class ETHDataLoader : public DataLoader {
public:
	ETHDataLoader() {
		std::cout << "Starting data loader" << std::endl;
		const std::string filenameSource = std::string("../../Data/apartment/PointCloud0.pcd");
		const std::string filenameTarget = std::string("../../Data/apartment/PointCloud1.pcd");
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(filenameSource, *cloud_source) == -1) {
			PCL_ERROR("Couldn't read source point cloud for apartment data");
			throw std::runtime_error("Could not open file");
		}
		std::cout << "Loaded "
			<< cloud_source->width * cloud_source->height
			<< " data points from source point cloud"
			<< std::endl;
		source_pc = PointCloud(cloud_source);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(filenameTarget, *cloud_target) == -1) {
			PCL_ERROR("Couldn't read target point cloud for apartment data");
			throw std::runtime_error("Could not open file");
		}
		std::cout << "Loaded "
			<< cloud_target->width * cloud_target->height
			<< " data points from target point cloud"
			<< std::endl;
		target_pc = PointCloud(cloud_target);
	}

	int getLength() {
		return 1;
	}

	Sample getItem(int index) {
		if (index != 0) {
			throw std::runtime_error("index out of range, only 1 sample available");
		}
		Sample data;
		data.source = source_pc;
		data.target = target_pc;
		data.pose = Matrix4f::Identity();
		return data;
	}

private:
	PointCloud source_pc;
	PointCloud target_pc;
};