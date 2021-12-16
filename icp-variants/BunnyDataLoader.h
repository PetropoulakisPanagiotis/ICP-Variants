#pragma once
#include <stdexcept>
#include "PointCloud.h"
#include "DataLoader.h"


class BunnyDataLoader : public DataLoader {
public:
	BunnyDataLoader() {
		const std::string filenameSource = std::string("../../Data/bunny_part2_trans.off");
		const std::string filenameTarget = std::string("../../Data/bunny_part1.off");
		SimpleMesh sourceMesh;
		if (!sourceMesh.loadMesh(filenameSource)) {
			std::cout << "Mesh file wasn't read successfully at location: " << filenameSource << std::endl;
			throw std::runtime_error("Could not open file");
		}

		SimpleMesh targetMesh;
		if (!targetMesh.loadMesh(filenameTarget)) {
			std::cout << "Mesh file wasn't read successfully at location: " << filenameTarget << std::endl;
			throw std::runtime_error("Could not open file");
		}
		source_bunny = sourceMesh;
		target_bunny = targetMesh;

	}
	int getLength() {
		return 1;
	}
	Sample getItem(int index) {
		if (index != 0) {
			throw std::runtime_error("index out of range, only 1 sample available");
		}
		Sample bunny_data;
		bunny_data.source = PointCloud(source_bunny);
		bunny_data.target = PointCloud(target_bunny);
		bunny_data.pose = Matrix4f::Identity();
		return bunny_data;
	}
	SimpleMesh getSourceMesh() {
		return source_bunny;
	}
	SimpleMesh getTargetMesh() {
		return target_bunny;
	}
private:
	SimpleMesh source_bunny;
	SimpleMesh target_bunny;
};