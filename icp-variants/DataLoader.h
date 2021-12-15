#pragma once
#include "PointCloud.h"

struct Sample {
	PointCloud source;
	PointCloud target;
	Matrix4f pose;
};

class DataLoader {
public:
	DataLoader() {}
	virtual int getLength() = 0;
	virtual Sample getItem(int index) = 0;
};