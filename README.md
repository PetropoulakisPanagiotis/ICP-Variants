# ICP-Variants
The Iterative Closest Point (ICP) algorithm has been successfully used for registering 3D scans, especially for robotics tasks. Throughout the years many variants have emerged that either try to reduce the execution time and/or increase the registration quality. In this work, we analyse and examine the most prominent variants in synthetic and real-world scans. We conclude that there is a trade-off between speed and accuracy and only very limited variants can achieve both while we also point out some configurations that do not work well in practice.

<p align="center">
<img src="images/bunny.png" width="120px" height="120px"> 
<img src="images/freiburg1.png" width="220px" height="170px"> 
<img src="images/hauptgebaude.png" width="200px" height="170px"> 
<img src="images/projective.png" width="230px" height="190px"> 
</p>
<br /> 

| Step  | Available Methods          | 
| --------------  | ----------       | 
| 1. Selection    | All & Random       | 
| 2. Matching     | k-NN & Projective    |
| 3. Weighting    | Constant & Point distances & Normals compatibility & Colors compatibility |
| 4. Rejection    | Normals angle        |
| 5. Metric       | Point-to-Point & Point-to-Plane & Symmetric ICP |
| 6. Minimization | Linear & Non-Linear (Levenberg-Marquardt) Optimiztion |

<b>Extra variants: Multi-Resolution ICP & Color ICP (6-dim k-NN)</b>

## Requirements 
* [Eigen-3.3](https://eigen.tuxfamily.org/index.php?title=Main_Page): Linear Algebra (vectors, solvers, etc.)
* [Flann-1.8.4](https://github.com/flann-lib/flann): k-NN search
* [glog-0.3.1](https://github.com/google/glog) 
* [Ceres-2](http://ceres-solver.org/): Non-linear optimization 
* [PCL-1.3](https://pointclouds.org/): Point cloud processing 
* C++14 
* CMake


## Datasets
Apart from the bunny toy point cloud, we use [RGB-D SLAM ](https://vision.in.tum.de/data/datasets/rgbd-dataset) dataset (freiburg_xyz sequence) for debugging and [ETH  point cloud registration](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration) dataset (Apartment) for benchmark.

### ETH dataset format
Some files to note in original ETH:
- PointCloud<i>.csv: [File format](https://projects.asl.ethz.ch/datasets/doku.php?id=hardware:tiltinglaser#file_formats)

    The header has the following elements:

    - Time_in_sec: timestamps are recorded for each 2D scan produced by the Hokuyo and then uniformly distributed to all points of the 2D scan to reconstruct a timestamps per point (second)
    - x, y, z: coordinates of laser point expressed in global frame (meter)
    - Intensities: intensity returned on each beam. If -1, values weren't recorded
    - 2DscanId: start at 0 and is incremented every time a 2D scan is taken (Each PointCloud has ~344 2D Scan. Id 0 -> 343)
    - Points: start at 0 and is incremented for every point. The value reset at zero for every 2D scan. (Each 2D Scan have 1077 points. Id 1 -> 1077.)

- overlap_apartment.csv: overlap between scene 0 -> 44 (total 45 point cloud scans)
- pose_scanner_leica.csv: poseId, timestamp, 4x4 extrinsic matrix (rotation and translation)

### Preprocess ETH Dataset
To use the ground truth, we need the matching csv file in addition to the .pcd files downloaded by the point clouds registration algorithms.
- Clone [point clouds registration repo](https://github.com/iralabdisco/point_clouds_registration_benchmark)
- Cd to the cloned repository, edit "eth_setup.py" line 42: datasets -> datasets[:1] to down load only "apartment".
- run "python eth_setup.py".
- Move "apartment" folder created to Data folder inside this repo.


### Authors
* Florian Donhauser
* Cuong Ha
* Panagiotis Petropoulakis
* Suren Sritharan

:zap: <em>Equal contribution</em>

##### Αcknowledgements
Prof. Dr. Angela Dai @ 3D AI Lab <br />
Prof. Dr. Matthias Nießner @ Visual Computing Lab <br /> 
Dr. Justus Thies <br /> 
Teaching Assistants: M. Sc. Andrei Burov, M. Sc. YuchenRao <br /> 
Department of Informatics <br />
Technical University of Munich (TUM) <br />
3D AI Lab: https://www.3dunderstanding.org/ <br />
Visual Computing & Artificial Intelligence Lab: http://niessnerlab.org/
