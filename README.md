Solutions for [UZH Robot Perception Course 2019](http://rpg.ifi.uzh.ch/teaching.html)
---

## Course overview
For a robot to be autonomous, it has to perceive and understand the world around it. This course introduces you to the key computer vision algorithms used in mobile robotics, such as image formation, filtering, feature extraction, multiple view geometry, dense reconstruction, tracking, image retrieval, event-based vision, visual-inertial odometry, Simultaneous Localization And Mapping (SLAM), and some basics of deep learning.
## What this repository contain
  1. Implemented basic algorithms for robot perception:
    - [Basic VR of a user specified wireframe cube](/src/01_ar_wireframe_cube).
    - [Direct Linear Transform algorithm for solving PNP problem](/src/02_pnp_dlt).
    - [Feature detection and tracking with Harris corner detector](/src/03_harris_detection_and_tracking).
    - [The principal pipeline of SIFT feature detection and description](src/04_sift).
    - [Basic stereo dense reconstruction on KITTI dataset](src/05_stereo_dense_reconstruction).
    - [Normalized eight point algorithm used in two view geometry](src/06_two_view_geometry). 
    - [Camera localization with RANSAC outlier rejection](src/07_ransac_localization).
    - [Lucas-Kanade tracker for feature tracking](src/08_lucas_kanade_tracker).
    - [Basic bundle adjustment pipeline on KITTI dataset](src/09_bundle_adjustment).
  2. Alongside their [results](./results).
## Compilation environment
  1. OS: 
    - Ubuntu (tested version: 18.04 LTS and Ubuntu 20.04 LTS).
    - macOS  (tested version: Catalina 10.15.3).
  2. Tools: 
    - CMake (tested version: 3.10 and 3.18).
    - g++   (tested version: 7.5.0 for Ubuntu).
    - clang (tested version: 11.0.0 for macOS).
    - Visual Studio Code.
  3. Dependencies:
    - Armadillo (tested version: 9.900.3)         , for most matrix computations.
    - Eigen3    (tested version: 3.3.7)           , for the rest matrix computations.
    - Ceres     (tested version: 1.14)            , for solving general non-linear least squares.
    - OpenCV    (tested version: 4.3.0 and 4.4.0) , for drawing figures.
    - PCL       (tested version: 1.8.1 and 1.11.1), for drawing plots and visualizing point clouds.
    - Glog                                        , for general logging.
    - Gflags                                      , for parsing command line arguments.
    - Gtest                                       , for testing (only used for developers).
## How to compile.
  1. `cd <path_to>/uzh_robot_perception_robot`
  2. `mkdir build && cd build` 
  3. `cmake -j4 -DCMAKE_BUILD_TYPE=Release ..` 
  4. `make`
  5. The compiled binary files will be in the `bin` directory.
