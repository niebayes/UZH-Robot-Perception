Solutions for [UZH Robot Perception Course 2019](http://rpg.ifi.uzh.ch/teaching.html)
---

## Course overview
For a robot to be autonomous, it has to perceive and understand the world around it. This course introduces you to the key computer vision algorithms used in mobile robotics, such as image formation, filtering, feature extraction, multiple view geometry, dense reconstruction, tracking, image retrieval, event-based vision, visual-inertial odometry, Simultaneous Localization And Mapping (SLAM), and some basics of deep learning.

## What this repository contain
### Implemented basic algorithms used for robot perception:
  - [Basic VR of a user specified wireframe cube](/src/01_ar_wireframe_cube). 
  <p align="center">
    <img width="200" height="95" src="https://github.com/niebayes/uzh_robot_perception_codes/blob/master/results/01_ar_wireframe_cube/cube.gif">
  <p>

  - [Direct Linear Transform algorithm for solving PNP problem](/src/02_pnp_dlt). 
  <p align="center">
    <img width="173" height="112" src="https://github.com/niebayes/uzh_robot_perception_codes/blob/master/results/02_pnp_dlt/reprojected.gif">
  <p>

  - [Feature detection and tracking with Harris corner detector](/src/03_harris_detection_and_tracking). 
  <p align="center">
    <img width="372" height="113" src="https://github.com/niebayes/uzh_robot_perception_codes/blob/master/results/03_harris_detection_and_tracking/match.gif">
  <p>

  - [The principal pipeline of SIFT feature detection and description](src/04_sift).

  - [Basic stereo dense reconstruction on KITTI dataset](src/05_stereo_dense_reconstruction).
  <p align="center">
  <img width="414" height="126" src="https://github.com/niebayes/uzh_robot_perception_codes/blob/master/results/05_stereo_dense_reconstruction/disp_map.gif">
  <p>
  <p align="center">
  <img width="192" height="108" src="https://github.com/niebayes/uzh_robot_perception_codes/blob/master/results/05_stereo_dense_reconstruction/point_cloud.gif">
  <p>

  - [Normalized eight point algorithm used in two view geometry](src/06_two_view_geometry). 
  - [Camera localization with RANSAC outlier rejection](src/07_ransac_localization).
  <p align="center">
    <img width="213" height="263" src="https://github.com/niebayes/uzh_robot_perception_codes/blob/master/results/07_ransac_localization/ransac.gif">
  <p>

  - [Lucas-Kanade tracker for feature tracking](src/08_lucas_kanade_tracker).
  <p align="center">
    <img width="347" height="186" src="https://github.com/niebayes/uzh_robot_perception_codes/blob/master/results/08_lucas_kanade_tracker/gauss_newton_iter.gif">
  <p>
  <p align="center">
  <img width="414" height="126" src="https://github.com/niebayes/uzh_robot_perception_codes/blob/master/results/08_lucas_kanade_tracker/robust_klt.gif">
  <p>

  - [Basic bundle adjustment pipeline on KITTI dataset](src/09_bundle_adjustment).
### Alongside their [results](./results).

## Compilation environment
1. OS: 
  - **Ubuntu** (tested version: 18.04 LTS and 20.04 LTS).
  - **macOS**  (tested version: Catalina 10.15.3).
2. Tools: 
  - **CMake** (tested version: 3.10 and 3.18).
  - **g++**   (tested version: 7.5.0 for Ubuntu).
  - **clang** (tested version: 11.0.0 for macOS).
  - **Visual** Studio Code.
3. Dependencies:
  - **C++17**                                       , for `std::optional`, `std::tuple`, etc.
  - **Armadillo** (tested version: 9.900.3)         , for most matrix computations.
  - **Eigen3**    (tested version: 3.3.7)           , for the rest matrix computations.
  - **Ceres**     (tested version: 1.14)            , for solving general non-linear least squares.
  - **OpenCV**    (tested version: 4.3.0 and 4.4.0) , for drawing figures.
  - **PCL**       (tested version: 1.8.1 and 1.11.1), for drawing plots and visualizing point clouds.
  - **Glog**                                        , for general logging.
  - **Gflags**                                      , for parsing command line arguments.
  - **Gtest**                                       , for testing (only used for developers).

## How to compile.
1. `cd <path_to>/uzh_robot_perception_codes`
2. `mkdir build && cd build` 
3. `cmake -j4 -DCMAKE_BUILD_TYPE=Release ..` 
4. `make`
5. The compiled binary files will be in the `bin` directory.