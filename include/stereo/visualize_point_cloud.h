#ifndef UZH_STEREO_VISUALIZE_POINT_CLOUD_H_
#define UZH_STEREO_VISUALIZE_POINT_CLOUD_H_

#include "armadillo"
#include "boost/thread.hpp"
#include "pcl/visualization/pcl_visualizer.h"

namespace stereo {

//@brief Viewer wrapper helper function.
//@ref
// https://pcl.readthedocs.io/projects/tutorials/en/latest/pcl_visualizer.html#pcl-visualizer
pcl::visualization::PCLVisualizer::Ptr simpleVis(
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) {
  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("Point cloud viewer"));
  viewer->setBackgroundColor(255, 255, 255);
  viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  return (viewer);
}

void VisualizePointCloud(const arma::mat& point_cloud) {
  // Construct a pcl point cloud object.
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Populate the points
  const int kNumPoints = point_cloud.n_cols;
  for (int i = 0; i < kNumPoints; ++i) {
    pcl::PointXYZ point;
    point.x = point_cloud.col(i)(0);
    point.y = point_cloud.col(i)(1);
    point.z = point_cloud.col(i)(2);
    cloud->points.push_back(point);
  }
  cloud->width = cloud->size();
  cloud->height = 1;

  // Visualize
  pcl::visualization::PCLVisualizer::Ptr viewer = simpleVis(cloud);
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    // boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }
}

}  // namespace stereo

#endif  // UZH_STEREO_VISUALIZE_POINT_CLOUD_H_