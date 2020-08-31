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
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  return (viewer);
}

pcl::visualization::PCLVisualizer::Ptr rgbVis(
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) {
  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("Point cloud RGB Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
      cloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  return (viewer);
}

void VisualizePointCloud(const arma::mat& point_cloud) {
  // Construct a pcl point cloud object.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);

  // Populate the points
  const int kNumPoints = point_cloud.n_cols;
  for (int i = 0; i < kNumPoints; ++i) {
    pcl::PointXYZRGB point;
    point.x = point_cloud.col(i)(0);
    point.y = point_cloud.col(i)(1);
    point.z = point_cloud.col(i)(2);
    std::uint8_t r(255), g(255), b(255);
    std::uint32_t rgb =
        (static_cast<std::uint32_t>(r) | static_cast<std::uint32_t>(g) |
         static_cast<std::uint32_t>(b));
    point.rgb = *reinterpret_cast<float*>(&rgb);
    cloud->points.push_back(point);
  }
  cloud->width = cloud->size();
  cloud->height = 1;

  // Visualize
  pcl::visualization::PCLVisualizer::Ptr viewer = rgbVis(cloud);
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    // boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }
}

}  // namespace stereo

#endif  // UZH_STEREO_VISUALIZE_POINT_CLOUD_H_