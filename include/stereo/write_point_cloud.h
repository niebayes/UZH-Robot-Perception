#ifndef UZH_STEREO_WRITE_POINT_CLOUD_H_
#define UZH_STEREO_WRITE_POINT_CLOUD_H_

#include <string>

#include "armadillo"
#include "glog/logging.h"
#include "pcl/io/pcd_io.h"

namespace stereo {

void WritePointCloud(const std::string& file_name,
                     const arma::mat& point_cloud) {
  // Construect pcl point cloud container.
  pcl::PointCloud<pcl::PointXYZ> cloud;
  const int kNumPoints = point_cloud.n_cols;
  for (int i = 0; i < kNumPoints; ++i) {
    pcl::PointXYZ point;
    point.x = point_cloud.col(i)(0);
    point.y = point_cloud.col(i)(1);
    point.z = point_cloud.col(i)(2);
    cloud.push_back(point);
  }
  cloud.width = cloud.size();
  cloud.height = 1;

  // Save to .pcd file.
  pcl::io::savePCDFileASCII(file_name, cloud);

  LOG(INFO) << "Successfully save " << kNumPoints
            << " points to the pcd file: " << file_name;
}

}  // namespace stereo

#endif  // UZH_STEREO_WRITE_POINT_CLOUD_H_