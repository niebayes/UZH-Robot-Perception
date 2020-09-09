project(ransac_localization)
add_executable(ransac_localization ransac_localization.cc)
# Change the level of optimization to -O2 or -O3 if don't debug.
target_compile_options(ransac_localization PRIVATE -O3 -march=native)
target_link_libraries(ransac_localization
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
  ${ARMADILLO_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
)