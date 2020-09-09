project(ex9)
add_executable(bundle_adjustment bundle_adjustment.cc)
# Change the level of optimization to -O2 or -O3 if don't debug.
target_compile_options(bundle_adjustment PRIVATE -O0 -march=native)
target_link_libraries(bundle_adjustment
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
  ${ARMADILLO_LIBRARIES}
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
)