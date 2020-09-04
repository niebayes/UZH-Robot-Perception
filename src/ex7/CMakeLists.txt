project(ex7)
add_executable(localizing_with_ransac localizing_with_ransac.cc)
# Change the level of optimization to -O2 or -O3 if don't debug.
target_compile_options(localizing_with_ransac PRIVATE -O3 -march=native)
target_link_libraries(localizing_with_ransac
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
  ${ARMADILLO_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
)