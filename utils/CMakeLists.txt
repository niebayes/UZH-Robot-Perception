project(utils)
add_executable(pcd2ply pcd2ply.cc)
# Change the level of optimization to -O2 or -O3 if don't debug.
target_compile_options(pcd2ply PRIVATE -O0 -march=native)
target_link_libraries(pcd2ply
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
  ${ARMADILLO_LIBRARIES}
  ${PCL_LIBRARIES}
)