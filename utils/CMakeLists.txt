project(utils)

add_executable(pcd2ply pcd2ply.cc)
# Change the level of optimization to -O2 or -O3 if don't debug.
target_compile_options(pcd2ply PRIVATE -O3 -march=native)
target_link_libraries(pcd2ply
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
  ${ARMADILLO_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(imgs2video imgs2video.cc)
# Change the level of optimization to -O2 or -O3 if don't debug.
target_compile_options(imgs2video PRIVATE -O3 -march=native)
target_link_libraries(imgs2video
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
  ${GFLAGS_LIBRARIES}
)