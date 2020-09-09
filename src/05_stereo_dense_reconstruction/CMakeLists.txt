project(stereo_dense_reconstruction)
add_executable(stereo_dense_reconstruction stereo_dense_reconstruction.cc)
# Change the level of optimization to -O2 or -O3 if don't debug.
target_compile_options(stereo_dense_reconstruction PRIVATE -O3 -march=native)
target_link_libraries(stereo_dense_reconstruction
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
  ${ARMADILLO_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${PCL_LIBRARIES}
)