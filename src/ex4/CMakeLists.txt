project(ex4)
add_executable(sift sift.cc)
# Change the level of optimization to -O2 or -O3 if don't debug.
target_compile_options(sift PRIVATE -O0 -march=native)
target_link_libraries(sift
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARY}
)