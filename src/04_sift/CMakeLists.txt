project(sift)
add_executable(sift sift.cc)
# Change the level of optimization to -O2 or -O3 if don't debug.
target_compile_options(sift PRIVATE -O3 -march=native)
target_link_libraries(sift
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
  ${ARMADILLO_LIBRARIES}
)