project(ex4)
add_executable(sift sift.cc)
target_compile_options(sift PRIVATE -O3 -march=native)
target_link_libraries(sift
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARY}
)