project(ex3)
add_executable(harris_detection_and_tracking harris_detection_and_tracking.cc)
target_compile_options(harris_detection_and_tracking PRIVATE -O3 -march=native)
target_link_libraries(harris_detection_and_tracking
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARY}
)