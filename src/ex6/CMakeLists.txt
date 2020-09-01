project(ex6)
add_executable(two_view_geometry two_view_geometry.cc)
# Change the level of optimization to -O2 or -O3 if don't debug.
target_compile_options(two_view_geometry PRIVATE -O0 -march=native)
target_link_libraries(two_view_geometry
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
  ${ARMADILLO_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${CERES_LIBRARIES}
)