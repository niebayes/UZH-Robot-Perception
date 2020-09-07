project(ex8)
add_executable(lucas_kanade_tracker lucas_kanade_tracker.cc)
# Change the level of optimization to -O2 or -O3 if don't debug.
target_compile_options(lucas_kanade_tracker PRIVATE -O3 -march=native)
target_link_libraries(lucas_kanade_tracker
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
  ${ARMADILLO_LIBRARIES}
)