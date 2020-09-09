project(ar_wireframe_cube)
add_executable(ar_wireframe_cube ar_wireframe_cube.cc)
# Change the level of optimization to -O2 or -O3 if don't debug.
target_compile_options(ar_wireframe_cube PRIVATE -O3 -march=native)
target_link_libraries(ar_wireframe_cube
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
)