project(ex1)
add_executable(ar_wireframe_cube ar_wireframe_cube.cc)
target_link_libraries(ar_wireframe_cube
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
)