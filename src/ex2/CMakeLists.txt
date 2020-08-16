project(ex2)
add_executable(pnp_dlt pnp_dlt.cc)
target_link_libraries(pnp_dlt
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARY}
)