project(pnp_dlt)
add_executable(pnp_dlt pnp_dlt.cc)
target_compile_options(pnp_dlt PRIVATE -O3 -march=native)
target_link_libraries(pnp_dlt
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARY}
  ${ARMADILLO_LIBARIES}
)