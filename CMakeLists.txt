cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SFM)

# Configure compile mode: Debug, Release, etc.
set(CMAKE_BUILD_TYPE Release)
# if (NOT CMAKE_BUILD_TYPE)
#     set(CMAKE_BUILD_TYPE Release)
# endif()
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable C++17 standard
# Check if the compiler supports C++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
   message(STATUS "Using flag -std=c++17.")
   set(CMAKE_CXX_STANDARD 17)
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

# Enable optimization 
# Check if the compiler supports this intruction 
# Pass flags using modern CMake, cf. https://stackoverflow.com/a/23995391/11240780
# CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
# if(COMPILER_SUPPORTS_MARCH_NATIVE) 
#     set(CMAKE_CXX_FLAGS_RELEASE "{CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
# endif()

# Configure compile output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)

# Add required packages
# Add module path 
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
# OpenCV 
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message("Found OpenCV" " Path ${OpenCV_INCLUDE_DIRS}")
else()
    message("Not Found OpenCV")
endif()

find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message("Found Eigen3" " Path ${Eigen3_INCLUDE_DIRS}")
else()
    message("Not Found Eigen3")
endif()

# manif
find_package(manif REQUIRED)
if (manif_FOUND)
    message("Found manif" " Path ${Eigen3_INCLUDE_DIRS}")
else()
    message("Not Found Eigen3")
endif()

# ceres
find_package(Ceres REQUIRED)
if (CERES_FOUND)
    message("Found Ceres" " Path ${CERES_INCLUDE_DIRS}")
else()
    message("Not Found Ceres")
endif()

# glog
find_package(Glog REQUIRED)
if (Glog_FOUND)
    message("Found glog" " Path ${glog_INCLUDE_DIRS}")
else()
    message("Not Found glog")
endif()

# gflags
find_package(GFLAGS REQUIRED)
if (GFLAGS_FOUND)
    message("Found gflags")
else() 
    message("Not Found gflags")
endif()

# pcl
find_package(PCL REQUIRED)
if (PCL_FOUND)
    message("Found PCL")
else() 
    message("Not Found PCL")
endif()
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add additional include path for header files
include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/common
    ${PROJECT_SOURCE_DIR}/utils
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    "/usr/include"
    ${EIGEN3_INCLUDE_DIRS}
    "/usr/include/eigen3/unsupported"
    ${manif_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    # ${Sophus_INCLUDE_DIRS}
    # "/Users/bayes/slam_lib/Sophus"
    # ${Pangolin_INCLUDE_DIRS}
    # ${G2O_INCLUDE_DIRS}
) # path relative to this CMakeLists.txt


# Prepare googletest for use 
#@ref build instruction: https://github.com/google/googletest/blob/master/googletest/README.md
#@ref usage: https://github.com/google/googletest/blob/master/googletest/docs/primer.md
# Start of gtest 
# This pre-process will take place every time you modify this file and the sub-files belong to it.
# Download and unpack googletest at configure time
# configure_file(external/googletest/CMakeLists.txt.in googletest/CMakeLists.txt)
# execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
#   RESULT_VARIABLE result
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest)
# if(result)
#   message(FATAL_ERROR "CMake step for googletest failed: ${result}")
# endif()
# execute_process(COMMAND ${CMAKE_COMMAND} --build .
#   RESULT_VARIABLE result
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest)
# if(result)
#   message(FATAL_ERROR "Build step for googletest failed: ${result}")
# endif()

# # Prevent overriding the parent project's compiler/linker
# # settings on Windows
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# # Add googletest directly to our build. This defines
# # the gtest and gtest_main targets.
# add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
#                  ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
#                  EXCLUDE_FROM_ALL)

# # The gtest/gtest_main targets carry header search path
# # dependencies automatically when using CMake 2.8.11 or
# # later. Otherwise we have to add them here ourselves.
# if (CMAKE_VERSION VERSION_LESS 2.8.11)
#   include_directories("${gtest_SOURCE_DIR}/include")
# endif()

##################################################################
# Add subdirectory in which put the source files
# add_subdirectory(src/ex1)
add_subdirectory(src/ex2)