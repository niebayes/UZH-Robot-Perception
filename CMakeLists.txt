cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(UZH_ROBOT_PERCEPTION)

# Configure compile mode: Debug, Release, etc.
# set(CMAKE_BUILD_TYPE Release)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable C++17 standard
# Check if the compiler supports C++17
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
   message(STATUS "Using flag -std=c++17.")
   set(CMAKE_CXX_STANDARD 17)
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

# Configure compile output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)

# Add cmake module path 
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Add required packages
# armadillo 
find_package(Armadillo REQUIRED)
if (ARMADILLO_FOUND)
    message("Found armadillo")
else()
    message("Not Found armadillo")
endif()

# OpenCV 
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message("Found OpenCV")
else()
    message("Not Found OpenCV")
endif()

find_package(Eigen3 3.3.7 REQUIRED)
if (Eigen3_FOUND)
    message("Found Eigen3")
else()
    message("Not Found Eigen3")
endif()

# ceres
find_package(Ceres REQUIRED)
if (CERES_FOUND)
    message("Found ceres")
else()
    message("Not Found ceres")
endif()

# glog
find_package(Glog REQUIRED)
if (Glog_FOUND)
    message("Found glog")
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
# find_package(PCL 1.11 QUIET COMPONENTS common io visualization)
if (PCL_FOUND)
    message("Found PCL")
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
else() 
    message("Not Found PCL")
endif()

# boost 
# cf. https://cmake.org/cmake/help/latest/module/FindBoost.html
find_package(Boost QUIET) 
if (Boost_FOUND)
    message("Found Boost")
else() 
    message("Not Found Boost")
endif()

# Add additional include path for header files
include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/utils
    ${PROJECT_SOURCE_DIR}/src
    ${ARMADILLO_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    "/usr/local/include/eigen3"
    "/usr/local/include/eigen3/unsupported"
    ${CERES_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
) # path relative to this CMakeLists.txt

# Optional googletest
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

# Add subdirectories in which put the source files
# Uncomment to enable building.

# add_subdirectory(utils)
# add_subdirectory(src/ex1)
# add_subdirectory(src/ex2)
# add_subdirectory(src/ex3)
# add_subdirectory(src/ex4)
# add_subdirectory(src/ex5)
# add_subdirectory(src/ex6)
# add_subdirectory(src/ex7)
# add_subdirectory(src/ex8)
# add_subdirectory(src/ex9)