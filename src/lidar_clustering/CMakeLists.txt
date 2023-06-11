cmake_minimum_required(VERSION 3.5)
project(lidar_clustering)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ./lib
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# find_package(CUDA REQUIRED)
# find_package(Boost REQUIRED COMPONENTS system)

find_package(custom_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# include_directories(
#     ./lib
#     ${CUDA_INCLUDE_DIRS}
#     ${PCL_INCLUDE_DIRS}
# )
# link_directories(lib)
# add_library(libcudacluster SHARED IMPORTED GLOBAL)
# set_target_properties(libcudacluster PROPERTIES 
#   IMPORTED_LOCATION /projects/mycar/slam/src/lidar_clustering/lib/libcudacluster.so
# )

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# add_executable(clustering_gpu src/clustering_gpu.cpp)
add_executable(clustering src/clustering.cpp)
add_executable(pc2img src/pc2img.cpp)
ament_target_dependencies(clustering sensor_msgs pcl_conversions pcl_ros custom_msgs Eigen3)
ament_target_dependencies(pc2img sensor_msgs pcl_ros)

# target_link_libraries(clustering_gpu
#   libcudacluster
# )

# ament_target_dependencies(clustering_gpu sensor_msgs pcl_conversions pcl_ros CUDA Boost)

install(TARGETS
  # clustering_gpu
  clustering
  pc2img
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
