cmake_minimum_required(VERSION 3.8)
project(image_conversion)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_srvs REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${cv_bridge_INCLUDE_DIRS})
include_directories(/opt/ros/jazzy/include/cv_bridge) # Manual addition if needed

# Add executable
add_executable(image_conversion_node src/image_conversion.cpp)
ament_target_dependencies(image_conversion_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  std_srvs
  OpenCV
)

# Install target
install(TARGETS image_conversion_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()

