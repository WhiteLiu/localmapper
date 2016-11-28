cmake_minimum_required(VERSION 2.8)
project(localmapper)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_BUILD_TYPE Release)


add_definitions(-std=c++11 -Wall -Werror -Wno-reorder -Wno-unused-but-set-variable -Wno-unused-variable -Wno-unused-result -Wno-deprecated-declarations -Wno-misleading-indentation)

find_package(OpenCV  REQUIRED)
include_directories(include)

find_package(Eigen3 REQUIRED) # On some distribs this requires Eigen3 module
include_directories(${EIGEN_INCLUDE_DIRS})

find_package(catkin COMPONENTS
  roscpp
  rospy
  std_msgs
  message_filters
  cv_bridge
  image_transport
  nav_msgs
  tf_conversions
)




include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
CATKIN_DEPENDS roscpp rospy std_msgs message_filters cv_bridge image_transport
DEPENDS system_lib
)

add_executable(localmapper src/localmapper.cpp src/occupancy.cpp include/occupancy.hpp README.md package.xml)
TARGET_LINK_LIBRARIES(localmapper ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(scanlaser src/scanlaser.cpp src/occupancy.cpp include/occupancy.hpp README.md package.xml)
TARGET_LINK_LIBRARIES(scanlaser ${OpenCV_LIBS} ${catkin_LIBRARIES})

#############
## Install ##
#############
install(TARGETS localmapper RUNTIME DESTINATION lib/${PROJECT_NAME})


