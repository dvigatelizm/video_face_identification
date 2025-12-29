cmake_minimum_required(VERSION 3.16)
project(cluster_cleaner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(cluster_cleaner cluster_cleaner.cpp)
target_link_libraries(cluster_cleaner PRIVATE ${OpenCV_LIBS})