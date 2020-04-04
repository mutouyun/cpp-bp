cmake_minimum_required(VERSION 3.10)
project(bp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
endif()

include_directories(${CMAKE_SOURCE_DIR}/../)
file(GLOB SRC_FILES ${CMAKE_SOURCE_DIR}/../*.cpp)
file(GLOB INC_FILES ${CMAKE_SOURCE_DIR}/../*.h)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
add_executable(${PROJECT_NAME} ${SRC_FILES} ${INC_FILES})

if(NOT MSVC)
  target_link_libraries(${PROJECT_NAME} pthread)
endif()