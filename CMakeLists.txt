cmake_minimum_required(VERSION 3.8.0)
project(test)

set(LITE_AI_DIR ${CMAKE_SOURCE_DIR})

find_package(OpenCV REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(CURL REQUIRED)

include_directories(${LITE_AI_DIR}/include)

# add_subdirectory(${CMAKE_SOURCE_DIR} websocketpp) 
link_directories(${LITE_AI_DIR}/lib)
link_directories(${CMAKE_SOURCE_DIR}/websocketpp)
set(CMAKE_CXX_STANDARD 11)

set(TOOLKIT_LIBS lite.ai.toolkit onnxruntime)
set(OpenCV_LIBS opencv_core opencv_imgcodecs opencv_imgproc opencv_video opencv_videoio opencv_tracking)
# add_subdirectory(include/dlib dlib_build)
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-std=c++11 -Wno-deprecated ${CMAKE_CXX_FLAGS} ")
    message(STATUS "[Linux GCC Compiler Options]+:-std=c++11 -Wno-deprecated")
endif ()

# add_executable(test test.cpp)
# add_executable(test2 test2.cpp)
add_executable(test3 test3.cpp)
add_executable(video video.cpp)
add_executable(cURL cURL.cpp)
# add_executable(testSORT testSORT.cpp)

# target_link_libraries(test ${TOOLKIT_LIBS} ${OpenCV_LIBS})
# target_link_libraries(test2 ${TOOLKIT_LIBS} ${OpenCV_LIBS})
target_link_libraries(test3 ${TOOLKIT_LIBS} ${OpenCV_LIBS})
target_link_libraries(video ${TOOLKIT_LIBS} ${OpenCV_LIBS})
target_link_libraries(cURL CURL::libcurl)