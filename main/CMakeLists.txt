cmake_minimum_required (VERSION 3.5)

project(ObjectTracking)

set(sources main.cpp)

include_directories(
                    ${PROJECT_SOURCE_DIR}/../src
                    ${PROJECT_SOURCE_DIR}/../src/Detector
                    ${PROJECT_SOURCE_DIR}/../src/Tracker
                    ${PROJECT_SOURCE_DIR}/../src/Classification
)   

find_package (Threads)

set(LIBS
    ${OpenCV_LIBS}
    tracking_src
    ${CMAKE_THREAD_LIBS_INIT}
)


add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} ${LIBS})

add_executable(bgs_comparison bgs_comparison.cpp)
target_link_libraries(bgs_comparison ${OpenCV_LIBS})

add_executable(tracker_test tracker_test.cpp)
target_link_libraries(tracker_test ${OpenCV_LIBS})

add_executable(pedestrian_detection pedestrian_detection.cpp)
target_link_libraries(pedestrian_detection ${OpenCV_LIBS})