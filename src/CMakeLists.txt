cmake_minimum_required(VERSION 3.5)

project(tracking_src)

set(main_src definition.h)

set(detector_src 
        Detector/DetectorBase.cpp
        Detector/MotionDetector.cpp
        Detector/BGS.cpp
        Detector/FaceDetector.cpp
        Detector/PeopleDetector.cpp
        Detector/CarDetector.cpp

        Detector/DetectorBase.h
        Detector/MotionDetector.h
        Detector/BGS.h
        Detector/FaceDetector.h
        Detector/PeopleDetector.h
        Detector/CarDetector.h
)

set(tracker_src
        Tracker/Kalman.cpp
        Tracker/MultiTracker.cpp
        Tracker/SingleTracker.cpp
        Tracker/HungarianAlg/HungarianAlg.cpp
     
        Tracker/Kalman.h
        Tracker/MultiTracker.h
        Tracker/SingleTracker.h
        Tracker/HungarianAlg/HungarianAlg.h        
        )

set(classif_src
        Classification/Classifier.cpp
        
        Classification/Classifier.h
)


SOURCE_GROUP("Src" FILES ${main_src})
SOURCE_GROUP("Detector" FILES ${detector_src})
SOURCE_GROUP("Tracker" FILES ${tracker_src})
SOURCE_GROUP("Classif" FILES ${classif_src})

INCLUDE_DIRECTORIES(
                    ${PROJECT_SOURCE_DIR}
                    ${PROJECT_SOURCE_DIR}/../src
)

add_library(${PROJECT_NAME} SHARED 
                ${main_src}
                ${detector_src}
                ${tracker_src}
                ${classif_src} )

set(LIBS ${OpenCV_LIBS})

target_link_libraries(${PROJECT_NAME} ${LIBS})