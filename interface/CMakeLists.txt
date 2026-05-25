cmake_minimum_required(VERSION 3.10)
project(MNN_YOLO)

option(BUILD_FOR_ANDROID "Build for Android platform" OFF)
option(BUILD_DEMO "Build demo/exec or not" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# output directory
if(BUILD_FOR_ANDROID)
    set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin/android)
else()
    set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin/x86)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR})

# 3rd_party include directories
get_filename_component(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
set(3RD_PARTY_DIR ${PROJECT_ROOT_DIR}/3rd_party)
set(MNN_INCLUDE_DIRS ${3RD_PARTY_DIR}/MNN/include ${3RD_PARTY_DIR}/MNN/tools/cv/include)

# MNN link libraries
if(BUILD_FOR_ANDROID)
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/android)
else()
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86)
endif()
set(MNN_LIBS ${LIB_DIR}/libMNN_Express.so ${LIB_DIR}/libMNNOpenCV.so ${LIB_DIR}/libMNN.so)

# Library
add_library(detector SHARED src/detector.cc)
target_include_directories(detector
    PRIVATE ${MNN_INCLUDE_DIRS}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
           $<INSTALL_INTERFACE:include>
)
target_link_libraries(detector PUBLIC ${MNN_LIBS})

add_custom_command(TARGET detector POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MNN_LIBS} $<TARGET_FILE_DIR:detector>
)

# Executable
if(BUILD_DEMO)
    add_executable(${PROJECT_NAME} src/main.cc)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${MNN_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE detector)

    if(BUILD_FOR_ANDROID)
        target_include_directories(${PROJECT_NAME} PRIVATE ${3RD_PARTY_DIR}/jsoncpp/include ${3RD_PARTY_DIR}/yaml-cpp/include)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${LIB_DIR}/libjsoncpp.so ${LIB_DIR}/libyaml-cpp.so)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD 
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIB_DIR}/libjsoncpp.so ${LIB_DIR}/libyaml-cpp.so $<TARGET_FILE_DIR:${PROJECT_NAME}>
        )
    else()
        find_package(jsoncpp REQUIRED)
        find_package(yaml-cpp REQUIRED)
        target_link_libraries(${PROJECT_NAME} PRIVATE jsoncpp yaml-cpp)
    endif()
endif()
