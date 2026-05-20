cmake_minimum_required(VERSION 3.10)
project(MNN_YOLO)

option(BUILD_FOR_ANDROID "Build for Android platform" OFF)
option(BUILD_DEMO "Build demo/exec or not" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# 3rd_party include directories
get_filename_component(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
set(3RD_PARTY_DIR ${PROJECT_ROOT_DIR}/3rd_party)
set(3RD_PARTY_INCLUDE_DIRS ${3RD_PARTY_DIR}/MNN/include ${3RD_PARTY_DIR}/MNN/tools/cv/include)
if(BUILD_FOR_ANDROID)
    list(APPEND 3RD_PARTY_INCLUDE_DIRS ${3RD_PARTY_DIR}/jsoncpp/include ${3RD_PARTY_DIR}/yaml-cpp/include)
endif()

# 3rd_party link libraries
if(BUILD_FOR_ANDROID)
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/android)
else()
    set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86)
endif()
set(DEP_LIBS ${LIB_DIR}/libMNN_Express.so ${LIB_DIR}/libMNNOpenCV.so ${LIB_DIR}/libMNN.so)
if(BUILD_FOR_ANDROID)
    list(APPEND DEP_LIBS ${LIB_DIR}/libjsoncpp.so ${LIB_DIR}/libyaml-cpp.so)
else()
    list(APPEND DEP_LIBS jsoncpp yaml-cpp)
endif()

# Library
add_library(detector SHARED src/detector.cc)

target_include_directories(detector
    PRIVATE ${3RD_PARTY_INCLUDE_DIRS}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)

target_link_libraries(detector PUBLIC ${DEP_LIBS})

foreach(lib ${DEP_LIBS})
    if(lib MATCHES "\\.so$")
        add_custom_command(TARGET detector POST_BUILD 
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${lib} $<TARGET_FILE_DIR:detector>
        )
    endif()
endforeach()

# Executable
if(BUILD_DEMO)
    add_executable(${PROJECT_NAME} src/main.cc)

    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    target_link_libraries(${PROJECT_NAME} PRIVATE detector)
endif()
