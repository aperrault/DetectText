cmake_minimum_required (VERSION 3.2) 
project (DetectText) 

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED core imgproc video highgui features2d)

################ Boost #################
set(Boost_USE_STATIC_LIBS        ON)
FIND_PACKAGE(Boost COMPONENTS system REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
########################################

file(GLOB DETECT_TEXT_SOURCES *.cpp *.c)
file(GLOB DETECT_TEXT_HEADERS *.hpp *.h)

add_executable(DetectText  
    ${DETECT_TEXT_SOURCES} 
    ${DETECT_TEXT_HEADERS} 
    )

target_link_libraries(DetectText
    ${Boost}
    ${OpenCV_LIBS}
)