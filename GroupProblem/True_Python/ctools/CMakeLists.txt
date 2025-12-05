cmake_minimum_required(VERSION 3.16)

project(astar_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(astar astar.cpp)

set_target_properties(astar PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}
)

target_include_directories(astar PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Оптимизация только в Release!
if(MSVC)
    target_compile_options(astar PRIVATE 
        $<$<CONFIG:Release>:/O2 /arch:AVX2 /EHsc>
    )
endif()
