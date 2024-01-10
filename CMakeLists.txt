cmake_minimum_required (VERSION 3.7)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# https://gitlab.kitware.com/cmake/cmake/-/issues/23079
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

set (CMAKE_VERBOSE_MAKEFILE OFF)

set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

project (mandelbulb LANGUAGES CXX CUDA)

add_executable (mandelbulb)

# https://forums.developer.nvidia.com/t/nvcc-fatal-error-if-compiler-warnings-are-set-on-cuda-10-2/122522
target_compile_options (mandelbulb PRIVATE
	-Wall -Wextra -pedantic -Werror
	)

find_package (SDL2 REQUIRED)
find_package (SDL2_ttf REQUIRED)

add_subdirectory(
	${CMAKE_CURRENT_SOURCE_DIR}/src
	)

target_sources (mandelbulb PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
	)

target_include_directories (mandelbulb PRIVATE
	)

target_link_libraries(mandelbulb
	core_utils
	sdl_engine
	sdl_core
	sdl_graphic
	sdl_app_core
	mandelbulb_lib
	${SDL2_LIBRARIES}
	${SDL2_TTF_LIBRARIES}
	)
