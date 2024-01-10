cmake_minimum_required (VERSION 3.7)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

add_compile_options (-Wall -Wextra -Werror -pedantic)

set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

project (mandelbulb LANGUAGES CXX CUDA)

add_executable (mandelbulb)

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
