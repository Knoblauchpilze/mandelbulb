project (mandelbulb_lib LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 14)

#set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_50,code=sm_50")

set (SOURCES
  MandelbulbRenderer.cc
  Camera.cc
  InfoPanel.cc
  Light.cc
  LightSettings.cc
  RenderMenu.cc
  RenderSettings.cc
  Fractal.cc
  RaytracingTile.cc
  Test.cu
  )

add_library (mandelbulb_lib SHARED
  ${SOURCES}
  )

set(MANDELBULB_LIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" PARENT_SCOPE)

target_link_libraries(mandelbulb_lib
  sdl_core
  sdl_graphic
  )

target_include_directories (mandelbulb_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
