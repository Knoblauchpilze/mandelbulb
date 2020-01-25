project (mandelbulb_lib)

set (CMAKE_CXX_STANDARD 14)

#set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(cuda)

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
  )

add_library (mandelbulb_lib SHARED
  ${SOURCES}
  )

target_link_libraries(mandelbulb_lib
  sdl_core
  sdl_graphic
  mandelbulb_cuda_lib
  )

target_include_directories (mandelbulb_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
