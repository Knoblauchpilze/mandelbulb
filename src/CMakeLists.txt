project (mandelbulb_lib)

set (CMAKE_CXX_STANDARD 14)

#set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

set (SOURCES
  Content.cc
  )

add_library (mandelbulb_lib SHARED
  ${SOURCES}
  )

set(MANDELBULB_LIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" PARENT_SCOPE)

target_link_libraries(mandelbulb_lib
  sdl_core
  )

target_include_directories (mandelbulb_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
