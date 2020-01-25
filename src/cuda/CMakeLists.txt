project (mandelbulb_cuda_lib LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 14)

#set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

set (CUDAFLAGS "${CUDAFLAGS};-arch=sm_50")

set (SOURCES
  CudaExecutor.cu
  CudaJob.cc
  CudaWrapper.cu
  )

add_library (mandelbulb_cuda_lib SHARED
  ${SOURCES}
  )

set(MANDELBULB_CUDA_LIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" PARENT_SCOPE)

target_link_libraries(mandelbulb_cuda_lib
  core_utils
  )

target_include_directories (mandelbulb_cuda_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
