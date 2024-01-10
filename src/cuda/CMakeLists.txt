
add_library (mandelbulb_cuda_lib SHARED "")

set (CMAKE_POSITION_INDEPENDENT_CODE ON)

set (CUDAFLAGS "${CUDAFLAGS};-arch=sm_50;-lineinfo")
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

target_sources (mandelbulb_cuda_lib PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/CudaExecutor.cu
	${CMAKE_CURRENT_SOURCE_DIR}/CudaWrapper.cu
	${CMAKE_CURRENT_SOURCE_DIR}/mandelbulb_kernel.cu
	)

target_link_libraries(mandelbulb_cuda_lib
	core_utils
	)

target_include_directories (mandelbulb_cuda_lib PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	)
