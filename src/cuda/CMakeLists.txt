
add_library (mandelbulb_cuda_lib SHARED "")

set (CMAKE_POSITION_INDEPENDENT_CODE ON)

# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
set_target_properties (mandelbulb_cuda_lib PROPERTIES CUDA_ARCHITECTURES "86")

target_compile_options(mandelbulb_cuda_lib PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
	)

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
