
add_library (mandelbulb_lib SHARED "")

set (CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(
	${CMAKE_CURRENT_SOURCE_DIR}/cuda
	)

target_sources (mandelbulb_lib PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/MandelbulbRenderer.cc
	${CMAKE_CURRENT_SOURCE_DIR}/Camera.cc
	${CMAKE_CURRENT_SOURCE_DIR}/InfoPanel.cc
	${CMAKE_CURRENT_SOURCE_DIR}/Light.cc
	${CMAKE_CURRENT_SOURCE_DIR}/LightSettings.cc
	${CMAKE_CURRENT_SOURCE_DIR}/RenderMenu.cc
	${CMAKE_CURRENT_SOURCE_DIR}/RenderSettings.cc
	${CMAKE_CURRENT_SOURCE_DIR}/Fractal.cc
	${CMAKE_CURRENT_SOURCE_DIR}/RaytracingTile.cc
	${CMAKE_CURRENT_SOURCE_DIR}/FractalSettings.cc
	)

target_link_libraries(mandelbulb_lib
	sdl_core
	sdl_graphic
	mandelbulb_cuda_lib
	)

target_include_directories (mandelbulb_lib PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
	)
