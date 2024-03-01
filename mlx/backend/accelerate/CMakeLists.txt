target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantized.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp
)

# FL: accelerate is Apple's cpu backend (math simd library)  https://developer.apple.com/documentation/accelerate