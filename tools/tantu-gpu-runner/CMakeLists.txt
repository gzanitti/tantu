add_executable(tantu-gpu-runner main.cpp)

target_include_directories(tantu-gpu-runner PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(tantu-gpu-runner PRIVATE
  CUDA::cuda_driver
)