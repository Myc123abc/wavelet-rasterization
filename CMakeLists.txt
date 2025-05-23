cmake_minimum_required(VERSION 3.30)
project(wavelet_rasterization)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# external libraries
set(SDL_SHARED OFF)
set(SDL_STATIC ON)
include(FetchContent)
FetchContent_Declare(
  SDL3 
  GIT_REPOSITORY https://github.com/libsdl-org/SDL
  GIT_TAG        release-3.2.12
)
FetchContent_MakeAvailable(SDL3)

include(FetchContent)
FetchContent_Declare(
  VMA
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG        v3.2.1
)
FetchContent_MakeAvailable(VMA)

find_package(Vulkan REQUIRED)

# compile executable
add_executable(wavelet_rasterization main.cpp)

target_include_directories(wavelet_rasterization PUBLIC
  ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(wavelet_rasterization PUBLIC
  SDL3::SDL3-static
  GPUOpen::VulkanMemoryAllocator
  ${Vulkan_LIBRARIES}
)

target_compile_definitions(wavelet_rasterization PUBLIC 
  GLM_FORCE_DEPTH_ZERO_TO_ONE
  GLM_FORCE_RADIANS
)