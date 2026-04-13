cmake_minimum_required(VERSION 3.15)

# MSVC: use /MD and /MDd everywhere (matches gtest_force_shared_crt and avoids LIBCMT vs MSVCRT mix).
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# Link cudart dynamically (default for targets) — avoids LIBCMT from cudart_static with /MD (LNK4098).
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# set the project name
project(cbns3d_cuda LANGUAGES CXX CUDA)
# Enable CUDA
enable_language(CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# specify the C++ standard (CUDA 13+ Thrust/CCCL expects at least C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Check if CMAKE_CUDA_ARCHITECTURES is set
if (NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is not defined. Please set it using -D CMAKE_CUDA_ARCHITECTURES=<value> when configuring the project.")
endif()

# Show the selected CUDA architecture
message(STATUS "CUDA architectures set to: ${CMAKE_CUDA_ARCHITECTURES}")

# Toolkit includes (Thrust/CCCL) — required for .cpp files that include Block3d_cuda.h, not only nvcc.
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# CUDA 13+ CCCL (Thrust): MSVC conforming preprocessor for nvcc.
# CUDA 13+ nvcc: `-cudart shared` is deprecated; add `-cudart hybrid` (must follow CMake's shared flag on the command line).
# CMAKE_CUDA_FLAGS is usually cached: use CACHE FORCE + MATCHES so reconfigures do not drop flags or duplicate them.
if(MSVC)
  if(NOT CMAKE_CUDA_FLAGS MATCHES "/Zc:preprocessor")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor" CACHE STRING "Flags used by the CUDA compiler" FORCE)
  endif()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
  if(NOT CMAKE_CUDA_FLAGS MATCHES "-cudart hybrid")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -cudart hybrid" CACHE STRING "Flags used by the CUDA compiler" FORCE)
  endif()
endif()

include_directories("${CUDAToolkit_INCLUDE_DIRS}")

# Bring the headers into the project
include_directories(include)

# Find all *.cu files in the src/ directory (stub is only for final exe device link, not the lib)
file(GLOB CU_FILES "src/*.cu")
list(REMOVE_ITEM CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_device_link_stub.cu")

add_library(block3d_cuda STATIC ${CU_FILES})
set_target_properties(block3d_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(block3d_cuda PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CUDAToolkit_INCLUDE_DIRS}"
)

option(INVIS_MODE "Inviscid simulation" ON)
if (INVIS_MODE)
  target_compile_definitions(block3d_cuda
                             PRIVATE IS_INVISCID
			     )
endif()

set(SOURCES src/cbns3d.cpp src/Block3d.cpp src/cuda_device_link_stub.cu)

# Add the executable
add_executable(cbns3d_cuda ${SOURCES})
set_target_properties(cbns3d_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                              CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(cbns3d_cuda block3d_cuda)
target_include_directories(cbns3d_cuda PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")

if (INVIS_MODE)
  target_compile_definitions(cbns3d_cuda
                             PRIVATE IS_INVISCID
			     )
endif()

# Create a toggle option for building tests
option(BUILD_TESTS "Build unit tests" OFF)

# Add tests if the option is enabled
if(BUILD_TESTS)
  message(STATUS "Building unit tests")

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)  

  enable_testing()
  
  add_executable(unit_tests
                 test/gradient_test.cpp
                 test/metrics_test.cpp
                 src/cuda_device_link_stub.cu
  	         )

  set_target_properties(unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                              CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  target_link_libraries(unit_tests gtest_main block3d_cuda)
  target_include_directories(unit_tests PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")

  include(GoogleTest)
  # POST_BUILD runs unit_tests.exe on every link; that can fail (no GPU, CRT issues) and breaks the build.
  gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)
else()
  message(STATUS "Unit tests are disabled")
endif()
