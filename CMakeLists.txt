cmake_minimum_required(VERSION 3.20)
project(esm_cpp LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ESM_BUILD_TESTS "Build C++ tests" ON)
option(ESM_BUILD_PYTHON "Build pybind11 Python module" OFF)
option(ESM_SANITIZERS "Enable ASan+UBSan on Debug" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_library(esm_warnings INTERFACE)
target_compile_options(esm_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
  -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused
  -Woverloaded-virtual -Wdouble-promotion -Wformat=2)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -fno-math-errno)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_compile_options(-march=x86-64-v3)
  endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND ESM_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

include(FetchContent)
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

add_library(esm_cpp_core STATIC
  src/status.cpp
  src/tokenizer.cpp
  src/io/safetensors.cpp
  src/kernels/cpu_features.cpp
  src/kernels/gemm_fp32.cpp
  src/kernels/layernorm.cpp
  src/kernels/gelu.cpp
  src/kernels/rope.cpp
  src/kernels/attention.cpp
  src/model.cpp
)
target_include_directories(esm_cpp_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(esm_cpp_core PRIVATE esm_warnings nlohmann_json::nlohmann_json)
target_compile_features(esm_cpp_core PUBLIC cxx_std_20)

if(ESM_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_subdirectory(tests/cpp)
endif()

if(ESM_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core python/esm_cpp/_core.cpp)
  target_link_libraries(_core PRIVATE esm_cpp_core esm_warnings)
  install(TARGETS _core LIBRARY DESTINATION esm_cpp)
endif()
