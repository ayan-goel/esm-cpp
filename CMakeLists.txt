cmake_minimum_required(VERSION 3.20)
project(esm_cpp LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ESM_BUILD_TESTS "Build C++ tests" ON)
option(ESM_BUILD_PYTHON "Build pybind11 Python module" OFF)
option(ESM_BUILD_BENCH "Build Google Benchmark microbenchmarks" OFF)
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

# Per-ISA kernel OBJECT libraries (Slice 1.4). The same kernel .cpp files are
# compiled multiple times, once per ISA. Each pass defines exactly one of
# ESM_KERNEL_{REFERENCE,AVX512,NEON} so only that ISA's impl emits symbols;
# everything else is #ifdef'd out. Per CLAUDE.md: target-attribute sprinkling
# is forbidden — flag isolation lives at the TU level, here.
set(ESM_KERNEL_SOURCES
  src/kernels/gemm_fp32.cpp
  src/kernels/gemm_int8.cpp
  src/kernels/layernorm.cpp
  src/kernels/gelu.cpp
  src/kernels/residual.cpp
  src/kernels/rope.cpp
  src/kernels/attention.cpp
)

add_library(esm_cpp_kernels_ref OBJECT ${ESM_KERNEL_SOURCES})
target_include_directories(esm_cpp_kernels_ref PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(esm_cpp_kernels_ref PRIVATE ESM_KERNEL_REFERENCE)
target_link_libraries(esm_cpp_kernels_ref PRIVATE esm_warnings)
target_compile_features(esm_cpp_kernels_ref PUBLIC cxx_std_20)

set(ESM_KERNEL_OBJECT_LIBS esm_cpp_kernels_ref)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  add_library(esm_cpp_kernels_avx512 OBJECT ${ESM_KERNEL_SOURCES})
  target_include_directories(esm_cpp_kernels_avx512 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_compile_definitions(esm_cpp_kernels_avx512 PRIVATE ESM_KERNEL_AVX512)
  target_compile_options(esm_cpp_kernels_avx512 PRIVATE
    -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mavx512bf16 -mfma)
  target_link_libraries(esm_cpp_kernels_avx512 PRIVATE esm_warnings)
  target_compile_features(esm_cpp_kernels_avx512 PUBLIC cxx_std_20)
  list(APPEND ESM_KERNEL_OBJECT_LIBS esm_cpp_kernels_avx512)

  # Slice 6: AMX-INT8 TDPBUSD path. Compiled into its own TU because
  # `-mamx-tile -mamx-int8` are GCC 11+ / Clang 14+ only and we want the
  # AMX intrinsics nowhere except this file. Runtime gate inside LinearAmx
  # handles Linux < 5.16 (no XSAVE permission) by falling back to VNNI.
  add_library(esm_cpp_kernels_amx OBJECT src/kernels/gemm_amx.cpp)
  target_include_directories(esm_cpp_kernels_amx PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_compile_definitions(esm_cpp_kernels_amx PRIVATE ESM_KERNEL_AMX)
  target_compile_options(esm_cpp_kernels_amx PRIVATE
    -mamx-tile -mamx-int8 -mavx512f -mavx512bw -mavx512dq -mavx512vl
    -mavx512vnni -mfma)
  target_link_libraries(esm_cpp_kernels_amx PRIVATE esm_warnings)
  target_compile_features(esm_cpp_kernels_amx PUBLIC cxx_std_20)
  list(APPEND ESM_KERNEL_OBJECT_LIBS esm_cpp_kernels_amx)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  add_library(esm_cpp_kernels_neon OBJECT ${ESM_KERNEL_SOURCES})
  target_include_directories(esm_cpp_kernels_neon PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_compile_definitions(esm_cpp_kernels_neon PRIVATE ESM_KERNEL_NEON)
  target_link_libraries(esm_cpp_kernels_neon PRIVATE esm_warnings)
  target_compile_features(esm_cpp_kernels_neon PUBLIC cxx_std_20)
  # The default NEON FP32 path is the hand-written FMLA microkernel, so the
  # engine runs on Linux ARM/Graviton where Accelerate does not exist. On
  # Apple (where Accelerate is a system framework) we link it and compile the
  # AMX path in; it is selected only at runtime via ESM_APPLE_AMX=on, so the
  # default behaviour stays the portable hand-written kernel.
  if(APPLE)
    target_link_libraries(esm_cpp_kernels_neon PUBLIC "-framework Accelerate")
    target_compile_definitions(esm_cpp_kernels_neon PRIVATE
      ESM_APPLE_AMX_AVAILABLE ACCELERATE_NEW_LAPACK)
  endif()
  list(APPEND ESM_KERNEL_OBJECT_LIBS esm_cpp_kernels_neon)

  # i8mm (SMMLA) path in its own TU compiled with +i8mm — the ARMv8.6 SMMLA
  # intrinsic must not leak into the ARMv8.0 baseline NEON TU. Runtime-gated
  # inside dispatch (NeonI8mm host only), so the OBJECT lib is always built
  # but its instructions only execute on a FEAT_I8MM host.
  add_library(esm_cpp_kernels_i8mm OBJECT src/kernels/gemm_i8mm.cpp)
  target_include_directories(esm_cpp_kernels_i8mm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_compile_definitions(esm_cpp_kernels_i8mm PRIVATE ESM_KERNEL_I8MM)
  target_compile_options(esm_cpp_kernels_i8mm PRIVATE -march=armv8.2-a+dotprod+i8mm)
  target_link_libraries(esm_cpp_kernels_i8mm PRIVATE esm_warnings)
  target_compile_features(esm_cpp_kernels_i8mm PUBLIC cxx_std_20)
  list(APPEND ESM_KERNEL_OBJECT_LIBS esm_cpp_kernels_i8mm)
endif()

add_library(esm_cpp_core STATIC
  src/status.cpp
  src/batch.cpp
  src/tokenizer.cpp
  src/io/gguf.cpp
  src/io/safetensors.cpp
  src/kernels/cpu_features.cpp
  src/kernels/dispatch.cpp
  src/kernels/rope_tables.cpp
  src/profile.cpp
  src/quant/observer.cpp
  src/quant/pack.cpp
  src/quant/smoothquant.cpp
  src/sched/scheduler.cpp
  src/threading/thread_pool.cpp
  src/model.cpp
  src/apple_amx.cpp
)
target_include_directories(esm_cpp_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(esm_cpp_core PRIVATE esm_warnings nlohmann_json::nlohmann_json)
target_compile_features(esm_cpp_core PUBLIC cxx_std_20)

# Apple-AMX fp16 backend (opt-in via ESM_APPLE_AMX=on at runtime). The PUBLIC
# define + framework link mean tests and the Python extension also see them.
# Linux ARM / Graviton builds skip this block entirely — apple_amx.cpp's body
# is fully #ifdef ESM_APPLE_AMX_AVAILABLE so the .o becomes empty.
if(APPLE)
  target_compile_definitions(esm_cpp_core PUBLIC
    ESM_APPLE_AMX_AVAILABLE ACCELERATE_NEW_LAPACK)
  target_link_libraries(esm_cpp_core PUBLIC "-framework Accelerate")
endif()

# Link the per-ISA kernel TUs into the static lib. The Ref TU is always
# present; SIMD TUs contribute symbols only once Slice 3 fills the
# corresponding ESM_KERNEL_AVX512 / ESM_KERNEL_NEON blocks.
foreach(t ${ESM_KERNEL_OBJECT_LIBS})
  target_link_libraries(esm_cpp_core PUBLIC ${t})
endforeach()

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

if(ESM_BUILD_BENCH)
  include(FetchContent)
  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.5)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
  add_subdirectory(bench)
endif()
