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
  # The NEON SDOT kernel uses `vdotq_s32` which requires the `+dotprod` target
  # feature. Apple's arm64 clang sets it by default; Linux ARM clang's baseline
  # is ARMv8.0-A and rejects the intrinsic without an explicit -march. ARMv8.2-A
  # + dotprod is the minimum that lets `vdotq_s32` inline (matches the i8mm TU's
  # base and AWS Graviton2+/Apple M1+ baselines).
  target_compile_options(esm_cpp_kernels_neon PRIVATE -march=armv8.2-a+dotprod)
  # The default NEON FP32 path is the hand-written FMLA microkernel, so the
  # engine runs on Linux ARM/Graviton where Accelerate does not exist. On
  # Apple we link Accelerate so LinearNeon can opt into cblas at runtime
  # under ESM_APPLE_AMX=on. The cblas path is gated by __APPLE__ in
  # gemm_fp32.cpp (always available on Apple, unlike ESM_APPLE_AMX_AVAILABLE
  # which now also requires <Accelerate/BNNSGraph.h>).
  if(APPLE)
    target_link_libraries(esm_cpp_kernels_neon PUBLIC "-framework Accelerate")
    target_compile_definitions(esm_cpp_kernels_neon PRIVATE
      ACCELERATE_NEW_LAPACK)
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
  src/apple_ane_stub.cpp
  src/apple_whole_graph_stub.cpp
  src/artifact_cache.cpp
)
# Apple-only Objective-C++ TUs. We add them via target_sources gated on
# APPLE so the .mm extension never reaches g++ on Linux (where it would
# try cc1objplus, which isn't installed) or the linker (where there's no
# __gnu_objc_personality_v0 to resolve against). The C++ public surface
# is in include/esm_cpp/apple_*.h; on non-Apple the bridge classes are
# never instantiated.
if(APPLE)
  target_sources(esm_cpp_core PRIVATE
    src/apple_ane.mm
    src/apple_whole_graph.mm)
endif()

# Phase 14: stamp the artifact trace SHA into a generated header. The
# algorithm matches tools/_artifact_manifest.py's compute_trace_sha
# exactly: for each present input file, compute hex SHA256 of the bytes;
# join "<rel>=<hex>" parts with '|'; SHA256 the joined string.
set(_artifact_trace_inputs
  tools/esm_traceable.py
  tools/build_whole_graph_artifacts.py
  tools/build_amx_artifacts.py)
set(_artifact_trace_parts "")
foreach(_f ${_artifact_trace_inputs})
  set(_path "${CMAKE_CURRENT_SOURCE_DIR}/${_f}")
  if(EXISTS "${_path}")
    file(SHA256 "${_path}" _file_sha)
    if(_artifact_trace_parts STREQUAL "")
      set(_artifact_trace_parts "${_f}=${_file_sha}")
    else()
      set(_artifact_trace_parts "${_artifact_trace_parts}|${_f}=${_file_sha}")
    endif()
  endif()
endforeach()
string(SHA256 ESM_ARTIFACT_TRACE_SHA "${_artifact_trace_parts}")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/artifact_trace_sha.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/esm_cpp/artifact_trace_sha.h
  @ONLY)
target_include_directories(esm_cpp_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>)
target_include_directories(esm_cpp_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(esm_cpp_core PRIVATE esm_warnings nlohmann_json::nlohmann_json)
target_compile_features(esm_cpp_core PUBLIC cxx_std_20)

# Apple-AMX fp16 backend (opt-in via ESM_APPLE_AMX=on at runtime). The PUBLIC
# define + framework link mean tests and the Python extension also see them.
# Linux ARM / Graviton builds skip this block entirely — apple_amx.cpp's body
# is fully #ifdef ESM_APPLE_AMX_AVAILABLE so the .o becomes empty.
#
# We probe the SDK for <Accelerate/BNNSGraph.h> at configure time. It first
# shipped publicly with the macOS 14 SDK but the umbrella header for
# Accelerate didn't always include it; older Xcode (15.0 on some GHA
# macos-14 images) lacks the BNNSGraph types entirely. When the header
# isn't present we omit ESM_APPLE_AMX_AVAILABLE so apple_amx.cpp's body
# stays empty — the engine still builds, the AMX-fp16 path is just
# unavailable on that host. ANE/whole-graph (CoreML) is independent and
# stays enabled.
if(APPLE)
  # check_include_file_cxx synthesizes a compile that doesn't get
  # -framework Accelerate (CMake can't add framework flags to the probe
  # without manual setup), so framework-resident headers never resolve.
  # Probe the SDK directly: query the active SDK path with xcrun, then
  # check if Accelerate.framework/Headers/BNNSGraph.h exists. This is
  # the same path the compiler actually sees through framework search.
  execute_process(
    COMMAND xcrun --sdk macosx --show-sdk-path
    OUTPUT_VARIABLE _esm_sdk_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _esm_sdk_path_rc)
  set(ESM_HAVE_BNNSGRAPH_H FALSE)
  if(_esm_sdk_path_rc EQUAL 0)
    # BNNS lives inside Accelerate's vecLib sub-framework; the header is at
    # <vecLib/BNNS/bnns_graph.h> via the umbrella include. Probe the actual
    # SDK file. Older Xcode SDKs (e.g. macos-14 GHA images with Xcode 15.0)
    # may ship without it even though the API is documented as macOS 13.3+.
    set(_esm_bnnsgraph_h
        "${_esm_sdk_path}/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers/BNNS/bnns_graph.h")
    if(EXISTS "${_esm_bnnsgraph_h}")
      set(ESM_HAVE_BNNSGRAPH_H TRUE)
    endif()
  endif()
  if(ESM_HAVE_BNNSGRAPH_H)
    message(STATUS "Found Accelerate/BNNSGraph.h — AMX-fp16 path enabled")
    target_compile_definitions(esm_cpp_core PUBLIC ESM_APPLE_AMX_AVAILABLE)
  else()
    message(STATUS "Accelerate/BNNSGraph.h not found in SDK — AMX-fp16 path "
                    "disabled. (Bump Xcode / macOS SDK to enable it.)")
  endif()
  target_compile_definitions(esm_cpp_core PUBLIC
    ESM_APPLE_ANE_AVAILABLE
    ACCELERATE_NEW_LAPACK)
  target_link_libraries(esm_cpp_core PUBLIC
    "-framework Accelerate"
    "-framework Foundation"
    "-framework CoreML")
  # apple_ane.mm is Objective-C++; the source file extension drives ARC + ObjC++
  # mode in clang on Apple. The whole esm_cpp_core lib stays plain C++ on Linux
  # (the .mm becomes an unused TU that the gate fully #ifs out).
  set_source_files_properties(src/apple_ane.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc")
  set_source_files_properties(src/apple_whole_graph.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc")
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

# Phase 13: standalone whole-graph C++ bench harness (Apple-only). Built when
# ESM_BUILD_TOOLS is set or ESM_BUILD_BENCH is set, so it follows the same
# convention as the rest of the spike + bench tools.
if(APPLE AND (ESM_BUILD_BENCH OR ESM_BUILD_TOOLS))
  add_executable(bench_whole_graph_cpp tools/bench_whole_graph_cpp.cpp)
  target_link_libraries(bench_whole_graph_cpp PRIVATE esm_cpp_core esm_warnings)
endif()
