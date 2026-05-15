add_executable(bench_gemm bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE
  esm_cpp_core
  benchmark::benchmark
  esm_warnings)
