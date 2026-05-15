add_executable(esm_cpp_tests
  test_smoke.cpp
)
target_link_libraries(esm_cpp_tests PRIVATE
  esm_cpp_core
  GTest::gtest_main
  esm_warnings)

include(GoogleTest)
gtest_discover_tests(esm_cpp_tests)
