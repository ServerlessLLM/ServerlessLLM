cmake_minimum_required(VERSION 3.20)

# Also build these disk I/O tests without the CUDA/PyTorch extensions.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(checkpoint_writer_tests LANGUAGES CXX)
  enable_testing()
  find_package(GTest REQUIRED)
endif()

set(CHECKPOINT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../csrc/checkpoint)
add_executable(checkpointWriterTests
  test_tensor_writer.cpp
  ${CHECKPOINT_SOURCE_DIR}/aligned_buffer.cpp
  ${CHECKPOINT_SOURCE_DIR}/tensor_writer.cpp
)
target_compile_features(checkpointWriterTests PRIVATE cxx_std_17)
target_include_directories(checkpointWriterTests PRIVATE ${CHECKPOINT_SOURCE_DIR})
target_link_libraries(checkpointWriterTests PRIVATE GTest::gtest_main)
# Inject failures at the syscall boundary without changing production code.
target_link_options(checkpointWriterTests PRIVATE -Wl,--wrap=pwrite)
add_test(NAME checkpointWriterTests COMMAND checkpointWriterTests)
