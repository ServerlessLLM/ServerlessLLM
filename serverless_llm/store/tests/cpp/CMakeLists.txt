# Find GTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.13.0
)
FetchContent_MakeAvailable(googletest)

# Specify the test source file
set(TEST_SOURCES test_load_to_host.cpp)

# Create an executable for the tests
add_executable(runTests ${TEST_SOURCES})

# Link the test executable with GTest and the main project
target_link_libraries(runTests GTest::gtest_main sllm_store)

# Register the test
add_test(NAME runTests COMMAND runTests)
