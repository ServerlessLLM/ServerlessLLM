cmake_minimum_required(VERSION 3.20)

# Set the project name
project(storage LANGUAGES CXX)

# Set c++ 17
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS "Build tests" ON)

include(FetchContent)
FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags.git
  GIT_TAG        v2.2.2
)
FetchContent_Declare(
  glog
  GIT_REPOSITORY https://github.com/google/glog.git
  GIT_TAG        v0.6.0
)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.13.0
)
FetchContent_Declare(
  grpc
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.48.3
)
set(FETCHCONTENT_QUIET OFF)

FetchContent_GetProperties(gflags)
if(NOT gflags_POPULATED)
  FetchContent_Populate(gflags)
  add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

FetchContent_GetProperties(glog)
if(NOT glog_POPULATED)
  FetchContent_Populate(glog)
  set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
  add_subdirectory(${glog_SOURCE_DIR} ${glog_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

FetchContent_MakeAvailable(googletest grpc)

# Protobuf and gRPC libraries
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

# Proto file
get_filename_component(hw_proto "proto/storage.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/storage.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/storage.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/storage.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/storage.grpc.pb.h")

add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
      DEPENDS "${hw_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

find_package(CUDAToolkit QUIET)
find_package(HIP QUIET)

# pthread
find_package(Threads REQUIRED)

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")

# Supported HIP architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100")

if (CUDAToolkit_FOUND)
  file(GLOB SOURCES "csrc/sllm_store/*.cpp")
  add_library(sllm_store SHARED ${SOURCES})
  set_target_properties(sllm_store PROPERTIES CUDA_ARCHITECTURES "{$CUDA_SUPPORTED_ARCHS}")
  target_link_libraries(sllm_store PUBLIC CUDA::cudart Threads::Threads glog::glog)
  target_include_directories(sllm_store PUBLIC "csrc/sllm_store" ${CUDA_INCLUDE_DIRS})
  add_executable(sllm_store_server "csrc/server.cpp" ${hw_proto_srcs} ${hw_grpc_srcs})
  target_link_libraries(sllm_store_server PRIVATE sllm_store grpc++ grpc++_reflection libprotobuf gflags::gflags glog::glog)
  target_include_directories(sllm_store_server PRIVATE "csrc/sllm_store")
elseif (HIP_FOUND)
# Excute customized hipify script to convert CUDA code to HIP code
  execute_process(
    COMMAND ${CMAKE_SOURCE_DIR}/cmake/hipify_cli.py -p ${CMAKE_SOURCE_DIR}/csrc/sllm_store -o ${CMAKE_SOURCE_DIR}/csrc/hip ${ORIGINAL_SOUORCES})
  file(GLOB SOURCES "csrc/hip/*.cpp")
  add_library(sllm_store SHARED ${SOURCES})
  set_target_properties(sllm_store PROPERTIES HIP_ARCHITECTURES "${HIP_SUPPORTED_ARCHS}")
  target_link_libraries(sllm_store PUBLIC hip::host Threads::Threads glog::glog)
  target_include_directories(sllm_store PUBLIC "csrc/hip" ${HIP_INCLUDE_DIRS})
  add_executable(sllm_store_server "csrc/server.cpp" ${hw_proto_srcs} ${hw_grpc_srcs})
  target_compile_definitions(sllm_store_server PRIVATE USE_HIP)
  target_link_libraries(sllm_store_server PRIVATE sllm_store grpc++ grpc++_reflection libprotobuf gflags::gflags glog::glog)
  target_include_directories(sllm_store_server PRIVATE "csrc/hip")
else()
  message(FATAL_ERROR "Neither CUDA nor HIP found")
endif()

# Adapted from https://github.com/vllm-project/vllm/blob/a1242324c99ff8b1e29981006dfb504da198c7c3/CMakeLists.txt
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

#
# Try to find python package with an executable that exactly matches
# `SLLM_STORE_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (SLLM_STORE_PYTHON_EXECUTABLE)
  find_python_from_executable(${SLLM_STORE_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set SLLM_STORE_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

#
# Normally `torch.utils.cpp_extension.CUDAExtension` would add
# `libtorch_python.so` for linking against an extension. Torch's cmake
# configuration does not include this library (presumably since the cmake
# config is used for standalone C++ binaries that link against torch).
# The `libtorch_python.so` library defines some of the glue code between
# torch/python via pybind and is required by SLLM_STORE extensions for this
# reason. So, add it by manually with `find_library` using torch's
# installed library path.
#
find_library(torch_python_LIBRARY torch_python PATHS
  "${TORCH_INSTALL_PREFIX}/lib")

if (NOT HIP_FOUND AND CUDAToolkit_FOUND)
  set(SLLM_STORE_GPU_LANG "CUDA")
  enable_language(CUDA)
elseif (HIP_FOUND)
  set(SLLM_STORE_GPU_LANG "HIP")
  enable_language(HIP)
else()
  message(FATAL_ERROR "Neither CUDA nor HIP found")
endif()

#
# Override the GPU architectures detected by cmake/torch and filter them by
# the supported versions for the current language.
# The final set of arches is stored in `SLLM_STORE_GPU_ARCHES`.
#
override_gpu_arches(SLLM_STORE_GPU_ARCHES
${SLLM_STORE_GPU_LANG}
"${${SLLM_STORE_GPU_LANG}_SUPPORTED_ARCHS}")

#
# Query torch for additional GPU compilation flags for the given
# `SLLM_STORE_GPU_LANG`.
# The final set of arches is stored in `SLLM_STORE_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(SLLM_STORE_GPU_FLAGS ${SLLM_STORE_GPU_LANG})

#
# Define extension targets
#

#
# _C extension
#

set(SLLM_STORE_EXT_SRC
  "csrc/checkpoint/aligned_buffer.cpp"
  "csrc/checkpoint/checkpoint.cu"
  "csrc/checkpoint/checkpoint_py.cpp"
  "csrc/checkpoint/tensor_writer.cpp"
)

define_gpu_extension_target(
  _C
  DESTINATION sllm_store
  LANGUAGE ${SLLM_STORE_GPU_LANG}
  SOURCES ${SLLM_STORE_EXT_SRC}
  COMPILE_FLAGS ${SLLM_STORE_GPU_FLAGS}
  ARCHITECTURES ${SLLM_STORE_GPU_ARCHES}
  WITH_SOABI)
