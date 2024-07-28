// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>
#include <errno.h>
#include <glog/logging.h>
#include <string.h>

#define CUDA_CHECK(x, msg)                                               \
  {                                                                      \
    if ((x) != cudaSuccess) {                                            \
      LOG(ERROR) << msg << " " << cudaGetErrorString(cudaGetLastError()) \
                 << std::endl;                                           \
      return -1;                                                         \
    }                                                                    \
  }

#define CHECK_POSIX(x, msg)                                                   \
  {                                                                           \
    if ((x) < 0) {                                                            \
      LOG(ERROR) << msg << " errno: " << errno << "msg: " << strerror(errno); \
      return -1;                                                              \
    }                                                                         \
  }

#define WAIT_FUTURES(futures, msg)           \
  {                                          \
    for (auto& future : futures) {           \
      int ret = future.get() if (ret != 0) { \
        LOG(ERROR) << msg;                   \
        return ret;                          \
      }                                      \
    }                                        \
  }

#define CHECK_RETURN(x, msg) \
  {                          \
    if ((x) != 0) {          \
      LOG(ERROR) << msg;     \
      return -1;             \
    }                        \
  }
