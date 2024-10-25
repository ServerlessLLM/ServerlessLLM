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
#include "memory_state.h"

#include <iostream>

std::ostream& operator<<(std::ostream& os, const MemoryState state) {
  return os << [state]() -> const char* {
#define PROCESS_STATE(p) \
  case (p):              \
    return #p;
    switch (state) {
      PROCESS_STATE(UNINITIALIZED);
      PROCESS_STATE(UNALLOCATED);
      PROCESS_STATE(ALLOCATED);
      PROCESS_STATE(LOADING);
      PROCESS_STATE(LOADED);
      PROCESS_STATE(CANCELLED);
      PROCESS_STATE(INTERRUPTED);
      default:
        return "UNKNOWN";
    }
#undef PROCESS_STATE
  }();
}