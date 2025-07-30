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
#include "shared_memory.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <vector>

#define CUDA_CHECK(call)                                       \
  do {                                                         \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
      LOG(FATAL) << "CUDA error: " << cudaGetErrorString(err); \
    }                                                          \
  } while (0)

// Constants
constexpr size_t kPageSize = 4096;
constexpr mode_t kFileMode = 0660;

// Global registry for cleanup
MemoryRegistry& MemoryRegistry::Instance() {
  static MemoryRegistry instance;
  return instance;
}

MemoryRegistry::MemoryRegistry() { RegisterCleanupHandlers(); }

void MemoryRegistry::Register(SharedMemoryInstance* pm) {
  std::lock_guard<std::mutex> lock(mutex_);
  regions_.push_back(pm);
}

void MemoryRegistry::Unregister(SharedMemoryInstance* pm) {
  std::lock_guard<std::mutex> lock(mutex_);
  regions_.erase(std::remove(regions_.begin(), regions_.end(), pm),
                 regions_.end());
}

void MemoryRegistry::RegisterSharedMemory(
    void* ptr, std::unique_ptr<SharedMemoryInstance> shm) {
  std::lock_guard<std::mutex> lock(mutex_);
  shared_memories_[ptr] = std::move(shm);
}

// Remove shared memory instance from registry
void MemoryRegistry::UnregisterSharedMemory(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = shared_memories_.find(ptr);
  if (it != shared_memories_.end()) {
    shared_memories_.erase(it);
  }
}

// Find shared memory instance by pointer
SharedMemoryInstance* MemoryRegistry::FindSharedMemory(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = shared_memories_.find(ptr);
  return (it != shared_memories_.end()) ? it->second.get() : nullptr;
}

void MemoryRegistry::CleanupAll() {
  if (shared_memories_.empty() && regions_.empty()) {
    return;  // Already cleaned up
  }
  // Clear the shared memories map to release shared_memory instances
  shared_memories_.clear();
  std::lock_guard<std::mutex> lock(mutex_);
  // Clear the list first to prevent double cleanup
  std::vector<SharedMemoryInstance*> to_cleanup = std::move(regions_);
  regions_.clear();

  for (auto* pm : to_cleanup) {
    // Force cleanup even if not owner in emergency
    if (std::getenv("FORCE_CLEANUP")) {
      pm->SetOwner(true);
    }
    delete pm;
  }
  // Destructors will run when unique_ptrs go out of scope
}

void MemoryRegistry::RegisterCleanupHandlers() {
  // Register exit handler
  std::atexit([]() { MemoryRegistry::Instance().CleanupAll(); });

  // Register signal handlers
  struct sigaction sa = {};
  sa.sa_handler = [](int sig) {
    LOG(WARNING) << "Caught signal " << sig << ", cleaning up...";
    MemoryRegistry::Instance().CleanupAll();

    // Re-raise the signal for default handling
    signal(sig, SIG_DFL);
    raise(sig);
  };
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESETHAND;

  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGQUIT, &sa, nullptr);
  sigaction(SIGABRT, &sa, nullptr);
}

// File metadata structure
struct FileMetadata {
  size_t size;
  size_t mapped_size;
  bool is_huge_page;
  pid_t creator_pid;
};

// Round up to page size
size_t RoundUpToPageSize(size_t size) {
  return (size + kPageSize - 1) & ~(kPageSize - 1);
}

// Get shared memory file path
std::filesystem::path GetShmPath(std::string_view name) {
  const char* tmpdir = std::getenv("TMPDIR");
  if (!tmpdir) {
    tmpdir = "/dev/shm";
  }

  // Remove leading slash if present to avoid creating subdirectories
  std::string clean_name(name);
  if (!clean_name.empty() && clean_name[0] == '/') {
    clean_name = clean_name.substr(1);
  }

  return std::filesystem::path(tmpdir) /
         (".shared_mem_" + clean_name + "_" + std::to_string(getuid()));
}

std::unique_ptr<SharedMemoryInstance> Create(std::string_view name, size_t size,
                                             void* base_addr) {
  auto pm = std::unique_ptr<SharedMemoryInstance>(new SharedMemoryInstance());

  pm->name_ = name;
  pm->size_ = size;
  pm->mapped_size_ = RoundUpToPageSize(size);
  pm->is_owner_ = true;

  std::filesystem::path path = GetShmPath(name);

  // Create shared memory file
  pm->fd_ = open(path.c_str(), O_CREAT | O_RDWR | O_EXCL, kFileMode);
  if (pm->fd_ < 0) {
    if (errno == EEXIST) {
      LOG(ERROR) << "Shared memory '" << name << "' already exists";
    } else {
      LOG(ERROR) << "Failed to create shared memory: " << strerror(errno);
    }
    return nullptr;
  }

  // Lock file for exclusive access during creation
  if (flock(pm->fd_, LOCK_EX | LOCK_NB) < 0) {
    close(pm->fd_);
    unlink(path.c_str());
    return nullptr;
  }

  // Set file size
  if (ftruncate(pm->fd_, pm->mapped_size_) < 0) {
    close(pm->fd_);
    unlink(path.c_str());
    return nullptr;
  }

  int flags = MAP_SHARED;
  if (base_addr) {
    flags |= MAP_FIXED;
  }
  pm->data_ = mmap(base_addr, pm->mapped_size_, PROT_READ | PROT_WRITE, flags,
                   pm->fd_, 0);
  if (pm->data_ == MAP_FAILED) {
    close(pm->fd_);
    unlink(path.c_str());
    return nullptr;
  }

  // Lock pages in memory
  pm->prefault_pages();
  CUDA_CHECK(
      cudaHostRegister(pm->data_, pm->mapped_size_, cudaHostRegisterDefault));

  // Write metadata to file
  FileMetadata metadata = {.size = pm->size_,
                           .mapped_size = pm->mapped_size_,
                           .is_huge_page = pm->is_huge_page_,
                           .creator_pid = getpid()};

  pwrite(pm->fd_, &metadata, sizeof(metadata), pm->mapped_size_);

  // Unlock file
  flock(pm->fd_, LOCK_UN);

  MemoryRegistry::Instance().Register(pm.get());
  return pm;
}

std::unique_ptr<SharedMemoryInstance> Open(std::string_view name) {
  auto pm = std::unique_ptr<SharedMemoryInstance>(new SharedMemoryInstance());

  pm->name_ = name;
  pm->is_owner_ = false;

  std::filesystem::path path = GetShmPath(name);

  // Open shared memory file
  pm->fd_ = open(path.c_str(), O_RDWR);
  if (pm->fd_ < 0) {
    return nullptr;
  }

  // Get file size
  struct stat st;
  if (fstat(pm->fd_, &st) < 0) {
    close(pm->fd_);
    return nullptr;
  }

  // Read metadata
  FileMetadata metadata;
  ssize_t bytes_read = pread(pm->fd_, &metadata, sizeof(metadata),
                             st.st_size - sizeof(metadata));

  if (bytes_read == sizeof(metadata)) {
    pm->size_ = metadata.size;
    pm->mapped_size_ = metadata.mapped_size;
    pm->is_huge_page_ = metadata.is_huge_page;
  } else {
    // Assume entire file is data for compatibility
    pm->size_ = st.st_size;
    pm->mapped_size_ = st.st_size;
  }

  // Map the memory
  pm->data_ = mmap(nullptr, pm->mapped_size_, PROT_READ | PROT_WRITE,
                   MAP_SHARED, pm->fd_, 0);
  if (pm->data_ == MAP_FAILED) {
    close(pm->fd_);
    return nullptr;
  }

  // Try to lock (may fail if not privileged)
  pm->prefault_pages();
  CUDA_CHECK(
      cudaHostRegister(pm->data_, pm->mapped_size_, cudaHostRegisterDefault));

  /*  LOG(INFO) << "Registered shared memory '" << name
             << "' at address " << pm->data_;

    std::cerr << "Size = " << pm->mapped_size_ << std::endl;*/

  MemoryRegistry::Instance().Register(pm.get());
  return pm;
}

SharedMemoryInstance::~SharedMemoryInstance() {
  MemoryRegistry::Instance().Unregister(this);

  // Unmap memory
  if (data_ && data_ != MAP_FAILED) {
    // Unlock pages
    CUDA_CHECK(cudaHostUnregister(data_));

    // Sync before unmapping
    if (!is_huge_page_) {
      msync(data_, mapped_size_, MS_SYNC);
    }

    // Unmap
    munmap(data_, mapped_size_);
  }

  // Close file descriptor
  if (fd_ >= 0) {
    close(fd_);
  }

  // Remove file if owner
  if (is_owner_) {
    std::filesystem::path path = GetShmPath(name_);
    unlink(path.c_str());
  }
}
