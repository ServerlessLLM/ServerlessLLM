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
#include "aligned_buffer.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <system_error>

AlignedBuffer::AlignedBuffer(const std::string& filename, size_t buffer_size)
    : filename_(filename),
      fd_(-1),
      buf_size_(buffer_size),
      buf_pos_(0),
      file_offset_(0),
      buffer_(nullptr) {
  if (buf_size_ == 0 || buf_size_ % kAlignment != 0) {
    throw std::invalid_argument(
        "Buffer size must be a positive multiple of the I/O alignment");
  }
  buffer_ = aligned_alloc(kAlignment, buf_size_);
  if (!buffer_) {
    throw std::bad_alloc();
  }
  fd_ = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
  if (fd_ < 0) {
    const int error = errno;
    free(buffer_);
    throw std::system_error(error, std::generic_category(),
                            "open " + filename_);
  }
}

AlignedBuffer::~AlignedBuffer() {
  // finish() is explicit: destructors cannot safely report a failed write.
  free(buffer_);
  if (fd_ >= 0) {
    close(fd_);
  }
}

void AlignedBuffer::flush() {
  size_t written = 0;
  while (written < buf_pos_) {
    const ssize_t ret = pwrite(fd_, static_cast<char*>(buffer_) + written,
                               buf_pos_ - written, file_offset_ + written);
    if (ret < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::system_error(errno, std::generic_category(),
                              "pwrite " + filename_);
    }
    if (ret == 0) {
      throw std::runtime_error("pwrite made no progress for " + filename_);
    }
    written += static_cast<size_t>(ret);
  }
  file_offset_ += buf_pos_;
  buf_pos_ = 0;
}

size_t AlignedBuffer::writeData(const void* data, size_t size) {
  if (fd_ < 0) {
    throw std::logic_error("Cannot write to a finished buffer");
  }
  size_t written = 0;
  while (written < size) {
    const size_t to_write = std::min(size - written, buf_size_ - buf_pos_);
    memcpy(static_cast<char*>(buffer_) + buf_pos_,
           static_cast<const char*>(data) + written, to_write);
    buf_pos_ += to_write;
    written += to_write;
    if (buf_pos_ == buf_size_) {
      flush();
    }
  }
  return written;
}

size_t AlignedBuffer::writePadding(size_t padding_size) {
  if (padding_size >= 8) {
    throw std::invalid_argument("Padding size must be less than 8 bytes");
  }
  const char zeros[8] = {};
  return writeData(zeros, padding_size);
}

size_t AlignedBuffer::finish() {
  if (finished_) {
    return file_offset_;
  }
  if (buf_pos_) {
    const size_t aligned_size =
        (buf_pos_ + kAlignment - 1) / kAlignment * kAlignment;
    memset(static_cast<char*>(buffer_) + buf_pos_, 0, aligned_size - buf_pos_);
    buf_pos_ = aligned_size;
    flush();
  }
  const int fd = fd_;
  fd_ = -1;
  if (close(fd) != 0) {
    throw std::system_error(errno, std::generic_category(),
                            "close " + filename_);
  }
  finished_ = true;
  return file_offset_;
}
