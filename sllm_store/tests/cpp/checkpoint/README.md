# Checkpoint writer tests

These tests build the production checkpoint writer without PyTorch or a GPU.
On Linux, install a C++17 compiler, CMake 3.20 or later, and GTest development
files (`libgtest-dev` on Ubuntu), then run from the repository root:

```sh
cmake -S sllm_store/tests/cpp/checkpoint -B build/checkpoint
cmake --build build/checkpoint --parallel 2
ctest --test-dir build/checkpoint --output-on-failure
```

The tests create temporary files under `/tmp`, which must support `O_DIRECT`.
They check tensor contents, zero-filled record and partition padding, offsets
across partitions, direct reads, and I/O error propagation. Small configurable
buffers and partition limits exercise boundaries without writing large models.
The linker wraps `pwrite` to inject interruptions, short writes, no-progress
writes, and disk-full errors; production code does not contain test hooks.

`TensorWriter::finish()` must succeed before tensor offsets are published.
Partition files end on a 4096-byte boundary, and offsets for later partitions
include that physical padding, matching the loader's concatenation of files.
Destructors release resources without performing deferred writes.

The Python binding tests in `sllm_store/tests/python/test_save_dict.py` additionally
check shared-storage offsets and ensure an open failure does not publish an index.
Those tests require a built `sllm_store` extension.
