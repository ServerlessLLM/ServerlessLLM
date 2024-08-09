cd build/

export SLLM_STORE_PYTHON_EXECUTABLE=$(which python3)
cmake -DCMAKE_BUILD_TYPE=Release \
  -DSLLM_STORE_PYTHON_EXECUTABLE=$SLLM_STORE_PYTHON_EXECUTABLE \
  -DBUILD_TESTS=ON \
  -G Ninja ..
cmake --build . --target all -j