cmake_minimum_required(VERSION 3.16)
project(baremetal_transformer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BM_NATIVE   "Tune for the build machine (-march=native)" ON)
option(BM_OPENMP   "Use OpenMP multi-threading when available" ON)
option(BM_SANITIZE "Build with AddressSanitizer + UBSan" OFF)

add_library(bm STATIC
  src/model_io.cpp
  src/kernels_naive.cpp
  src/kernels_tiled.cpp
  src/kernels_simd.cpp
  src/kernels_int8.cpp
  src/quantize.cpp
  src/ops.cpp
  src/transformer.cpp)
target_include_directories(bm PUBLIC include)

if(NOT MSVC)
  target_compile_options(bm PUBLIC -Wall -Wextra)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")  # deliberately no -ffast-math: results must stay reproducible
  if(BM_NATIVE)
    target_compile_options(bm PUBLIC -march=native)
  endif()
  if(BM_SANITIZE)
    target_compile_options(bm PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(bm PUBLIC -fsanitize=address,undefined)
  endif()
endif()

if(BM_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(bm PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

add_executable(generate src/generate.cpp)
target_link_libraries(generate PRIVATE bm)

# ---- tests (plain executables, no framework) -------------------------------------------
enable_testing()
set(CKPT_DIR ${CMAKE_SOURCE_DIR}/../checkpoints)
set(MODEL_STEM "tiny" CACHE STRING "checkpoint stem the reference test loads, e.g. tiny or small")

add_executable(test_kernels tests/test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE bm)
add_test(NAME kernels COMMAND test_kernels)

add_executable(test_reference tests/test_reference.cpp)
target_link_libraries(test_reference PRIVATE bm)
add_test(NAME reference_logits
  COMMAND test_reference ${CKPT_DIR}/${MODEL_STEM}_fp32.bin ${CKPT_DIR}/${MODEL_STEM}_int8.bin
                         ${CKPT_DIR}/${MODEL_STEM}_reference.bin)
set_tests_properties(reference_logits PROPERTIES SKIP_RETURN_CODE 77)

add_executable(benchmark_transformer src/benchmark.cpp)
target_link_libraries(benchmark_transformer PRIVATE bm)

# the same check at the real `small` shape (6 layers, d=512) using random weights:
#   python python_scratch/random_checkpoint.py && python python_scratch/export_weights.py \
#       --ckpt checkpoints/random_small.pt --name small_random --quantize-int8
add_test(NAME reference_logits_small_shape
  COMMAND test_reference ${CKPT_DIR}/small_random_fp32.bin ${CKPT_DIR}/small_random_int8.bin
                         ${CKPT_DIR}/small_random_reference.bin --smooth-only)
set_tests_properties(reference_logits_small_shape PROPERTIES SKIP_RETURN_CODE 77)

add_executable(eval_perplexity src/eval_perplexity.cpp)
target_link_libraries(eval_perplexity PRIVATE bm)
