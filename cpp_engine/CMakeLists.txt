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
option(BM_REFERENCE_SMOOTH_ONLY "Reference tests: skip the quantised-activation/KV modes (for random-weight models, as in CI)" OFF)

# The multiplier-less ternary accumulate kernel is an object library of its own so the build can
# disassemble exactly that code and assert it contains no multiply instruction (see tests below).
add_library(bm_ternary_accum OBJECT src/kernels_ternary_accum.cpp)
target_include_directories(bm_ternary_accum PUBLIC include)
set_target_properties(bm_ternary_accum PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(bm STATIC
  $<TARGET_OBJECTS:bm_ternary_accum>
  src/ternary.cpp
  src/lowbit_train.cpp
  src/model_io.cpp
  src/kernels_naive.cpp
  src/kernels_tiled.cpp
  src/kernels_simd.cpp
  src/kernels_int8.cpp
  src/quantize.cpp
  src/ops.cpp
  src/transformer.cpp)
target_include_directories(bm PUBLIC include)
set_target_properties(bm PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
  target_compile_options(bm PUBLIC -Wall -Wextra)
  target_compile_options(bm_ternary_accum PRIVATE -Wall -Wextra)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")  # deliberately no -ffast-math: results must stay reproducible
  if(BM_NATIVE)
    target_compile_options(bm PUBLIC -march=native)
    target_compile_options(bm_ternary_accum PRIVATE -march=native)
  endif()
  if(BM_SANITIZE)
    target_compile_options(bm PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_compile_options(bm_ternary_accum PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(bm PUBLIC -fsanitize=address,undefined)
  endif()
endif()

if(BM_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(bm PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(bm_ternary_accum PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

add_executable(generate src/generate.cpp)
target_link_libraries(generate PRIVATE bm)

# ---- tests (plain executables, no framework) -------------------------------------------
enable_testing()
set(CKPT_DIR ${CMAKE_SOURCE_DIR}/../checkpoints CACHE PATH "directory holding the exported model files")
set(MODEL_STEM "tiny" CACHE STRING "checkpoint stem the reference test loads, e.g. tiny or small")

add_executable(test_kernels tests/test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE bm)
add_test(NAME kernels COMMAND test_kernels)

add_executable(test_reference tests/test_reference.cpp)
target_link_libraries(test_reference PRIVATE bm)
if(BM_REFERENCE_SMOOTH_ONLY)
  set(REF_EXTRA_ARGS --smooth-only)
endif()
add_test(NAME reference_logits
  COMMAND test_reference ${CKPT_DIR}/${MODEL_STEM}_fp32.bin ${CKPT_DIR}/${MODEL_STEM}_int8.bin
                         ${CKPT_DIR}/${MODEL_STEM}_reference.bin ${REF_EXTRA_ARGS})
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

# ---- low-precision extension: ternary weights, multiplier-less GEMM, SR / BFP gradients ---------
add_executable(test_ternary tests/test_ternary.cpp)
target_link_libraries(test_ternary PRIVATE bm)
add_test(NAME ternary COMMAND test_ternary)

# The headline claim of the ternary kernel - no multiplier on the weight path - is checked on the
# machine code itself. The negative control proves the checker can fail: the same script must
# reject the FP32 SIMD kernels.
find_program(BM_OBJDUMP NAMES ${CMAKE_OBJDUMP} objdump)
if(BM_OBJDUMP)
  add_test(NAME ternary_has_no_multiplier
    COMMAND ${CMAKE_SOURCE_DIR}/../scripts/check_multiplierless.sh $<TARGET_OBJECTS:bm_ternary_accum> ${BM_OBJDUMP})
  add_test(NAME multiplier_checker_negative_control
    COMMAND ${CMAKE_SOURCE_DIR}/../scripts/check_multiplierless.sh ${CMAKE_BINARY_DIR}/CMakeFiles/bm.dir/src/kernels_simd.cpp.o ${BM_OBJDUMP})
  set_tests_properties(multiplier_checker_negative_control PROPERTIES WILL_FAIL TRUE)
endif()

# Shared library with a plain C interface, loaded from Python with ctypes
# (python_scratch/lowbit_native.py): the same kernels the C++ tests exercise.
add_library(bmlowbit SHARED src/lowbit_capi.cpp)
target_link_libraries(bmlowbit PRIVATE bm)

add_executable(benchmark_lowbit src/benchmark_lowbit.cpp)
target_link_libraries(benchmark_lowbit PRIVATE bm)
