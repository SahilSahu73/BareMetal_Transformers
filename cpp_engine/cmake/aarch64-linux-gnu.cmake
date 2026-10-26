# Cross-compile for 64-bit ARM (NEON) and run the tests under qemu user-mode emulation:
#
#   cmake -S cpp_engine -B build-arm -DCMAKE_TOOLCHAIN_FILE=cpp_engine/cmake/aarch64-linux-gnu.cmake
#   cmake --build build-arm -j && ctest --test-dir build-arm --output-on-failure
#
# Arch:    sudo pacman -S aarch64-linux-gnu-gcc qemu-user
# Debian:  sudo apt install g++-aarch64-linux-gnu qemu-user
#
# qemu validates correctness only: its timings say nothing about real ARM hardware.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Location of the target's libc / libstdc++ / libgomp (differs per distro).
if(NOT DEFINED AARCH64_SYSROOT)
  foreach(candidate /usr/aarch64-linux-gnu /usr/aarch64-linux-gnu/sysroot)
    if(EXISTS ${candidate}/lib)
      set(AARCH64_SYSROOT ${candidate})
      break()
    endif()
  endforeach()
endif()
set(CMAKE_SYSROOT ${AARCH64_SYSROOT})
set(CMAKE_FIND_ROOT_PATH ${AARCH64_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# -march=native is meaningless when cross-compiling; AArch64 guarantees NEON, nothing newer is assumed.
set(BM_NATIVE OFF CACHE BOOL "" FORCE)
set(CMAKE_CXX_FLAGS_INIT "-march=armv8-a")

# CTest runs every test binary through qemu (with the target sysroot for the dynamic linker).
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-aarch64 -L ${AARCH64_SYSROOT})
