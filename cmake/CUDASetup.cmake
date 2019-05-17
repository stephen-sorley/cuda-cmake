# CUDASetup.cmake
#
# Adds various CUDA flags, including those which determine what architectures
# will be supported by the produced binaries.
#
# The oldest CUDA toolkit supported by this file is CUDA 9.
#
# The lists of supported architectures must be manually updated whenever a new toolkit is released.
# When these definitions were last updated, CUDA 10 was the newest toolkit.
#
# Options:
#   CUDA_MIN_ARCH: minimum architecture to support (default: minimum version allowed by CUDA toolkit)
#   CUDA_MAX_ARCH: maximum architecture to support (default: maximum version allowed by CUDA toolkit)
#
# # # # # # # # # # # #
# The MIT License (MIT)
#
# Copyright (c) 2019 Stephen Sorley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# # # # # # # # # # # #

cmake_minimum_required(VERSION 3.14)

include_guard(DIRECTORY)

if(NOT CMAKE_CUDA_COMPILER_VERSION)
    return()
endif()

if(    CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
    # https://docs.nvidia.com/cuda/archive/10.1/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list
    set(arch_list
        30 # Kepler
        50 # Maxwell
        60 # Pascal
        70 # Volta
        75 # Turing
    )
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
    # https://docs.nvidia.com/cuda/archive/9.2/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list
    set(arch_list
        30 # Kepler
        50 # Maxwell
        60 # Pascal
        70 # Volta
    )
else()
    message(FATAL_ERROR "Current CUDA version (${CMAKE_CUDA_COMPILER_VERSION}) it too old, minimum allowed is 9.0")
endif()

# Allow user to select minimum and maximum supported architectures.
# Default to covering the entire list of arch's supported by this version of CUDA.
list(GET arch_list 0  default_min_arch)
list(GET arch_list -1 default_max_arch)

set(CUDA_MIN_ARCH ${default_min_arch} CACHE STRING "Minimum CUDA arch to support")
set(CUDA_MAX_ARCH ${default_max_arch} CACHE STRING "Maximum CUDA arch to support")
set_property(CACHE CUDA_MIN_ARCH PROPERTY STRINGS "${arch_list}")
set_property(CACHE CUDA_MAX_ARCH PROPERTY STRINGS "${arch_list}")

if(NOT CUDA_MIN_ARCH IN_LIST arch_list)
    message(FATAL_ERROR "CUDA_MIN_ARCH (${CUDA_MIN_ARCH}) is not supported by CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")
endif()
if(NOT CUDA_MAX_ARCH IN_LIST arch_list)
    message(FATAL_ERROR "CUDA_MAX_ARCH (${CUDA_MAX_ARCH}) is not supported by CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")
endif()
if(CUDA_MIN_ARCH GREATER CUDA_MAX_ARCH)
    message(FATAL_ERROR "CUDA_MIN_ARCH (${CUDA_MIN_ARCH}) cannot be greater than CUDA_MAX_ARCH (${CUDA_MAX_ARCH})")
endif()

# Construct flags to pass to nvcc to produce code for the requested range of architectures.
foreach(arch ${arch_list})
    if(arch LESS "${CUDA_MIN_ARCH}")
        continue()
    endif()
    if(arch GREATER "${CUDA_MAX_ARCH}")
        break()
    endif()

    if(arch LESS "${CUDA_MAX_ARCH}")
        # If this isn't the newest arch, just compile binary GPU code for it.
        string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${arch},code=sm_${arch}")
    else()
        # If this is the newest arch, compile binary GPU code for it, and also embed PTX code for it so
        # that the code will work on GPU arch's newer than our max supported one (uses JIT compilation).
        string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${arch},code=[compute_${arch},sm_${arch}]")
    endif()
endforeach()

# Force nvcc to treat headers from CUDA include dir as system headers. If we don't do this, we get tons of
# spam warnings from CUDA's headers when building with newer GCC or Clang.
foreach(incdir ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    string(APPEND CMAKE_CUDA_FLAGS " -isystem \"${incdir}\"")
endforeach()
