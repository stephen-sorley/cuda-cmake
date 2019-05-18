# CUDASetup.cmake
#
# Adds various CUDA flags, including those which determine what architectures will be supported by
# the produced binaries. By default, this will produce a fat binary containing binary versions
# for each major supported architecture, plus a PTX version for the newest supported arch to allow
# the code to run on architectures that will be released in the future. The CUDA_MIN_ARCH and
# CUDA_MAX_ARCH options can be used to restrict the range of architectures, if desired.
#
# Searches for CUDA libraries (cublas, cusolver, etc.) from the CUDA toolkit, and provides imported
# targets for them if found.
#
# The oldest CUDA toolkit supported by this file is CUDA 9.
#
# WARNING:
# The lists of supported architectures must be manually updated whenever a new toolkit is released.
#
# Options:
#   CUDA_MIN_ARCH: minimum architecture to support (default: minimum version allowed by CUDA toolkit)
#   CUDA_MAX_ARCH: maximum architecture to support (default: maximum version allowed by CUDA toolkit)
#
# Current CUDA release when file was last updated: 10.1 Update 1
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

# Find additional CUDA libraries.
get_filename_component(rootdir "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" DIRECTORY) #parse off "include/" from end of path

set(CUDA_all_libs)
function(int_cudasetup_find_lib name)
    find_library(CUDA_${name}_LIBRARY ${name}
        HINTS         "${rootdir}"
        PATH_SUFFIXES lib/x64 lib64 lib
    )
    mark_as_advanced(FORCE CUDA_${name}_LIBRARY)
    if(NOT CUDA_${name}_LIBRARY)
        return()
    endif()

    if(WIN32)
        # On Windows, need to find the DLL.
        file(GLOB dllfile LIST_DIRECTORIES FALSE
            "${rootdir}/bin/${name}64_*.dll"
            "${rootdir}/bin/${name}_*.dll"
        )
        if(dllfile)
            list(GET dllfile 0 dllfile)
        endif()
        if(dllfile)
            add_library(CUDA::${name} SHARED IMPORTED GLOBAL)
            set_target_properties(CUDA::${name} PROPERTIES
                IMPORTED_LOCATION             "${dllfile}"
                IMPORTED_IMPLIB               "${CUDA_${name}_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
        endif()
    else()
        add_library(CUDA::${name} SHARED IMPORTED GLOBAL)
        set_target_properties(CUDA::${name} PROPERTIES
            IMPORTED_LOCATION             "${CUDA_${name}_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        )
    endif()

    if(TARGET CUDA::${name})
        set(varnames CUDA_all_libs ${ARGN})
        foreach(varname ${varnames})
            list(APPEND ${varname} ${name})
            set(${varname} "${${varname}}" PARENT_SCOPE)
        endforeach()
    endif()
endfunction()

set(CUDA_all_libs_static)
function(int_cudasetup_find_lib_static name)
    set(CMAKE_FIND_LIBRARY_PREFIXES "${CMAKE_STATIC_LIBRARY_PREFIX}")
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_STATIC_LIBRARY_SUFFIX}")
    find_library(CUDA_${name}_LIBRARY ${name}
        HINTS         "${rootdir}"
        PATH_SUFFIXES lib/x64 lib64 lib
    )
    mark_as_advanced(FORCE CUDA_${name}_LIBRARY)
    if(NOT CUDA_${name}_LIBRARY)
        return()
    endif()

    add_library(CUDA::${name} STATIC IMPORTED GLOBAL)
    set_target_properties(CUDA::${name} PROPERTIES
        IMPORTED_LOCATION             "${CUDA_${name}_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    )

    if(TARGET CUDA::${name})
        set(varnames CUDA_all_libs_static ${ARGN})
        foreach(varname ${varnames})
            list(APPEND ${varname} ${name})
            set(${varname} "${${varname}}" PARENT_SCOPE)
        endforeach()
    endif()
endfunction()

#(TARGETS <list of targets> DEPS <list of deps, all appended to each target's properties>)
function(int_cudasetup_add_deps)
    cmake_parse_arguments(arg "" "" "TARGETS;DEPS" ${ARGN})

    if((NOT arg_TARGETS) OR (NOT arg_DEPS))
        return()
    endif()
    list(TRANSFORM arg_TARGETS PREPEND "CUDA::")
    list(TRANSFORM arg_DEPS PREPEND "CUDA::")

    set(targets)
    foreach(target ${arg_TARGETS})
        if(TARGET ${target})
            list(APPEND targets ${target})
        endif()
    endforeach()

    set(static_deps)
    set(shared_deps)
    foreach(dep ${arg_DEPS})
        if(TARGET ${dep})
            get_target_property(type ${dep} TYPE)
            if(type STREQUAL "SHARED_LIBRARY")
                list(APPEND shared_deps ${dep})
            else()
                list(APPEND static_deps ${dep})
            endif()
        endif()
    endforeach()

    if(shared_deps)
        set_property(TARGET ${targets} APPEND PROPERTY
            IMPORTED_LINK_DEPENDENT_LIBRARIES ${shared_deps}
        )
    endif()
    if(static_deps)
        set_property(TARGET ${targets} APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES ${static_deps}
        )
    endif()
endfunction()

# Static CUDA runtime. Only need to explicitly link to this if you're calling CUDA API or library
# functions from a pure C++ library or application. If at least one file in the target is a CUDA
# file, the CUDA runtime will be pulled in automatically for you, and you don't need this target.
int_cudasetup_find_lib_static(cudart_static)
if(TARGET CUDA::cudart_static AND NOT WIN32)
    # Need to explicitly link to a few extra system libraries on Linux.
    set(extras)
    if(NOT APPLE)
        set(extras -lrt)
    endif()
    set_property(TARGET CUDA::cudart_static APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES -lpthread -ldl ${extras}
    )
endif()

# CU libraries (these are the most commonly used ones).
int_cudasetup_find_lib(cublas)
int_cudasetup_find_lib(cublasLt)
int_cudasetup_find_lib(cufft)
int_cudasetup_find_lib(cufftw)
int_cudasetup_find_lib(curand)
int_cudasetup_find_lib(cusolver)
int_cudasetup_find_lib(cusparse)

int_cudasetup_add_deps(TARGETS cublas DEPS cublasLt)
int_cudasetup_add_deps(TARGETS cufftw DEPS cufft)

# NPP (NVIDIA Performance Primitives)
set(needc)
int_cudasetup_find_lib(nppc)
int_cudasetup_find_lib(nppial  needc)
int_cudasetup_find_lib(nppicc  needc)
int_cudasetup_find_lib(nppicom needc)
int_cudasetup_find_lib(nppidei needc)
int_cudasetup_find_lib(nppif   needc)
int_cudasetup_find_lib(nppig   needc)
int_cudasetup_find_lib(nppim   needc)
int_cudasetup_find_lib(nppist  needc)
int_cudasetup_find_lib(nppisu  needc)
int_cudasetup_find_lib(nppitc  needc)
int_cudasetup_find_lib(npps    needc)

int_cudasetup_add_deps(TARGETS ${needc} DEPS nppc)


# NV libraries (additional libs for very specialized uses)
int_cudasetup_find_lib(nvblas)
int_cudasetup_add_deps(TARGETS nvblas DEPS cublas)

int_cudasetup_find_lib(nvgraph)
set(deps curand cusolver)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.1)
    list(APPEND deps cublas cusparse)
endif()
int_cudasetup_add_deps(TARGETS nvgraph DEPS ${deps})

int_cudasetup_find_lib(nvjpeg)
int_cudasetup_find_lib(nvToolsExt)


# Static versions of everything (not available on Windows as of CUDA 10.1)
set(needos) #add libs to here that need to be linked to the culibos library.

# CU static libs
set(solver_deps)
int_cudasetup_find_lib_static(cublasLt_static         needos)
int_cudasetup_find_lib_static(cublas_static           needos solver_deps)
int_cudasetup_find_lib_static(cufft_static            needos)
int_cudasetup_find_lib_static(cufft_static_nocallback needos)
int_cudasetup_find_lib_static(cufftw_static           needos)
int_cudasetup_find_lib_static(curand_static           needos)
int_cudasetup_find_lib_static(cusolver_static         needos)
int_cudasetup_find_lib_static(cusparse_static         needos solver_deps)
# -- only needed for cusolver_static
int_cudasetup_find_lib_static(lapack_static                  solver_deps)
int_cudasetup_find_lib_static(metis_static                   solver_deps)

int_cudasetup_add_deps(TARGETS cublas_static   DEPS cublasLt_static)
int_cudasetup_add_deps(TARGETS cufftw_static   DEPS cufft_static)
int_cudasetup_add_deps(TARGETS cusolver_static DEPS ${solver_deps})

# NPP static libs
set(needc)
int_cudasetup_find_lib_static(nppc_static    needos)
int_cudasetup_find_lib_static(nppial_static  needc)
int_cudasetup_find_lib_static(nppicc_static  needc)
int_cudasetup_find_lib_static(nppicom_static needc)
int_cudasetup_find_lib_static(nppidei_static needc)
int_cudasetup_find_lib_static(nppif_static   needc)
int_cudasetup_find_lib_static(nppig_static   needc)
int_cudasetup_find_lib_static(nppim_static   needc)
int_cudasetup_find_lib_static(nppist_static  needc)
int_cudasetup_find_lib_static(nppisu_static  needc)
int_cudasetup_find_lib_static(nppitc_static  needc)
int_cudasetup_find_lib_static(npps_static    needc)

int_cudasetup_add_deps(TARGETS ${needc} DEPS nppc_static)

# NV static libs
int_cudasetup_find_lib(nvgraph_static)
set(deps curand_static cusolver_static)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.1)
    list(APPEND deps cublas_static cusparse_static)
endif()
int_cudasetup_add_deps(TARGETS nvgraph_static DEPS ${deps})

int_cudasetup_find_lib(nvjpeg_static)
int_cudasetup_add_deps(TARGETS nvjpeg_static DEPS cudart_static)

# Add culibos to every static lib that needs it (MUST BE LAST)
int_cudasetup_find_lib_static(culibos) # common dependency of almost all toolkit static libs
int_cudasetup_add_deps(TARGETS ${needos} DEPS culibos)

#message(STATUS "CUDA_all_libs = ${CUDA_all_libs}")
#message(STATUS "CUDA_all_libs_static = ${CUDA_all_libs_static}")
