# Root CMake file for CUDA/CMake example project.
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
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectSetup)

project(CudaExample
    VERSION   0.1
    LANGUAGES CXX CUDA
)

include(IDESetup)
include(InstallDeps)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Set up compiler and linker options.
include(AddFlags)
include(CUDASetup)

# Require C++14, and disable compiler-specific extensions (if possible).
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          14)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Find third-party dependencies.
find_package(OpenMP QUIET COMPONENTS CXX)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Build rules.
add_executable(cudatest
    main.cpp
    test.cu
)

add_executable(test_cublas
    test_cublas.cpp
)
target_link_libraries(test_cublas PRIVATE
    CUDA::cublas
    CUDA::cudart_static
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Install rules.
install(TARGETS
    cudatest
    test_cublas

    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install_deps(lib bin
    CUDA::cublas
)

install_deps_system(lib bin)

if(WIN32)
    configure_file(LICENSE   LICENSE.txt NEWLINE_STYLE WIN32)
    configure_file(README.md README.txt  NEWLINE_STYLE WIN32)
    set(license_file "${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt")
    set(readme_file  "${CMAKE_CURRENT_BINARY_DIR}/README.txt")
else()
    set(license_file "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
    set(readme_file  "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
endif()
install(FILES "${license_file}" "${readme_file}"
    DESTINATION .
)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Packaging rules.
set(CPACK_GENERATOR ZIP) # Always generate a ZIP archive (at least).

set(CPACK_VERBATIM_VARIABLES TRUE)

# Don't include version number in default installation directory.
set(CPACK_PACKAGE_INSTALL_DIRECTORY   "${PROJECT_NAME}")

set(CPACK_PACKAGE_VENDOR              "Stephen Sorley")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CUDA/CMake Example")
set(CPACK_PACKAGE_VERSION             ${PROJECT_VERSION})
set(CPACK_RESOURCE_FILE_README        "${readme_file}")
set(CPACK_RESOURCE_FILE_LICENSE       "${license_file}")

if(WIN32)
    # If targeting windows, also use WiX to create an MSI, if available.
    if(EXISTS "$ENV{WIX}" OR EXISTS "${CPACK_WIX_ROOT}")
        set(def TRUE)
    else()
        set(def FALSE)
    endif()
    option(PKG_WIX "Use WiX Toolset to create installer?" ${def})

    if(PKG_WIX)
        list(APPEND CPACK_GENERATOR WIX)

        set(CPACK_WIX_UPGRADE_GUID          78BF12A8-7691-45D2-818E-7BABF777E49C)
        set(CPACK_WIX_PROPERTY_ARPHELPLINK  https://github.com/stephen-sorley/cuda-cmake/)
    endif()
else()
    # If targeting *nix, also create a self-extracting gzipped tar file.
    list(APPEND CPACK_GENERATOR STGZ)
endif()

include(CPack)
