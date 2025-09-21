cmake_minimum_required(VERSION 3.10)
project(paq8px LANGUAGES CXX)

# --------------------------------------
# Build options
# --------------------------------------
option(NATIVECPU "Whether to build for your CPU (vs. general public)" OFF)
option(DISABLE_ZLIB "Whether to disable zlib" OFF)
option(NDEBUG "Suppress asserts and array bound checks" ON)

# --------------------------------------
# Compiler settings
# --------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Base optimization flags
set(BASE_FLAGS
    -O3
    -floop-strip-mine
    -funroll-loops
    -ftree-vectorize
    -fgcse-sm
    -falign-loops=16
)

# Architecture-specific flags
if (NATIVECPU)
    add_compile_options(-march=native -mtune=native)
else()
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
        add_compile_options(-march=nocona -mtune=generic)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l")
        add_compile_options(-march=armv7-a -mtune=cortex-a8)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        add_compile_options(-march=armv8-a)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
        add_compile_options(-march=rv64gc -mabi=lp64d)
    else()
        message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}, using generic optimizations")
    endif()
endif()

add_compile_options(${BASE_FLAGS})


# --------------------------------------
# NDEBUG
# --------------------------------------
if (NDEBUG)
    add_compile_definitions(NDEBUG)
endif()

# --------------------------------------
# Zlib handling
# --------------------------------------
if (DISABLE_ZLIB)
    message(STATUS "Zlib disabled by user")
    add_compile_definitions(DISABLE_ZLIB)
else()
    find_package(ZLIB REQUIRED)
endif()

# --------------------------------------
# Collect source files
# --------------------------------------
file(GLOB PAQ_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/file/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/filter/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/model/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/text/*.cpp"
)

# --------------------------------------
# Create executable
# --------------------------------------
add_executable(paq8px ${PAQ_SOURCES})

# Link Zlib if available
if (NOT DISABLE_ZLIB)
    target_include_directories(paq8px PRIVATE ${ZLIB_INCLUDE_DIRS})
    target_link_libraries(paq8px PRIVATE ${ZLIB_LIBRARIES})
endif()

# --------------------------------------
# IPO / LTO
# --------------------------------------
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if (supported)
    message(STATUS "IPO / LTO enabled")
    set_property(TARGET paq8px PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()
