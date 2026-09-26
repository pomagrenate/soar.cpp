cmake_minimum_required(VERSION 3.20)
project(soar_inference VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /W4 /permissive- /utf-8)
else()
    add_compile_options(-O3 -Wall -Wextra -Wno-attributes -Wno-unused-parameter)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann
    ${CMAKE_CURRENT_SOURCE_DIR}/palloc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/palloc/src
)

# Automatically initialize git submodules if palloc is missing
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/palloc/src/static.c")
    message(STATUS "palloc submodule not found. Initializing git submodules...")
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT
        )
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(WARNING "Failed to automatically fetch git submodules. Please run 'git submodule update --init --recursive'")
        endif()
    endif()
endif()

# Build palloc library
set(PALLOC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/palloc/src/static.c
    ${CMAKE_CURRENT_SOURCE_DIR}/palloc/src/arena_pomai.c
)
add_library(palloc_static STATIC ${PALLOC_SOURCES})
target_include_directories(palloc_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/palloc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/palloc/src
)
set_target_properties(palloc_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(palloc_static PRIVATE -Wno-attributes -Wno-unused-variable -Wno-type-limits)
endif()

find_package(OpenMP)

# Core SOAR library sources
set(SOAR_SOURCES
    src/core/logging.cpp
    src/core/shape.cpp
    src/core/config_parser.cpp
    src/memory/palloc_bridge.cpp
    src/memory/palloc_override.cpp
    src/memory/host_arena.cpp
    src/memory/device_slab.cpp
    src/vulkan/dynamic_loader.cpp
    src/vulkan/context.cpp
    src/vulkan/buffer.cpp
    src/vulkan/pipeline.cpp
    src/vulkan/command_queue.cpp
    src/tensor/tensor.cpp
    src/nn/layers.cpp
    src/nn/blocks.cpp
    src/nn/soar_model.cpp
    src/losses/losses.cpp
    src/optim/adamw.cpp
    src/optim/sgd.cpp
    src/cuda/cuda_runtime.cpp
    src/cuda/cuda_stream.cpp
    src/cuda/cuda_event.cpp
    src/cuda/cuda_allocator.cpp
    src/autograd/engine.cpp
    src/data/image_io.cpp
    src/data/coco_dataset.cpp
    src/data/yolo_dataset.cpp
    src/data/preprocess.cpp
    src/utils/metrics.cpp
    src/engine/trainer.cpp
    src/engine/validator.cpp
    src/engine/predictor.cpp
    src/api/c_api.cpp
)

add_library(soar_core STATIC ${SOAR_SOURCES})
target_link_libraries(soar_core PUBLIC palloc_static)
if(OpenMP_CXX_FOUND)
    target_link_libraries(soar_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(UNIX AND NOT APPLE)
    target_link_libraries(soar_core PUBLIC dl)
endif()
target_include_directories(soar_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan/include
)

# Shared dynamic library for C-ABI / Python bindings
add_library(soar_engine_shared SHARED ${SOAR_SOURCES})
target_link_libraries(soar_engine_shared PUBLIC palloc_static)
if(OpenMP_CXX_FOUND)
    target_link_libraries(soar_engine_shared PUBLIC OpenMP::OpenMP_CXX)
endif()
if(UNIX AND NOT APPLE)
    target_link_libraries(soar_engine_shared PUBLIC dl)
endif()
target_include_directories(soar_engine_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan/include
)
set_target_properties(soar_engine_shared PROPERTIES OUTPUT_NAME "soar_engine")

# Standalone CLI executable
add_executable(soar_engine_cli src/main.cpp)
target_link_libraries(soar_engine_cli PRIVATE soar_core)
set_target_properties(soar_engine_cli PROPERTIES OUTPUT_NAME "soar_engine")

# Test executables
enable_testing()

add_executable(test_palloc_memory tests/test_palloc_memory.cpp)
target_link_libraries(test_palloc_memory PRIVATE soar_core)
add_test(NAME test_palloc_memory COMMAND test_palloc_memory)

add_executable(test_vulkan_backend tests/test_vulkan_backend.cpp)
target_link_libraries(test_vulkan_backend PRIVATE soar_core)
add_test(NAME test_vulkan_backend COMMAND test_vulkan_backend)
set_tests_properties(test_vulkan_backend PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_operators tests/test_operators.cpp)
target_link_libraries(test_operators PRIVATE soar_core)
add_test(NAME test_operators COMMAND test_operators)
set_tests_properties(test_operators PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_full_engine tests/test_full_engine.cpp)
target_link_libraries(test_full_engine PRIVATE soar_core)
add_test(NAME test_full_engine COMMAND test_full_engine)

add_executable(test_data_pipeline tests/test_data_pipeline.cpp)
target_link_libraries(test_data_pipeline PRIVATE soar_core)
add_test(NAME test_data_pipeline COMMAND test_data_pipeline)

add_executable(test_losses tests/test_losses.cpp)
target_link_libraries(test_losses PRIVATE soar_core)
add_test(NAME test_losses COMMAND test_losses)

add_executable(test_metrics_preprocess tests/test_metrics_preprocess.cpp)
target_link_libraries(test_metrics_preprocess PRIVATE soar_core)
add_test(NAME test_metrics_preprocess COMMAND test_metrics_preprocess)

add_executable(test_cuda_subsystem tests/test_cuda_subsystem.cpp)
target_link_libraries(test_cuda_subsystem PRIVATE soar_core)
add_test(NAME test_cuda_subsystem COMMAND test_cuda_subsystem)

add_executable(test_autograd_engine tests/test_autograd_engine.cpp)
target_link_libraries(test_autograd_engine PRIVATE soar_core)
add_test(NAME test_autograd_engine COMMAND test_autograd_engine)

add_executable(test_optimizers tests/test_optimizers.cpp)
target_link_libraries(test_optimizers PRIVATE soar_core)
add_test(NAME test_optimizers COMMAND test_optimizers)

add_executable(test_async_dataloader tests/test_async_dataloader.cpp)
target_link_libraries(test_async_dataloader PRIVATE soar_core)
add_test(NAME test_async_dataloader COMMAND test_async_dataloader)

