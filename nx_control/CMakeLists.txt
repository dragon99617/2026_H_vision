cmake_minimum_required(VERSION 3.16)
project(ball_nx_control VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NX_CONTROL_BUILD_TESTS "Build the dependency-free unit tests" ON)
option(NX_CONTROL_NATIVE_OPTIMIZATION
       "Tune Release binaries for the Jetson CPU executing the build" ON)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options($<$<CONFIG:Release>:-O3>)
    if(NX_CONTROL_NATIVE_OPTIMIZATION)
        add_compile_options($<$<CONFIG:Release>:-mcpu=native>)
    endif()
endif()

add_library(nx_control_core
    src/protocol.cpp
    src/observer.cpp
    src/pid.cpp
    src/task_manager.cpp
    src/friction_compensator.cpp
    src/controller.cpp
    src/runtime_command.cpp
    src/io.cpp
)
target_include_directories(nx_control_core PUBLIC include)
target_link_libraries(nx_control_core PUBLIC Eigen3::Eigen Threads::Threads)
target_compile_options(nx_control_core PRIVATE -Wall -Wextra -Wpedantic)

include(CheckIPOSupported)
check_ipo_supported(RESULT NX_CONTROL_IPO_SUPPORTED OUTPUT NX_CONTROL_IPO_ERROR)
if(NX_CONTROL_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(TARGET nx_control_core PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

add_executable(ball_nx_control src/main.cpp)
target_link_libraries(ball_nx_control PRIVATE nx_control_core)

add_executable(ball_nx_replay src/replay.cpp)
target_link_libraries(ball_nx_replay PRIVATE nx_control_core)

add_executable(tube_hold_sender src/tube_hold_sender.cpp)
target_link_libraries(tube_hold_sender PRIVATE nx_control_core)
target_compile_options(tube_hold_sender PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS ball_nx_control ball_nx_replay tube_hold_sender RUNTIME DESTINATION bin)
install(FILES config/nx-control.conf DESTINATION share/ball-nx-control)

if(NX_CONTROL_BUILD_TESTS)
    enable_testing()
    add_executable(nx_control_tests tests/test_main.cpp)
    target_link_libraries(nx_control_tests PRIVATE nx_control_core)
    target_compile_definitions(
        nx_control_tests PRIVATE
        NX_CONTROL_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME nx_control_tests COMMAND nx_control_tests)
    add_test(
        NAME nx_control_replay_smoke
        COMMAND ball_nx_replay
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/replay.csv
                ${CMAKE_CURRENT_BINARY_DIR}/replay-test.csv
                ${CMAKE_CURRENT_SOURCE_DIR}/config/nx-control.conf
    )
    add_test(
        NAME tube_hold_sender_smoke
        COMMAND tube_hold_sender --dry-run --max-frames 3
                --start-command-id 100 --verbose
    )
endif()
