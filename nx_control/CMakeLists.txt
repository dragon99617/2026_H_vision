cmake_minimum_required(VERSION 3.16)
project(ball_nx_control VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NX_CONTROL_USE_OSQP "Use a system OSQP 0.6.x library" OFF)
option(NX_CONTROL_BUILD_TESTS "Build the dependency-free unit tests" ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

add_library(nx_control_core
    src/protocol.cpp
    src/observer.cpp
    src/mpc.cpp
    src/task_manager.cpp
    src/controller.cpp
    src/io.cpp
)
target_include_directories(nx_control_core PUBLIC include)
target_link_libraries(nx_control_core PUBLIC Eigen3::Eigen Threads::Threads)
target_compile_options(nx_control_core PRIVATE -Wall -Wextra -Wpedantic)

if(NX_CONTROL_USE_OSQP)
    find_path(OSQP_INCLUDE_DIR osqp.h PATH_SUFFIXES osqp REQUIRED)
    find_library(OSQP_LIBRARY osqp REQUIRED)
    target_sources(nx_control_core PRIVATE src/osqp_solver.cpp)
    target_include_directories(nx_control_core PRIVATE ${OSQP_INCLUDE_DIR})
    target_link_libraries(nx_control_core PRIVATE ${OSQP_LIBRARY})
    target_compile_definitions(nx_control_core PUBLIC NX_CONTROL_HAS_OSQP=1)
endif()

add_executable(ball_nx_control src/main.cpp)
target_link_libraries(ball_nx_control PRIVATE nx_control_core)

add_executable(ball_nx_replay src/replay.cpp)
target_link_libraries(ball_nx_replay PRIVATE nx_control_core)

install(TARGETS ball_nx_control ball_nx_replay RUNTIME DESTINATION bin)
install(FILES config/nx-control.conf DESTINATION share/ball-nx-control)

if(NX_CONTROL_BUILD_TESTS)
    enable_testing()
    add_executable(nx_control_tests tests/test_main.cpp)
    target_link_libraries(nx_control_tests PRIVATE nx_control_core)
    add_test(NAME nx_control_tests COMMAND nx_control_tests)
    add_test(
        NAME nx_control_replay_smoke
        COMMAND ball_nx_replay
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/replay.csv
                ${CMAKE_CURRENT_BINARY_DIR}/replay-test.csv
                ${CMAKE_CURRENT_SOURCE_DIR}/config/nx-control.conf
    )
endif()
