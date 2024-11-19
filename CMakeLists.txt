cmake_minimum_required(VERSION 3.28.1)
project(FLARE)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" )

include_directories(../libs)
include_directories(utils)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_definitions(-DUBUNTU_NOTEBOOK)
set(Torch_DIR "/mnt/c/Users/redamancyguy/Desktop/libtorch/share/cmake/Torch")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS fiber date_time system filesystem)
link_libraries("${TORCH_LIBRARIES}")

#add_executable(main main.cpp
#        utils/DEFINE.h
#        utils/WorkLoad.hpp
#        utils/dataset.hpp
#        indexes/VPPLUS.hpp
#        utils/calculate.hpp
#        indexes/Index.hpp
#)

add_executable(affix FLARE.cpp FLARE.hpp indexes/VPPLUS.hpp utils/Network.hpp utils/Model.hpp utils/TimerClock.hpp utils/WorkLoad.hpp FLARE.hpp)

add_executable(affix_fast FLAREFast.hpp runFLAREfast.cpp indexes/VPPLUSS.hpp utils/Network.hpp utils/Model.hpp utils/TimerClock.hpp utils/WorkLoad.hpp)





