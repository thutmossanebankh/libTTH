cmake_minimum_required(VERSION 3.18) # Version requise pour une bonne gestion de CUDA

# Activation explicite des langages C++ ET CUDA
project(TTH_Core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17) # Utilisation du C++17 pour CUDA également

find_package(pybind11 REQUIRED)

# Ajout du fichier .cu à la compilation du module
pybind11_add_module(tth_core tth_core.cpp tth_cuda.cu)

# Optimisation pour les GPU NVIDIA
target_compile_options(tth_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

