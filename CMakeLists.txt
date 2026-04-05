cmake_minimum_required(VERSION 3.15...3.30)
project(pysommer_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

# --- Armadillo (header-only, vendored) -----------------------------------------
# We use Armadillo in header-only mode and link directly against LAPACK/BLAS.
add_library(armadillo_headers INTERFACE)
target_include_directories(armadillo_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/armadillo/include
)
target_compile_definitions(armadillo_headers INTERFACE
  ARMA_DONT_USE_WRAPPER
  ARMA_DONT_PRINT_ERRORS
  ARMA_64BIT_WORD=1
)

# --- Find LAPACK/BLAS (Accelerate on macOS, system on Linux) -------------------
if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(armadillo_headers INTERFACE ${ACCELERATE_FRAMEWORK})
else()
  find_package(LAPACK REQUIRED)
  find_package(BLAS REQUIRED)
  target_link_libraries(armadillo_headers INTERFACE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()

# --- Build the _sommer_core extension module -----------------------------------
pybind11_add_module(_sommer_core
  pysommer/_cpp/sommer_core.cpp
  pysommer/_cpp/bindings.cpp
)
target_link_libraries(_sommer_core PRIVATE armadillo_headers)

install(TARGETS _sommer_core LIBRARY DESTINATION pysommer/_cpp)
