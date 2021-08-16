#Optional PETSc to enable linear dynamics in lindblad/linear
option(EnablePETSc "Enable lindbladian linear dynamics using PETSc" OFF)
if(EnablePETSc)
	find_package(PETSc REQUIRED)
	include_directories("${PETSC_INCLUDE_DIR}")
	add_definitions("-DPETSC_ENABLED")
endif()

add_FeynWann_executable(init)
add_FeynWann_executable(run)
add_FeynWann_executable(linear)
add_FeynWann_executable(makeModel)

target_link_libraries(run ${GSL_BLAS_LIBRARY})
target_link_libraries(linear ${GSL_BLAS_LIBRARY} ${PETSC_LIBRARY})
