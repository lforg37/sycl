set(LLVM_TOOLS_DIR "${LLVM_BINARY_DIR}/bin/")

get_target_property(SYCL_BINARY_DIR sycl-toolchain BINARY_DIR)

set(SYCL_INCLUDE "${SYCL_INCLUDE_BUILD_DIR}")
set(SYCL_TOOLS_SRC_DIR "${PROJECT_SOURCE_DIR}/tools/")
set(LLVM_BUILD_BINARY_DIRS "${LLVM_BINARY_DIR}/bin/")
set(LLVM_BUILD_LIBRARY_DIRS "${LLVM_BINARY_DIR}/lib/")

set(RT_TEST_ARGS ${RT_TEST_ARGS} "-v")
set(DEPLOY_RT_TEST_ARGS ${DEPLOY_RT_TEST_ARGS} "-v -D SYCL_TOOLS_DIR=${CMAKE_INSTALL_PREFIX}/bin -D SYCL_LIBS_DIR=${CMAKE_INSTALL_PREFIX}/lib${LLVM_LIBDIR_SUFFIX} -D SYCL_INCLUDE=${SYCL_INCLUDE_DEPLOY_DIR}")

find_package(Threads REQUIRED)
set(SYCL_THREADS_LIB ${CMAKE_THREAD_LIBS_INIT})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

list(APPEND SYCL_TEST_DEPS
  sycl-toolchain
  FileCheck
  not
  get_device_count_by_type
  llvm-config
  llvm-cxxdump
  llvm-readobj
  )

list(APPEND SYCL_DEPLOY_TEST_DEPS
  ${SYCL_TEST_DEPS}
  deploy-sycl-toolchain
  )

add_lit_testsuite(check-sycl-deploy "Running the SYCL regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${DEPLOY_RT_TEST_ARGS}
  PARAMS "SYCL_BE=PI_OPENCL"
  DEPENDS ${SYCL_DEPLOY_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl-deploy PROPERTIES FOLDER "SYCL tests")

add_lit_testsuites(SYCL ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${SYCL_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )

add_lit_testsuites(SYCL-DEPLOY ${CMAKE_CURRENT_SOURCE_DIR}
  ARGS ${DEPLOY_RT_TEST_ARGS}
  PARAMS "SYCL_BE=PI_OPENCL"
  DEPENDS ${SYCL_DEPLOY_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )

add_lit_testsuite(check-sycl-opencl "Running the SYCL regression tests for OpenCL"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  PARAMS "SYCL_BE=PI_OPENCL"
  DEPENDS ${SYCL_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl-opencl PROPERTIES FOLDER "SYCL tests")

add_lit_testsuite(check-sycl-level-zero "Running the SYCL regression tests for Level Zero"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  PARAMS "SYCL_BE=PI_LEVEL_ZERO"
  DEPENDS ${SYCL_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl-level-zero PROPERTIES FOLDER "SYCL tests")

add_lit_testsuite(check-sycl-xocc-jmax "Running the SYCL regression tests for sycl-xocc"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  PARAMS "XOCC=only"
  DEPENDS ${SYCL_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl-xocc-jmax PROPERTIES FOLDER "SYCL tests")

add_lit_testsuite(check-sycl-xocc-j4 "Running the SYCL regression tests for sycl-xocc in j4"
  ${CMAKE_CURRENT_BINARY_DIR} "-j 4"
  ARGS ${RT_TEST_ARGS}
  PARAMS "XOCC=only"
  DEPENDS ${SYCL_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl-xocc-j4 PROPERTIES FOLDER "SYCL tests")

add_lit_testsuite(check-sycl-xocc-all-jmax "Running the SYCL regression tests for SYCL on v++"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  PARAMS "XOCC=all"
  DEPENDS ${SYCL_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl-xocc-all-jmax PROPERTIES FOLDER "SYCL tests")

add_lit_testsuite(check-sycl-xocc-all-j4 "Running the SYCL regression tests for SYCL on v++ in -j4"
  ${CMAKE_CURRENT_BINARY_DIR} "-j 4"
  ARGS ${RT_TEST_ARGS}
  PARAMS "XOCC=all"
  DEPENDS ${SYCL_TEST_DEPS}
  EXCLUDE_FROM_CHECK_ALL
  )
set_target_properties(check-sycl-xocc-all-j4 PROPERTIES FOLDER "SYCL tests")

add_custom_target(check-sycl)
add_dependencies(check-sycl check-sycl-opencl check-sycl-level-zero)
set_target_properties(check-sycl PROPERTIES FOLDER "SYCL tests")

if(SYCL_BUILD_PI_CUDA)
  add_lit_testsuite(check-sycl-cuda "Running the SYCL regression tests for CUDA"
    ${CMAKE_CURRENT_BINARY_DIR}
    ARGS ${RT_TEST_ARGS}
    PARAMS "SYCL_BE=PI_CUDA"
    DEPENDS ${SYCL_TEST_DEPS}
    EXCLUDE_FROM_CHECK_ALL
  )
  set_target_properties(check-sycl-cuda PROPERTIES FOLDER "SYCL CUDA tests")

  add_dependencies(check-sycl check-sycl-cuda)

  add_lit_testsuites(SYCL-CUDA ${CMAKE_CURRENT_SOURCE_DIR}
    PARAMS "SYCL_BE=PI_CUDA"
    DEPENDS ${SYCL_TEST_DEPS}
    EXCLUDE_FROM_CHECK_ALL
  )
endif()
