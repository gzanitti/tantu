find_program(LLVM_LIT lit)
if(NOT LLVM_LIT)
  message(WARNING "lit not found, tests will not be available")
  return()
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
)

add_custom_target(check-tantu
  COMMAND ${LLVM_LIT} ${CMAKE_CURRENT_BINARY_DIR} -v
  DEPENDS tantu-opt
  COMMENT "Running Tantu tests"
)