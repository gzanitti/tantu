add_library(TantuFrontend
  AST.cpp
  Lexer.cpp
  Parser.cpp
  PrettyPrinter.cpp
)

target_link_libraries(TantuFrontend
    PUBLIC
    LLVMSupport
)

target_include_directories(TantuFrontend
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)