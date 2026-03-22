add_library(TantuFrontend
  AST.cpp
  Lexer.cpp
  Parser.cpp
  PrettyPrinter.cpp
)

target_link_libraries(TantuFrontend
    PUBLIC
    LLVM
)

target_include_directories(TantuFrontend
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)