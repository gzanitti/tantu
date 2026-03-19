add_library(TantuSema
  TypeChecker.cpp
)

target_link_libraries(TantuSema
    PUBLIC
    TantuFrontend
    LLVMSupport
)

target_include_directories(TantuSema
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)