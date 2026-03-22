add_library(TantuIRGen
  IRGen.cpp
)

target_link_libraries(TantuIRGen
    PUBLIC
    TantuFrontend
    TantuDialect
    MLIRFuncDialect
    MLIR
)

target_include_directories(TantuIRGen
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)