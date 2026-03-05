import lit.formats
import lit.llvm

config.name = "TantuBufferizationExamples"
config.test_format = lit.formats.ShTest(not lit.llvm.llvm_config.use_lit_shell)
config.suffixes = [".mlir"]

config.substitutions.append(("%tantu-opt", config.tantu_opt))