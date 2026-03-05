import lit.formats
import os

config.name = "Tantu"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

# TANTU_BUILD_DIR is injected by lit.site.cfg.py.in via CMake
tantu_build_dir = config.environment.get("TANTU_BUILD_DIR", "")
tantu_opt = os.path.join(tantu_build_dir, "tools", "tantu-opt", "tantu-opt")

config.substitutions.append(("%tantu-opt", tantu_opt))