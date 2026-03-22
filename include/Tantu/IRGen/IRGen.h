
#pragma once

#include "Tantu/Frontend/AST.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include <unordered_map>

class IRGen {
public:
  IRGen(mlir::MLIRContext *ctx) : ctx(ctx), builder(ctx) {}

  void emit(Program &program);
  mlir::ModuleOp getModule() { return module; }

private:
  mlir::MLIRContext *ctx;
  mlir::OpBuilder builder;
  mlir::ModuleOp module;
  std::unordered_map<std::string, mlir::Value> name_value;

  void emit(Statement &stmt);
  mlir::Value emit(Expression &expr);
  mlir::Type emit(Param &param);
  mlir::Type emit(Type *type);
};
