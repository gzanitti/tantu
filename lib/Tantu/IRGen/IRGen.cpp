

#include "Tantu/IRGen/IRGen.h"
#include "Tantu/Dialect/TantuOps.h"
#include "Tantu/Frontend/AST.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include <tuple>
#include <utility>
#include <vector>
void IRGen::emit(Program &program) {
  // std::unordered_map<Expression *, mlir::Value> expr_value;
  // std::unordered_map<std::string, mlir::Value> name_value;
  std::vector<std::pair<mlir::Attribute, mlir::Type>> consts;

  auto loc = mlir::UnknownLoc::get(ctx);
  module = mlir::ModuleOp::create(loc);

  for (auto &def : program.definitions()) {
    if (def->kind == DefinitionKind::Const) {
      // ConstDef are always ScalarType
      auto constDef = static_cast<ConstDef *>(def.get());
      auto scalarType = static_cast<ScalarType *>(constDef->type.get());
      mlir::Type type = emit(scalarType);
      mlir::Attribute value =
          scalarType->kind == ScalarKind::Float
              ? static_cast<mlir::Attribute>(
                    mlir::FloatAttr::get(type, constDef->value.value))
              : static_cast<mlir::Attribute>(mlir::IntegerAttr::get(
                    type, static_cast<int64_t>(constDef->value.value)));
      consts.push_back(std::make_pair(value, type));

      name_value.insert_or_assign(constDef->name, value);
    } else {
      auto functionDef = static_cast<FunctionDef *>(def.get());
      auto loc = mlir::UnknownLoc::get(ctx);
      std::vector<mlir::Type> params;
      for (auto &param : functionDef->params) {
        params.push_back(emit(param));
      }
      auto returnType = emit(functionDef->returnType.get());
      auto fnType = mlir::FunctionType::get(ctx, params, {returnType});
      auto fn = mlir::func::FuncOp::create(loc, functionDef->name, fnType);

      module.push_back(fn);
      fn.addEntryBlock();
      auto block = &fn.getBody().front();
      builder.setInsertionPointToEnd(block);

      for (auto &c : consts) {
        auto op = builder.create<tantu::ConstantOp>(loc, c.first, c.second)
                      .getValue();
      }

      for (auto &stmt : functionDef->body) {
        emit(*stmt);
      }
    }
  }
}

void IRGen::emit(Statement &stmt) {
  auto &let = static_cast<LetBinding &>(stmt);
  std::string name = let.identifier.getName();
  auto value = emit(*let.expr);
  name_value.insert_or_assign(name, value);
}

mlir::Value IRGen::emit(Expression &expr) {
  if (expr.kind == ExpressionKind::ScalarLiteral) {
    auto scalarLiteral = static_cast<ScalarLiteralExpr *>(&expr);
    mlir::Type type = emit(scalarLiteral->inferredType);
    mlir::Attribute value =
        scalarLiteral->kind == ScalarKind::Float
            ? static_cast<mlir::Attribute>(
                  mlir::FloatAttr::get(type, scalarLiteral->value))
            : static_cast<mlir::Attribute>(mlir::IntegerAttr::get(
                  type, static_cast<int64_t>(scalarLiteral->value)));
    return builder.create<tantu::ConstantOp>(mlir::UnknownLoc::get(ctx), value,
                                             type);
  } else if (expr.kind == ExpressionKind::Identifier) {
    auto identifier = static_cast<IdentifierExpr *>(&expr);
    return name_value[identifier->name];
  } else if (expr.kind == ExpressionKind::Call) {
    auto callExpr = static_cast<CallExpr *>(&expr);
    std::vector<mlir::Value> operands;
    for (auto &arg : callExpr->args) {
      operands.push_back(emit(*arg));
    }
    std::string callee = callExpr->callee.getName();
    if (callee == "transpose") {
      auto permutation =
          static_cast<PermutationExpr *>(callExpr->args[1].get());
      std::vector<mlir::Attribute> permAttrs;
      for (auto val : permutation->values)
        permAttrs.push_back(mlir::IntegerAttr::get(builder.getI64Type(),
                                                   static_cast<int64_t>(val)));
      auto permAttr = mlir::ArrayAttr::get(ctx, permAttrs);
      return builder
          .create<tantu::TransposeOp>(mlir::UnknownLoc::get(ctx), operands[0],
                                      permAttr)
          .getResult();
    } else {
    }
  } else if (expr.kind == ExpressionKind::TensorKind) {
    auto tensorLiteral = static_cast<TensorExpr *>(&expr);
    std::vector<mlir::Attribute> values;
    for (auto &value : tensorLiteral->values) {
      auto scalarLiteral = static_cast<ScalarLiteralExpr *>(value.get());
      mlir::Type type = emit(scalarLiteral->inferredType);
      mlir::Attribute attr =
          scalarLiteral->kind == ScalarKind::Float
              ? static_cast<mlir::Attribute>(
                    mlir::FloatAttr::get(type, scalarLiteral->value))
              : static_cast<mlir::Attribute>(mlir::IntegerAttr::get(
                    type, static_cast<int64_t>(scalarLiteral->value)));
      values.push_back(attr);
    }
    auto tensorType =
        mlir::RankedTensorType::get(tensorLiteral->shape, builder.getF32Type());
    return builder.create<tantu::TensorLiteralOp>(
        mlir::UnknownLoc::get(ctx),
        mlir::DenseElementsAttr::get(tensorType, values));
  } else {
    // Case (expr.kind == ExpressionKind::Permutation)
    // Solved inside ExpressionKind::Call when name == 'transpose'
    llvm_unreachable("PermutationExpr should not be emitted directly");
  }
}
mlir::Type IRGen::emit(Param &param) {
  std::string name = param.identifier.getName();
  auto type = emit(param.type.get());
  name_value.insert_or_assign(name, type);
  return type;
}

mlir::Type IRGen::emit(Type *type) {
  if (type->kind == TypeKind::Scalar) {
    auto scalarType = static_cast<ScalarType *>(type);
    if (scalarType->kind == ScalarKind::Float) {
      return builder.getF32Type();
    } else {
      return builder.getI32Type();
    }
  } else {
    auto tensorType = static_cast<TensorType *>(type);
    return mlir::RankedTensorType::get(tensorType->shape, builder.getF32Type());
  }
}
