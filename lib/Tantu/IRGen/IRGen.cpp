

#include "Tantu/IRGen/IRGen.h"
#include "Tantu/Dialect/TantuOps.h"
#include "Tantu/Frontend/AST.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include <tuple>
#include <utility>
#include <vector>
void IRGen::emit(Program &program) {
  std::vector<std::pair<std::string, mlir::Attribute>> consts;

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
      consts.emplace_back(constDef->name, value);
    } else {
      auto functionDef = static_cast<FunctionDef *>(def.get());
      std::vector<mlir::Type> paramTypes;
      for (auto &param : functionDef->params)
        paramTypes.push_back(emit(param));

      auto returnType = emit(functionDef->returnType.get());
      auto fnType = mlir::FunctionType::get(ctx, paramTypes, {returnType});
      auto fn = mlir::func::FuncOp::create(loc, functionDef->name, fnType);

      module.push_back(fn);
      fn.addEntryBlock();
      auto block = &fn.getBody().front();
      builder.setInsertionPointToEnd(block);

      for (auto &[name, attr] : consts)
        name_value[name] = builder.create<tantu::ConstantOp>(
            loc, mlir::cast<mlir::TypedAttr>(attr)).getResult();

      auto args = block->getArguments();
      for (size_t i = 0; i < functionDef->params.size(); ++i)
        name_value[functionDef->params[i].identifier.getName()] = args[i];

      for (auto &stmt : functionDef->body)
        emit(*stmt);

      auto retVal = emit(*functionDef->returnExpr);
      builder.create<mlir::func::ReturnOp>(loc, retVal);
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
    mlir::TypedAttr value =
        scalarLiteral->kind == ScalarKind::Float
            ? mlir::cast<mlir::TypedAttr>(
                  mlir::FloatAttr::get(type, scalarLiteral->value))
            : mlir::cast<mlir::TypedAttr>(mlir::IntegerAttr::get(
                  type, static_cast<int64_t>(scalarLiteral->value)));
    return builder.create<tantu::ConstantOp>(mlir::UnknownLoc::get(ctx), value);
  } else if (expr.kind == ExpressionKind::Identifier) {
    auto identifier = static_cast<IdentifierExpr *>(&expr);
    return name_value[identifier->name];
  } else if (expr.kind == ExpressionKind::Call) {
    auto callExpr = static_cast<CallExpr *>(&expr);

    std::string callee = callExpr->callee.getName();
    if (callExpr->builtinOp == BuiltinOp::Transpose) {
      auto permutation =
          static_cast<PermutationExpr *>(callExpr->args[1].get());
      std::vector<mlir::Attribute> permAttrs;
      for (auto val : permutation->values)
        permAttrs.push_back(mlir::IntegerAttr::get(builder.getI64Type(),
                                                   static_cast<int64_t>(val)));
      auto permAttr = mlir::ArrayAttr::get(ctx, permAttrs);

      auto tantuType = static_cast<TensorType *>(callExpr->inferredType);
      auto resultType =
          mlir::RankedTensorType::get(tantuType->shape, builder.getF32Type());
      return builder.create<tantu::TransposeOp>(
          mlir::UnknownLoc::get(ctx), resultType, emit(*callExpr->args[0]),
          permAttr);
    } else if (callExpr->builtinOp == BuiltinOp::Size) {
      auto input = emit(*callExpr->args[0]);
      return builder.create<tantu::SizeOp>(mlir::UnknownLoc::get(ctx),
                                           builder.getI32Type(), input);
    } else if (callExpr->builtinOp == BuiltinOp::Cast) {
      auto input = emit(*callExpr->args[0]);
      return builder.create<tantu::CastOp>(mlir::UnknownLoc::get(ctx),
                                           builder.getF32Type(), input);
    } else if (callExpr->builtinOp == BuiltinOp::Print) {
      auto input = emit(*callExpr->args[0]);
      builder.create<tantu::PrintOp>(mlir::UnknownLoc::get(ctx), input);
      return mlir::Value{};
    } else if (callExpr->builtinOp == BuiltinOp::Matmul) {
      auto lhs = emit(*callExpr->args[0]);
      auto rhs = emit(*callExpr->args[1]);
      auto tantuType = static_cast<TensorType *>(callExpr->inferredType);
      auto resultType =
          mlir::RankedTensorType::get(tantuType->shape, builder.getF32Type());
      return builder.create<tantu::MatmulOp>(mlir::UnknownLoc::get(ctx),
                                             resultType, lhs, rhs);
    } else if (callExpr->builtinOp.has_value()) {
      std::vector<mlir::Value> operands;
      for (auto &arg : callExpr->args) {
        operands.push_back(emit(*arg));
      }

      switch (callExpr->builtinOp.value()) {
      case BuiltinOp::Add:
        return builder.create<tantu::AddOp>(mlir::UnknownLoc::get(ctx),
                                            operands[0], operands[1]);
      case BuiltinOp::Sub:
        return builder.create<tantu::SubOp>(mlir::UnknownLoc::get(ctx),
                                            operands[0], operands[1]);
      case BuiltinOp::Mul:
        return builder.create<tantu::MulOp>(mlir::UnknownLoc::get(ctx),
                                            operands[0], operands[1]);
      case BuiltinOp::Div:
        return builder.create<tantu::DivOp>(mlir::UnknownLoc::get(ctx),
                                            operands[0], operands[1]);
      case BuiltinOp::Max:
        return builder.create<tantu::MaxOp>(mlir::UnknownLoc::get(ctx),
                                            operands[0], operands[1]);
      case BuiltinOp::Exp:
        return builder.create<tantu::ExpOp>(mlir::UnknownLoc::get(ctx),
                                            operands[0]);
      case BuiltinOp::Neg:
        return builder.create<tantu::NegOp>(mlir::UnknownLoc::get(ctx),
                                            operands[0]);
      case BuiltinOp::AddScalar:
        return builder.create<tantu::AddScalarOp>(mlir::UnknownLoc::get(ctx),
                                                  operands[0], operands[1]);
      case BuiltinOp::SubScalar:
        return builder.create<tantu::SubScalarOp>(mlir::UnknownLoc::get(ctx),
                                                  operands[0], operands[1]);
      case BuiltinOp::MulScalar:
        return builder.create<tantu::MulScalarOp>(mlir::UnknownLoc::get(ctx),
                                                  operands[0], operands[1]);
      case BuiltinOp::DivScalar:
        return builder.create<tantu::DivScalarOp>(mlir::UnknownLoc::get(ctx),
                                                  operands[0], operands[1]);
      case BuiltinOp::MaxScalar:
        return builder.create<tantu::MaxScalarOp>(mlir::UnknownLoc::get(ctx),
                                                  operands[0], operands[1]);
      case BuiltinOp::Sum:
        return builder.create<tantu::SumOp>(mlir::UnknownLoc::get(ctx),
                                            builder.getF32Type(), operands[0]);
      case BuiltinOp::MaxReduce:
        return builder.create<tantu::MaxReduceOp>(
            mlir::UnknownLoc::get(ctx), builder.getF32Type(), operands[0]);
      default:
        llvm_unreachable("unhandled builtin op in switch");
      }
    } else { // User-defined function call
      std::vector<mlir::Value> operands;
      for (auto &arg : callExpr->args) {
        operands.push_back(emit(*arg));
      }
      auto func = module.lookupSymbol<mlir::func::FuncOp>(callee);
      return builder
          .create<mlir::func::CallOp>(mlir::UnknownLoc::get(ctx), func,
                                      operands)
          .getResult(0);
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
  return emit(param.type.get());
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
