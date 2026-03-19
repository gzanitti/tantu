#include "Tantu/Sema/TypeChecker.h"
#include "Tantu/Frontend/AST.h"
#include "llvm/Support/Error.h"
#include <iostream>
#include <set>
#include <vector>

static size_t hash_combine(size_t seed, const int64_t elem) {
  return (seed ^ (std::hash<int64_t>{}(elem) + 0x9e3779b9 + (seed << 6) +
                  (seed >> 2)));
}

size_t VectorHash::operator()(const std::vector<int64_t> &vec) const {
  size_t seed = 0;
  for (const auto &elem : vec) {
    seed = hash_combine(seed, elem);
  }
  return seed;
}

TypeContext::TypeContext() {}
TypeContext::~TypeContext() {}

Type *TypeContext::lookup(ScalarKind kind) {
  if (scalarTypes.find(kind) != scalarTypes.end()) {
    return scalarTypes[kind].get();
  }
  auto t = std::make_unique<ScalarType>(kind);
  scalarTypes[kind] = std::move(t);
  return scalarTypes[kind].get();
}

Type *TypeContext::lookup(std::vector<int64_t> shape) {
  if (tensorTypes.find(shape) != tensorTypes.end()) {
    return tensorTypes[shape].get();
  }
  auto t = std::make_unique<TensorType>(shape);
  tensorTypes[shape] = std::move(t);
  return tensorTypes[shape].get();
}

Type *TypeChecker::validateSameShapeTensors(std::vector<Expression *> &args,
                                            TypeContext &ctx,
                                            std::vector<std::string> &errors) {
  if (args.size() != 2) {
    errors.push_back("Function add expects 2 arguments.");
    return nullptr;
  }
  auto arg1 = args[0]->inferredType;
  auto arg2 = args[1]->inferredType;
  if (arg1->kind != arg2->kind) {
    errors.push_back("Function add expects arguments of the same type.");
    return nullptr;
  }
  if (arg1->kind == TypeKind::Scalar) {
    errors.push_back("Function ´add´ arguments must be tensor.");
    return nullptr;
  } else if (arg1->kind == TypeKind::Tensor) {
    auto tensorType1 = static_cast<TensorType *>(arg1);
    auto tensorType2 = static_cast<TensorType *>(arg2);

    if (tensorType1->shape != tensorType2->shape) {
      errors.push_back("Function ´add´ arguments must have the same shape.");
      return nullptr;
    }
    auto tensor = ctx.lookup(tensorType1->shape);
    return tensor;
  }
  return nullptr;
}

Type *TypeChecker::validateSingleTensor(std::vector<Expression *> &args,
                                        TypeContext &ctx,
                                        std::vector<std::string> &errors) {
  if (args.size() != 1) {
    errors.push_back("Function expects 1 argument.");
    return nullptr;
  }
  auto arg = args[0]->inferredType;
  if (arg->kind != TypeKind::Tensor) {
    errors.push_back("Function expects a single tensor argument.");
    return nullptr;
  }
  auto tensorType = static_cast<TensorType *>(arg);
  auto tensor = ctx.lookup(tensorType->shape);
  return tensor;
};

Type *TypeChecker::validateTensorScalarType(std::vector<Expression *> &args,
                                            TypeContext &ctx,
                                            std::vector<std::string> &errors) {
  if (args.size() != 2) {
    errors.push_back("Function expects 2 arguments.");
    return nullptr;
  }

  auto arg1 = args[0]->inferredType;
  auto arg2 = args[1]->inferredType;
  if (arg1->kind != TypeKind::Tensor) {
    errors.push_back("First argument must be a tensor.");
    return nullptr;
  }
  if (arg2->kind != TypeKind::Scalar) {
    errors.push_back("Second argument must be a scalar.");
    return nullptr;
  }

  auto tensorType = static_cast<TensorType *>(arg1);
  auto tensor = ctx.lookup(tensorType->shape);
  return tensor;
};

Type *TypeChecker::validateTensorToF32(std::vector<Expression *> &args,
                                       TypeContext &ctx,
                                       std::vector<std::string> &errors) {
  if (args.size() != 1) {
    errors.push_back("Function expects 1 argument.");
    return nullptr;
  }
  auto arg = args[0]->inferredType;
  if (arg->kind != TypeKind::Tensor) {
    errors.push_back("Argument must be a tensor.");
    return nullptr;
  }
  auto returnType = ctx.lookup(ScalarKind::Float);
  return returnType;
}

Type *TypeChecker::validateTensorToI32(std::vector<Expression *> &args,
                                       TypeContext &ctx,
                                       std::vector<std::string> &errors) {
  if (args.size() != 1) {
    errors.push_back("Function expects 1 argument.");
    return nullptr;
  }
  auto arg = args[0]->inferredType;
  if (arg->kind != TypeKind::Tensor) {
    errors.push_back("Argument must be a tensor.");
    return nullptr;
  }
  auto returnType = ctx.lookup(ScalarKind::Integer);
  return returnType;
}

Type *TypeChecker::validateMatMul(std::vector<Expression *> &args,
                                  TypeContext &ctx,
                                  std::vector<std::string> &errors) {
  if (args.size() != 2) {
    errors.push_back("Function matmul expects 2 arguments.");
    return nullptr;
  }
  auto arg1 = args[0]->inferredType;
  auto arg2 = args[1]->inferredType;
  if (arg1->kind != TypeKind::Tensor || arg2->kind != TypeKind::Tensor) {
    errors.push_back("Function matmul expects two tensors.");
    return nullptr;
  }
  auto tensorType1 = static_cast<TensorType *>(arg1);
  auto tensorType2 = static_cast<TensorType *>(arg2);

  if (tensorType1->shape.size() != 2 || tensorType2->shape.size() != 2) {
    errors.push_back("Function matmul expects tensors with 2 dimensions.");
    return nullptr;
  }

  if (tensorType1->shape[1] != tensorType2->shape[0]) {
    errors.push_back("Function matmul expects tensors with compatible shapes.");
    return nullptr;
  }
  auto resultShape = {tensorType1->shape[0], tensorType2->shape[1]};
  auto result = ctx.lookup(resultShape);
  return result;
}

Type *TypeChecker::validateTranspose(std::vector<Expression *> &args,
                                     TypeContext &ctx,
                                     std::vector<std::string> &errors) {
  if (args.size() != 2) {
    errors.push_back("Function transpose expects 2 arguments.");
    return nullptr;
  }

  auto arg1 = args[0]->inferredType;
  if (arg1->kind != TypeKind::Tensor) {
    errors.push_back("First argument must be a tensor.");
    return nullptr;
  }
  auto perm = static_cast<PermutationExpr *>(args[1]);
  auto tensorType = static_cast<TensorType *>(arg1);

  if (perm->values.size() != tensorType->shape.size()) {
    errors.push_back("Function transpose expects a permutation with the same "
                     "size as the tensor.");
    return nullptr;
  }

  std::set<size_t> seen;
  std::vector<int64_t> resultShape(perm->values.size());
  for (size_t i = 0; i < perm->values.size(); i++) {
    int val = perm->values[i];
    if (val < 0 || val >= tensorType->shape.size())
      errors.push_back("perm value out of range: " + std::to_string(val));
    if (!seen.insert(val).second)
      errors.push_back("perm contains duplicate value: " + std::to_string(val));
    resultShape[i] = tensorType->shape[val];
  }

  auto resultType = ctx.lookup(resultShape);
  return resultType;
}

Type *TypeChecker::validatePrint(std::vector<Expression *> &args,
                                 TypeContext &ctx,
                                 std::vector<std::string> &errors) {

  if (args.size() != 1) {
    errors.push_back("Function print expects 1 argument.");
    return nullptr;
  }
  auto arg = args[0]->inferredType;
  if (arg->kind != TypeKind::Tensor) {
    errors.push_back("Function print expects a tensor argument.");
    return nullptr;
  }
  auto tensorType = static_cast<TensorType *>(arg);
  auto tensor = ctx.lookup(tensorType->shape);
  return tensor;
}

Type *TypeChecker::validateCast(std::vector<Expression *> &args,
                                TypeContext &ctx,
                                std::vector<std::string> &errors) {
  if (args.size() != 1) {
    errors.push_back("Function cast expects 1 argument.");
    return nullptr;
  }

  auto arg = args[0]->inferredType;
  if (arg->kind != TypeKind::Scalar) {
    errors.push_back("Function cast expects argument to be scalar.");
    return nullptr;
  }
  auto scalarType = static_cast<ScalarType *>(arg);
  if (scalarType->kind != ScalarKind::Integer) {
    errors.push_back("Function cast expects argument to be integer.");
    return nullptr;
  }

  auto scalar = ctx.lookup(ScalarKind::Float);
  return scalar;
}

TypeChecker::TypeChecker() {
  builtin_descriptors["add"] = &TypeChecker::validateSameShapeTensors;
  builtin_descriptors["sub"] = &TypeChecker::validateSameShapeTensors;
  builtin_descriptors["mul"] = &TypeChecker::validateSameShapeTensors;
  builtin_descriptors["div"] = &TypeChecker::validateSameShapeTensors;
  builtin_descriptors["max"] = &TypeChecker::validateSameShapeTensors;

  builtin_descriptors["neg"] = &TypeChecker::validateSingleTensor;
  builtin_descriptors["exp"] = &TypeChecker::validateSingleTensor;

  builtin_descriptors["add_scalar"] = &TypeChecker::validateTensorScalarType;
  builtin_descriptors["sub_scalar"] = &TypeChecker::validateTensorScalarType;
  builtin_descriptors["mul_scalar"] = &TypeChecker::validateTensorScalarType;
  builtin_descriptors["div_scalar"] = &TypeChecker::validateTensorScalarType;
  builtin_descriptors["max_scalar"] = &TypeChecker::validateTensorScalarType;

  builtin_descriptors["sum"] = &TypeChecker::validateTensorToF32;
  builtin_descriptors["max_reduce"] = &TypeChecker::validateTensorToF32;
  builtin_descriptors["size"] = &TypeChecker::validateTensorToI32;

  builtin_descriptors["matmul"] = &TypeChecker::validateMatMul;
  builtin_descriptors["transpose"] = &TypeChecker::validateTranspose;
  builtin_descriptors["print"] = &TypeChecker::validatePrint;
  builtin_descriptors["cast"] = &TypeChecker::validateCast;
}

llvm::Error TypeChecker::check(Program *prog) {
  prog->accept(*this);
  if (!errors.empty()) {
    std::string msg;
    for (const auto &e : errors)
      msg += e + "\n";
    return llvm::make_error<llvm::StringError>(msg,
                                               llvm::inconvertibleErrorCode());
  }
  return llvm::Error::success();
}

Type *TypeChecker::canonalize(Type *type) {
  if (type->kind == TypeKind::Scalar) {
    auto scalarType = static_cast<ScalarType *>(type);
    auto scalar = context.lookup(scalarType->kind);
    return scalar;
  } else if (type->kind == TypeKind::Tensor) {
    auto tensorType = static_cast<TensorType *>(type);
    auto tensor = context.lookup(tensorType->shape);
    return tensor;
  }
  return nullptr;
}

void TypeChecker::visit(Program &stmt) {
  for (const auto &def : stmt.definitions()) {
    std::cout << "Checking definition: " << def->getName() << std::endl;
    def->accept(*this);
  }
}

void TypeChecker::visit(ConstDef &stmt) {
  if (stmt.type->kind == TypeKind::Scalar) {
    auto scalarType = static_cast<ScalarType *>(stmt.type.get());
    auto scalar = context.lookup(scalarType->kind);
    symbol_table[stmt.name] = scalar;
  } else if (stmt.type->kind == TypeKind::Tensor) {
    auto tensorType = static_cast<TensorType *>(stmt.type.get());
    auto tensor = context.lookup(tensorType->shape);
    symbol_table[stmt.name] = tensor;
  }
}

void TypeChecker::visit(FunctionDef &stmt) {
  for (auto &param : stmt.params) {
    std::cout << "Checking parameter: " << param.identifier.getName()
              << std::endl;
    param.accept(*this);
  }
  for (const auto &stmt : stmt.body) {
    stmt->accept(*this);
  }
  stmt.returnExpr->accept(*this);
  if (errors.empty()) {
    auto returnType = canonalize(stmt.returnType.get());
    if (returnType != stmt.returnExpr->inferredType) {
      errors.push_back("Function return type does not match inferred type. "
                       "Expected: " +
                       stmt.returnType->toString() +
                       ", got: " + stmt.returnExpr->inferredType->toString());
    }
  }
}

void TypeChecker::visit(Param &param) {
  if (param.type->kind == TypeKind::Scalar) {
    auto scalarType = static_cast<ScalarType *>(param.type.get());
    auto scalar = context.lookup(scalarType->kind);
    symbol_table[param.identifier.getName()] = scalar;
  } else if (param.type->kind == TypeKind::Tensor) {
    auto tensorType = static_cast<TensorType *>(param.type.get());
    auto tensor = context.lookup(tensorType->shape);
    symbol_table[param.identifier.getName()] = tensor;
  }
}

void TypeChecker::visit(LetBinding &stmt) {
  if (symbol_table.count(stmt.identifier.getName())) {
    errors.push_back("Redefinition of '" + stmt.identifier.getName() + "'");
    return;
  }
  stmt.expr->accept(*this);
  symbol_table[stmt.identifier.getName()] = stmt.expr->inferredType;
}

void TypeChecker::visit(IdentifierExpr &expr) {
  if (symbol_table.find(expr.getName()) != symbol_table.end()) {
    expr.inferredType = symbol_table[expr.getName()];
  } else {
    errors.push_back("Undefined identifier: " + expr.getName());
  }
}
void TypeChecker::visit(TensorExpr &expr) {
  int64_t expectedElems = 1;
  for (int64_t dim : expr.shape)
    expectedElems *= dim;
  if ((int64_t)expr.values.size() != expectedElems) {
    errors.push_back(
        "tensor literal has " + std::to_string(expr.values.size()) +
        " elements but type requires " + std::to_string(expectedElems));
    return;
  }
  expr.inferredType = context.lookup(expr.shape);
}

void TypeChecker::visit(ScalarLiteralExpr &expr) {
  expr.inferredType = context.lookup(expr.kind);
}

void TypeChecker::visit(CallExpr &expr) {
  std::vector<Expression *> args;
  for (const auto &arg : expr.args) {
    arg->accept(*this);
    args.push_back(arg.get());
  }

  if (!errors.empty())
    return;

  // TODO: resolve user-defined function calls, only builtins are supported now
  // See TypeCheckerTest.TwoFunctions [  FAILED  ]

  expr.inferredType =
      builtin_descriptors[expr.callee.getName()](args, context, errors);
}

void TypeChecker::visit(TensorType &type) {}
void TypeChecker::visit(ScalarType &type) {}
void TypeChecker::visit(PermutationExpr &expr) {}
