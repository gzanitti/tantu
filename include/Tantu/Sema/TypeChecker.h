
#pragma once

#include "Tantu/Frontend/AST.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <unordered_map>
#include <vector>

struct VectorHash {
  size_t operator()(const std::vector<int64_t> &vec) const;
};

class TypeContext {
public:
  TypeContext();
  ~TypeContext();

  Type *lookup(ScalarKind kind);
  Type *lookup(std::vector<int64_t> shape);

private:
  std::unordered_map<ScalarKind, std::unique_ptr<Type>> scalarTypes;
  std::unordered_map<std::vector<int64_t>, std::unique_ptr<Type>, VectorHash>
      tensorTypes;
};

class TypeChecker : public Visitor {
  TypeContext context;
  std::unordered_map<std::string, Type *> symbol_table;
  std::unordered_map<
      std::string,
      std::function<Type *(std::vector<Expression *> &, TypeContext &,
                           std::vector<std::string> &)>>
      builtin_descriptors;
  std::optional<Type *> currentReturnType;
  std::vector<std::string> errors;

public:
  llvm::Error check(Program *prog);
  TypeChecker();

private:
  void visit(LetBinding &stmt) override;
  void visit(Param &param) override;
  void visit(IdentifierExpr &expr) override;
  void visit(ScalarLiteralExpr &expr) override;
  void visit(PermutationExpr &expr) override;
  void visit(TensorType &type) override;
  void visit(ScalarType &type) override;
  void visit(CallExpr &expr) override;
  void visit(TensorExpr &expr) override;
  void visit(FunctionDef &def) override;
  void visit(ConstDef &def) override;
  void visit(Program &prog) override;
  Type *canonalize(Type *type);

  static Type *validateSameShapeTensors(std::vector<Expression *> &args,
                                        TypeContext &ctx,
                                        std::vector<std::string> &errors);
  static Type *validateSingleTensor(std::vector<Expression *> &args,
                                    TypeContext &ctx,
                                    std::vector<std::string> &errors);
  static Type *validateTensorScalarType(std::vector<Expression *> &args,
                                        TypeContext &ctx,
                                        std::vector<std::string> &errors);

  static Type *validateTensorToF32(std::vector<Expression *> &args,
                                   TypeContext &ctx,
                                   std::vector<std::string> &errors);
  static Type *validateTensorToI32(std::vector<Expression *> &args,
                                   TypeContext &ctx,
                                   std::vector<std::string> &errors);

  static Type *validateMatMul(std::vector<Expression *> &args, TypeContext &ctx,
                              std::vector<std::string> &errors);

  static Type *validateCast(std::vector<Expression *> &args, TypeContext &ctx,
                            std::vector<std::string> &errors);

  static Type *validateTranspose(std::vector<Expression *> &args,
                                 TypeContext &ctx,
                                 std::vector<std::string> &errors);

  static Type *validatePrint(std::vector<Expression *> &args, TypeContext &ctx,
                             std::vector<std::string> &errors);
};
