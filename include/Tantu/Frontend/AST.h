#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class Visitor;

enum TypeKind { Scalar, Tensor };

class Type {
public:
  Type(TypeKind kind) : kind(kind) {}
  virtual void accept(Visitor &visitor) = 0;
  virtual std::string toString() const = 0;
  virtual ~Type() = default;
  TypeKind kind;
};

enum ScalarKind { Integer, Float };

class ScalarType : public Type {
public:
  ScalarType(ScalarKind kind) : Type(TypeKind::Scalar), kind(kind) {}
  ScalarKind kind;
  void accept(Visitor &visitor) override;
  std::string toString() const override;
  bool operator==(const ScalarType &other) const;
};

class TensorType : public Type {
public:
  TensorType(std::vector<int64_t> shape)
      : Type(TypeKind::Tensor), shape(shape) {}
  std::vector<int64_t> shape;
  void accept(Visitor &visitor) override;
  std::string toString() const override;
  bool operator==(const TensorType &other) const;
};

class Statement {
public:
  virtual void accept(Visitor &visitor) = 0;
  virtual ~Statement() = default;
};

enum ExpressionKind {
  ScalarLiteral,
  Identifier,
  Permutation,
  Call,
  TensorKind
};

class Expression {
public:
  Expression(ExpressionKind kind) : kind(kind){};
  virtual void accept(Visitor &visitor) = 0;
  virtual ~Expression() = default;
  ExpressionKind kind;
  Type *inferredType = nullptr;
};

class ScalarLiteralExpr : public Expression {
public:
  ScalarLiteralExpr(double value, ScalarKind kind)
      : Expression(ExpressionKind::ScalarLiteral), value(value), kind(kind) {}
  double value;
  ScalarKind kind;
  void accept(Visitor &visitor) override;
};

class IdentifierExpr : public Expression {
public:
  IdentifierExpr(std::string name)
      : Expression(ExpressionKind::Identifier), name(name) {}
  const std::string &getName() const { return name; }
  std::string name;
  void accept(Visitor &visitor) override;
};

class PermutationExpr : public Expression {
public:
  PermutationExpr(std::vector<size_t> values)
      : Expression(ExpressionKind::Permutation), values(values) {}
  std::vector<size_t> values;
  void accept(Visitor &visitor) override;
};

class Param {
public:
  Param(IdentifierExpr identifier, std::unique_ptr<Type> type)
      : identifier(identifier), type(std::move(type)) {}
  IdentifierExpr identifier;
  std::unique_ptr<Type> type;
  void accept(Visitor &visitor);
};

class LetBinding : public Statement {
public:
  LetBinding(IdentifierExpr identifier, std::unique_ptr<Expression> expr)
      : identifier(identifier), expr(std::move(expr)) {}
  IdentifierExpr identifier;
  std::unique_ptr<Expression> expr;
  void accept(Visitor &visitor) override;
};

enum BuiltinOp {
  Add,
  Sub,
  Mul,
  Div,
  Max,
  Exp,
  Neg,
  AddScalar,
  SubScalar,
  MulScalar,
  DivScalar,
  MaxScalar,
  Sum,
  MaxReduce,
  Size,
  Cast,
  Transpose,
  Matmul,
  Print
};

const char *builtinOpName(BuiltinOp op);
std::optional<BuiltinOp> getBuiltinOp(std::string_view name);

class CallExpr : public Expression {
public:
  CallExpr(IdentifierExpr callee, std::vector<std::unique_ptr<Expression>> args,
           std::optional<BuiltinOp> builtinOp)
      : Expression(ExpressionKind::Call), callee(callee), args(std::move(args)),
        builtinOp(builtinOp) {}
  IdentifierExpr callee;
  std::vector<std::unique_ptr<Expression>> args;
  std::optional<BuiltinOp> builtinOp;
  void accept(Visitor &visitor) override;
};

class TensorExpr : public Expression {
public:
  TensorExpr(std::vector<int64_t> shape,
             std::vector<std::unique_ptr<ScalarLiteralExpr>> values)
      : Expression(ExpressionKind::TensorKind), shape(shape),
        values(std::move(values)) {}
  std::vector<int64_t> shape;
  std::vector<std::unique_ptr<ScalarLiteralExpr>> values;
  void accept(Visitor &visitor) override;
};

enum DefinitionKind { Const, Function };

class Definition {
public:
  Definition(std::string name, DefinitionKind kind) : name(name), kind(kind) {}
  const std::string &getName() const { return name; }
  virtual void accept(Visitor &visitor) = 0;
  std::string name;
  DefinitionKind kind;
  virtual ~Definition() = default;
};

class FunctionDef : public Definition {
public:
  FunctionDef(std::string name, std::vector<Param> params,
              std::vector<std::unique_ptr<Statement>> body,
              std::unique_ptr<Expression> returnExpr,
              std::unique_ptr<Type> returnType)
      : Definition(name, DefinitionKind::Function), params(std::move(params)),
        body(std::move(body)), returnExpr(std::move(returnExpr)),
        returnType(std::move(returnType)) {}
  std::vector<Param> params;
  std::vector<std::unique_ptr<Statement>> body;
  std::unique_ptr<Expression> returnExpr;
  std::unique_ptr<Type> returnType;
  void accept(Visitor &visitor) override;
  const std::string &getName() const { return name; }
};

class ConstDef : public Definition {
public:
  ConstDef(std::string name, ScalarLiteralExpr value,
           std::unique_ptr<Type> type)
      : Definition(name, DefinitionKind::Const), value(value),
        type(std::move(type)) {}
  ScalarLiteralExpr value;
  std::unique_ptr<Type> type;
  void accept(Visitor &visitor) override;
};

class Program {
public:
  Program(std::vector<std::unique_ptr<Definition>> definitions)
      : defs(std::move(definitions)) {}
  void accept(Visitor &visitor);
  const std::vector<std::unique_ptr<Definition>> &definitions() const {
    return defs;
  }

private:
  std::vector<std::unique_ptr<Definition>> defs;
};

class Visitor {
public:
  virtual ~Visitor() = default;
  virtual void visit(LetBinding &stmt) = 0;
  virtual void visit(Param &param) = 0;
  virtual void visit(PermutationExpr &expr) = 0;
  virtual void visit(IdentifierExpr &expr) = 0;
  virtual void visit(ScalarLiteralExpr &expr) = 0;
  virtual void visit(TensorType &type) = 0;
  virtual void visit(ScalarType &type) = 0;
  virtual void visit(CallExpr &expr) = 0;
  virtual void visit(TensorExpr &expr) = 0;
  virtual void visit(FunctionDef &def) = 0;
  virtual void visit(ConstDef &def) = 0;
  virtual void visit(Program &prog) = 0;
};