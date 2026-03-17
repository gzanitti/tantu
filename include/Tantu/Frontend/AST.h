#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class Visitor;

class Type {
public:
  virtual void accept(Visitor &visitor) = 0;
  virtual ~Type() = default;
};

enum ScalarKind { Integer, Float };

class ScalarType : public Type {
public:
  ScalarType(ScalarKind kind) : kind(kind) {}
  ScalarKind kind;
  void accept(Visitor &visitor) override;
};

class TensorType : public Type {
public:
  TensorType(std::vector<int> shape) : shape(shape) {}
  std::vector<int> shape;
  void accept(Visitor &visitor) override;
};

class Statement {
public:
  virtual void accept(Visitor &visitor) = 0;
  virtual ~Statement() = default;
};

class Expression {
public:
  virtual void accept(Visitor &visitor) = 0;
  virtual ~Expression() = default;
};


class ScalarLiteralExpr : public Expression {
public:
  ScalarLiteralExpr(double value) : value(value) {}
  double value;
  void accept(Visitor &visitor) override;
};

class IdentifierExpr : public Expression {
public:
  IdentifierExpr(std::string name) : name(name) {}
  const std::string &getName() const { return name; }
  std::string name;
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

class CallExpr : public Expression {
public:
  CallExpr(IdentifierExpr callee, std::vector<std::unique_ptr<Expression>> args,
           std::optional<BuiltinOp> builtinOp)
      : callee(callee), args(std::move(args)), builtinOp(builtinOp) {}
  IdentifierExpr callee;
  std::vector<std::unique_ptr<Expression>> args;
  std::optional<BuiltinOp> builtinOp;
  void accept(Visitor &visitor) override;
};

class TensorExpr : public Expression {
public:
  TensorExpr(std::vector<int> shape,
             std::vector<std::unique_ptr<ScalarLiteralExpr>> values)
      : shape(shape), values(std::move(values)) {}
  std::vector<int> shape;
  std::vector<std::unique_ptr<ScalarLiteralExpr>> values;
  void accept(Visitor &visitor) override;
};

class Definition {
public:
  Definition(std::string name) : name(name) {}
  const std::string &getName() const { return name; }
  virtual void accept(Visitor &visitor) = 0;
  std::string name;
  virtual ~Definition() = default;
};

class FunctionDef : public Definition {
public:
  FunctionDef(std::string name, std::vector<Param> params,
              std::vector<std::unique_ptr<Statement>> body,
              std::unique_ptr<Expression> returnExpr,
              std::unique_ptr<Type> returnType)
      : Definition(name), params(std::move(params)), body(std::move(body)),
        returnExpr(std::move(returnExpr)), returnType(std::move(returnType)) {}
  std::vector<Param> params;
  std::vector<std::unique_ptr<Statement>> body;
  std::unique_ptr<Expression> returnExpr;
  std::unique_ptr<Type> returnType;
  void accept(Visitor &visitor) override;
};

class ConstDef : public Definition {
public:
  ConstDef(std::string name, int value, std::unique_ptr<Type> type)
      : Definition(name), value(value), type(std::move(type)) {}
  int value;
  std::unique_ptr<Type> type;
  void accept(Visitor &visitor) override;
};

class Program {
public:
  Program(std::vector<std::unique_ptr<Definition>> definitions)
      : defs(std::move(definitions)) {}
  void accept(Visitor &visitor);
  std::vector<std::unique_ptr<Definition>> definitions() {
    return std::move(defs);
  }

private:
  std::vector<std::unique_ptr<Definition>> defs;
};

class Visitor {
public:
  virtual ~Visitor() = default;
  virtual void visit(LetBinding &stmt) = 0;
  virtual void visit(Param &param) = 0;
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