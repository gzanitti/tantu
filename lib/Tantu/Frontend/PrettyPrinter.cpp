
#include "Tantu/Frontend/PrettyPrinter.h"
#include "Tantu/Frontend/AST.h"
#include <iostream>

void PrettyPrinter::visit(Program &prog) {
  std::cout << "Program(" << std::endl;
  for (const auto &def : prog.definitions()) {
    def->accept(*this);
  }
  std::cout << ")" << std::endl;
}

void PrettyPrinter::visit(ConstDef &def) {
  std::cout << "ConstDef(" << def.name << " " << def.value << " : ";
  def.type->accept(*this);
  std::cout << ")" << std::endl;
}

void PrettyPrinter::visit(FunctionDef &def) {
  std::cout << "FunctionDef(" << def.name << " " << std::endl;
  std::cout << "  Params: ";
  for (auto &param : def.params) {
    param.accept(*this);
  }
  std::cout << "  Body: ";
  for (auto &stmt : def.body) {
    stmt->accept(*this);
  }
  std::cout << "  Return: ";
  def.returnExpr->accept(*this);
  std::cout << " : ";
  def.returnType->accept(*this);
  std::cout << ")" << std::endl;
}

void PrettyPrinter::visit(Param &param) {
  std::cout << "Param(";
  param.identifier.accept(*this);
  std::cout << " : ";
  param.type->accept(*this);
  std::cout << ")" << std::endl;
}

void PrettyPrinter::visit(LetBinding &stmt) {
  std::cout << "LetBinding(";
  stmt.identifier.accept(*this);
  std::cout << " = ";
  stmt.expr->accept(*this);
  std::cout << ");" << std::endl;
}

void PrettyPrinter::visit(IdentifierExpr &expr) {
  std::cout << expr.name;
}

void PrettyPrinter::visit(ScalarLiteralExpr &expr) {
  std::cout << expr.value;
}

void PrettyPrinter::visit(ScalarType &type) {
  std::cout << (type.kind == Integer ? "i32" : "f32");
}

void PrettyPrinter::visit(TensorType &type) {
  std::cout << "Tensor<";
  for (size_t i = 0; i < type.shape.size(); ++i) {
    if (i > 0)
      std::cout << "x";
    std::cout << type.shape[i];
  }
  std::cout << ">";
}

static const char *builtinOpName(BuiltinOp op) {
  switch (op) {
  case Add:        return "add";
  case Sub:        return "sub";
  case Mul:        return "mul";
  case Div:        return "div";
  case Max:        return "max";
  case Exp:        return "exp";
  case Neg:        return "neg";
  case AddScalar:  return "add_scalar";
  case SubScalar:  return "sub_scalar";
  case MulScalar:  return "mul_scalar";
  case DivScalar:  return "div_scalar";
  case MaxScalar:  return "max_scalar";
  case Sum:        return "sum";
  case MaxReduce:  return "max_reduce";
  case Size:       return "size";
  case Cast:       return "cast";
  case Transpose:  return "transpose";
  case Matmul:     return "matmul";
  case Print:      return "print";
  }
  return "unknown";
}

void PrettyPrinter::visit(CallExpr &expr) {
  if (expr.builtinOp)
    std::cout << builtinOpName(*expr.builtinOp) << "(";
  else {
    expr.callee.accept(*this);
    std::cout << "(";
  }
  for (size_t i = 0; i < expr.args.size(); ++i) {
    if (i > 0)
      std::cout << ", ";
    expr.args[i]->accept(*this);
  }
  std::cout << ")";
}

void PrettyPrinter::visit(TensorExpr &expr) {
  std::cout << "Tensor<";
  for (size_t i = 0; i < expr.shape.size(); ++i) {
    if (i > 0)
      std::cout << "x";
    std::cout << expr.shape[i];
  }
  std::cout << ">[";
  for (size_t i = 0; i < expr.values.size(); ++i) {
    if (i > 0)
      std::cout << ", ";
    expr.values[i]->accept(*this);
  }
  std::cout << "]";
}
