#include "Tantu/Frontend/AST.h"
#include <sstream>

const char *builtinOpName(BuiltinOp op) {
  if (op == Add)
    return "add";
  if (op == Sub)
    return "sub";
  if (op == Mul)
    return "mul";
  if (op == Div)
    return "div";
  if (op == Max)
    return "max";
  if (op == Exp)
    return "exp";
  if (op == Neg)
    return "neg";
  if (op == AddScalar)
    return "add_scalar";
  if (op == SubScalar)
    return "sub_scalar";
  if (op == MulScalar)
    return "mul_scalar";
  if (op == DivScalar)
    return "div_scalar";
  if (op == MaxScalar)
    return "max_scalar";
  if (op == Sum)
    return "sum";
  if (op == MaxReduce)
    return "max_reduce";
  if (op == Size)
    return "size";
  if (op == Cast)
    return "cast";
  if (op == Transpose)
    return "transpose";
  if (op == Matmul)
    return "matmul";
  if (op == Print)
    return "print";
  return "unknown";
}

std::optional<BuiltinOp> getBuiltinOp(std::string_view name) {
  if (name == "add")
    return Add;
  if (name == "sub")
    return Sub;
  if (name == "mul")
    return Mul;
  if (name == "div")
    return Div;
  if (name == "max")
    return Max;
  if (name == "exp")
    return Exp;
  if (name == "neg")
    return Neg;
  if (name == "add_scalar")
    return AddScalar;
  if (name == "sub_scalar")
    return SubScalar;
  if (name == "mul_scalar")
    return MulScalar;
  if (name == "div_scalar")
    return DivScalar;
  if (name == "max_scalar")
    return MaxScalar;
  if (name == "sum")
    return Sum;
  if (name == "max_reduce")
    return MaxReduce;
  if (name == "size")
    return Size;
  if (name == "cast")
    return Cast;
  if (name == "transpose")
    return Transpose;
  if (name == "matmul")
    return Matmul;
  if (name == "print")
    return Print;
  return std::nullopt;
}

void Program::accept(Visitor &visitor) { visitor.visit(*this); }
void FunctionDef::accept(Visitor &visitor) { visitor.visit(*this); }
void ConstDef::accept(Visitor &visitor) { visitor.visit(*this); }
void TensorExpr::accept(Visitor &visitor) { visitor.visit(*this); }
void CallExpr::accept(Visitor &visitor) { visitor.visit(*this); }
void LetBinding::accept(Visitor &visitor) { visitor.visit(*this); }
void Param::accept(Visitor &visitor) { visitor.visit(*this); }
void IdentifierExpr::accept(Visitor &visitor) { visitor.visit(*this); }
void ScalarLiteralExpr::accept(Visitor &visitor) { visitor.visit(*this); }
void TensorType::accept(Visitor &visitor) { visitor.visit(*this); }
void ScalarType::accept(Visitor &visitor) { visitor.visit(*this); }
void PermutationExpr::accept(Visitor &visitor) { visitor.visit(*this); }

bool ScalarType::operator==(const ScalarType &other) const {
  return kind == other.kind;
}

bool TensorType::operator==(const TensorType &other) const {
  return shape == other.shape;
}

std::string ScalarType::toString() const {
  return kind == ScalarKind::Integer ? "i32" : "f32";
}

std::string TensorType::toString() const {
  std::ostringstream oss;
  oss << "tensor<";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i)
      oss << "x";
    oss << shape[i];
  }
  oss << ">";
  return oss.str();
}