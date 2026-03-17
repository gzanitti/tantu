#include "Tantu/Frontend/AST.h"

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
