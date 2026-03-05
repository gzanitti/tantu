// Registration functions for Tantu bufferization external models.
// Call these before running --one-shot-bufferize in tantu-opt.

#ifndef TANTU_BUFFERIZATION_REGISTRATIONS_H
#define TANTU_BUFFERIZATION_REGISTRATIONS_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

void registerTantuNegBufferizationModel(mlir::DialectRegistry &registry);
void registerTantuAddBufferizationModel(mlir::DialectRegistry &registry);

#endif // TANTU_BUFFERIZATION_REGISTRATIONS_H