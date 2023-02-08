add_mlir_dialect_library(MLIRTinyGrad
	TinyGradDialect.cpp
	TinyGradOps.cpp
	LowerToAffine.cpp
	LowerToLLVM.cpp

	ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/include/TinyGrad

	DEPENDS
	MLIRTinyGradOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
)
