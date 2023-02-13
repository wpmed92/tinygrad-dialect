// RUN: hello-opt %s | FileCheck %s

// CHECK: define void @main()
func.func @main() {
    %0 = "tinygrad.constant"() {value = dense<[[3.140000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    "tinygrad.print"(%0) : (tensor<2x3xf64>) -> ()
    %1 = "tinygrad.constant"() { value = dense<[[1.12]]> : tensor<1x1xf64> } : () -> tensor<1x1xf64>
    "tinygrad.print"(%1) : (tensor<1x1xf64>) -> ()
    %2 = "tinygrad.add"(%0, %0) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    "tinygrad.print"(%2) : (tensor<2x3xf64>) -> ()
    %3 = "tinygrad.sub"(%0, %0) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    "tinygrad.print"(%3) : (tensor<2x3xf64>) -> ()
    %4 = "tinygrad.constant"() {value = dense<[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %5 = "tinygrad.pow"(%0, %4) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    "tinygrad.print"(%5) : (tensor<2x3xf64>) -> ()
    %6 = "tinygrad.cmpeq"(%0, %0) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xi1>
    "tinygrad.print"(%6) : (tensor<2x3xi1>) -> ()
    %7 = "tinygrad.exp"(%4) : (tensor<2x3xf64>) -> tensor<2x3xf64>
    "tinygrad.print"(%7) : (tensor<2x3xf64>) -> ()
    %8 = "tinygrad.log"(%7) : (tensor<2x3xf64>) -> tensor<2x3xf64>
    "tinygrad.print"(%8) : (tensor<2x3xf64>) -> ()
    %9 = "tinygrad.constant"() {value = dense<[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %10 = "tinygrad.mul"(%0, %4) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    %11 = "tinygrad.div"(%10, %4) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    %12 = "tinygrad.neg"(%9) : (tensor<2x3xf64>) -> tensor<2x3xf64>
    %13 = "tinygrad.gt0"(%12) : (tensor<2x3xf64>) -> tensor<2x3xi1>
    %14 = "tinygrad.relu"(%12) :  (tensor<2x3xf64>) -> tensor<2x3xf64>
    "tinygrad.print"(%14) : (tensor<2x3xf64>) -> ()
    %15 = "tinygrad.reshape"(%14) { shape = dense<[3, 2]> : tensor<2xi32>} : (tensor<2x3xf64>) -> memref<3x2xf64>
    "tinygrad.print"(%15) : (memref<3x2xf64>) -> ()
    return
}
