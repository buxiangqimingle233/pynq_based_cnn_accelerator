# CNN Accelerator Kernel in Vivado HLS

## Circumstance

* Vivado HLS 2018.3
* Vivado 2018.3

## How to Use

For the kernel to use:

1. import all the \*.cpp files into the working space of HLS 
2. set function *cnn* as the top function
3. synthesis and export RTL

## Notice

* The projects are designed and optimized toward PYNQ-Z2, the parameters should be fine tuned if implemented to other board to get better performance.
* The interfaces of kernels are set as AXI-LITE.
* Size of input & output features are fixed, which are defined in macro in head files.

* The codes are redundant in order to explicitly indicate the hardware architecture.
