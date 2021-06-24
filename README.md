# CNN Accelerator Kernel in Vivado HLS

## Environments

* Vivado HLS 2018.3
* Vivado 2018.3

## How to use ? 

Pick the kernel you want to synthesis and 

1. Import all the \*.cpp files into your HLS working space.
2. Set function *cnn* as the top function.
3. Click *synthesis and export RTL* and wait for a long time ...

## Some notes

* These projects are tuned for PYNQ-Z2, which will incur sub-optimal issues when be deployed on other boards. Please tune the parameters for better performance.
* The interfaces of these kernels are set as AXI-LITE.
* We use fixed-shape input and output activations in this project, you can find the parameters defined as macros in head files.
* It seems that the code is not clean, as there are mass of redundancy and repitition. But it do help for HLS tools to do synthesis much more faster.
