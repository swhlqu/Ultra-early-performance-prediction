# Ultra-early-performance-prediction
Data and code for the paper "Ultra-early prediction of lithium-ion battery performance using mechanism and data-driven fusion model"
This version in python is available on https://github.com/swhlqu/Ultra-early-performance-prediction. 
The simulated dataset using COMSOL is available on COMSOL_DATASET.
Note: The P2D model using COMSOL is available from the corresponding authors upon request.
Overview
DNN and Attention is a propoed general framework which can predict the charging curves over full life cycles of NCM523 and NCM811 batteries in the case of only knowing the charge protocols.

System Requirements
Hardware requirements
DNN and Attention requires only a standard computer with enough RAM to support the in-memory operations.

Software requirements
COMSOL Multiphysics
pytorch
The python package is supported for windows. The package has been tested on the following systems:

windows: windows11 
Python Dependencies
DNN and Attention mainly depends on the Python scientific stack. The data processing and the development of all proposed methods are implemented in Python 3.9 with Pytorch 1.11. The computation is executed based on an RTX 3090 graphics processing unit.

torch
numpy
pandas
time
warnings
os
math
pathlib 

The NCM523 and NCM811 fold includes the actual charging data.

