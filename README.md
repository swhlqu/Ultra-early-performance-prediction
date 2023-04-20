# Ultra-early-performance-prediction
Data and code for the paper "Ultra-early prediction of lithium-ion battery performance using mechanism and data-driven fusion model"

This version in python is available on https://github.com/swhlqu/Ultra-early-performance-prediction. 

## Dataset overview

The simulated dataset using COMSOL is available on COMSOL_DATASET.
It consists the simulated data of NCM523 under charging rates of 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0 C at 25 ℃, 0.5 C at 35 ℃, 0.6, 0.7, and 1.0 C at 45 ℃. It also includes the simulated data of the NCM811 battery under charging rates of 0.5, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0 C at 25 ℃, 0.7, 0.8, 0.9, and 1.0 C at 35 ℃, and 0.8, 0.9, and 1.0 C at 45 ℃. 

Note: The P2D model using COMSOL is available from the corresponding authors upon request.

The actual charging curves over the full life cycle of the NCM523 and NCM811 batteries is available on NCM523 and NCM811 folds, which 25_0.5CC_0.5CD.xlsx refers the charging data is testing under 25 ℃ with 0.5 C charging and discharging rates.

The actual ISC test dataset of the NCM523 and NCM811 battery is available on NCM523-ISC and NCM811-ISC folds, which BD refers to the normal test without paralleled ISC resistance, and CS refers to the ISC test with paralleled ISC resistance. The dataste in BD fold is used to correct the simulated results, and that in CS fold is utilized to detect ISC.

## Model Overview
DNN and Attention are the propoed general framework which can predict the charging curves over full life cycles of NCM523 and NCM811 batteries in the case of only knowing the charge protocols.

### System Requirements
### Hardware requirements
DNN and Attention requires only a standard computer with enough RAM to support the in-memory operations.

### Software requirements
COMSOL Multiphysics
pytorch
The python package is supported for windows. The package has been tested on the following systems:

windows: windows11 

### Python Dependencies
DNN and Attention mainly depends on the Python scientific stack. The data processing and the development of all proposed methods are implemented in Python 3.9 with Pytorch 1.11. The computation is executed based on an RTX 3090 graphics processing unit.

torch
numpy
pandas
time
warnings
os
math
pathlib 

### Training, Validation, and Test set generation
It can be found in the main text of "Ultra-early prediction of lithium-ion battery performance using mechanism and data-driven fusion model"

## Contact

If you have any questions, you can reach the author at the following e-mail: cydu@hit.edu.cn
