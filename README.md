# BEV-SA

This repository provides hardware source code for the MSDA sampling-and-aggregation kernel used in BEV-SA.

## Directory Structure

- `original/`: original source version of the MSDA sampling-and-aggregation kernel.
- `optimized/`: optimized INT16 source version of the MSDA sampling-and-aggregation kernel.

## Original Version

The `original/` directory contains:

- `msda_top.hpp`: kernel interface, fixed MSDA parameters, data types, and constant feature-level metadata.
- `msda_top.cpp`: single-kernel implementation of MSDA sampling and aggregation.

This version uses:

- floating-point feature values, sampling locations, attention weights, and output values;
- 4 feature levels;
- 8 sampling points per level;
- bilinear interpolation;
- a direct loop-based implementation without query grouping or feature-value quantization.

## Optimized Version

The `optimized/` directory contains an INT16 optimized source version with query-grouped execution.

This version uses:

- INT16 feature values with higher-precision SL, AW, and accumulation data paths;
- 4 query groups;
- 4 feature levels;
- 8 sampling points per level;
- nearest-neighbor sampling.

The paper's deployed 8-bit feature-value configuration will be added separately.

## Scope

This repository provides source files for the hardware kernel versions included here.

Testbench code, input data, datasets, training code, the full BEVFormer model, and the DGP/CQP preprocessing flow are outside the scope of this release.
