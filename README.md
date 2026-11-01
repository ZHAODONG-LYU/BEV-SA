# BEV-SA

This repository provides hardware source code for the MSDA sampling-and-aggregation kernel used in BEV-SA.

## Directory Structure

- `original/`: original source version of the MSDA sampling-and-aggregation kernel.
- `optimized/`: optimized BEV-SA sampling-and-aggregation kernel.
- `template/`: configurable-precision source template for the MSDA sampling-and-aggregation kernel.
- `reim_comp/`: kernels reimplemented for comparison with BEV-SA.

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

## Optimized Example

The `optimized/` directory contains an optimized source example with query-grouped execution.

This example uses:

- INT8 feature values with higher-precision SL, AW, and accumulation data paths;
- 4 query groups;
- 4 feature levels;
- 8 sampling points per level;
- nearest-neighbor sampling.

## Configurable-Precision Template

The `template/` directory provides a configurable-precision source template.

Feature-value, SL, AW, sampling, accumulation widths, and the number of feature values per feature pack are defined in `template/msda_top.hpp`.

This template does not assume that lower feature precision must use the full feature bus width.

Alternative precision settings require matching packed inputs and experimental validation before use. Directly changing precision parameters without validation may introduce large numerical errors.

## Scope

This repository provides source files for the hardware kernel versions included here.

Testbench code, input data, datasets, training code, the full BEVFormer model, and the DGP/CQP preprocessing flow are outside the scope of this release.
