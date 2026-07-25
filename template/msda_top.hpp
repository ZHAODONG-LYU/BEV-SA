/**
 * @file msda_top.h
 * @brief MSDA HLS Implementation Header
 */

#ifndef MSDA_TOP_H
#define MSDA_TOP_H

// Hardware headers
#include <cstdint>
#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 4096
#endif
#include "ap_int.h"
#include "ap_fixed.h"

// ===== MSDA configuration parameters =====
#define MSDA_BATCH_SIZE     6
#ifndef MSDA_NUM_QUERIES
#define MSDA_NUM_QUERIES    5000
#endif
#define MSDA_NUM_HEADS      8
#define MSDA_CHANNELS       32
#define MSDA_NUM_LEVELS     4
#define MSDA_NUM_POINTS     8
// Padded to 32 alignment (original 21250 -> 21280) to enable aligned 512b loads
#define MSDA_SPATIAL_SIZE   21280

// ===== Precision configuration =====
// These widths are compile-time configuration parameters. Alternative feature,
// SL, AW, sample, or accumulation widths require matching input packing and
// numerical/accuracy validation before use.
#ifndef MSDA_FEATURE_VALUE_BITS
#define MSDA_FEATURE_VALUE_BITS 16
#endif
#ifndef MSDA_SL_TOTAL_BITS
#define MSDA_SL_TOTAL_BITS 27
#endif
#ifndef MSDA_SL_INT_BITS
#define MSDA_SL_INT_BITS 11
#endif
#ifndef MSDA_AW_TOTAL_BITS
#define MSDA_AW_TOTAL_BITS 23
#endif
#ifndef MSDA_AW_INT_BITS
#define MSDA_AW_INT_BITS 1
#endif
#ifndef MSDA_SAMPLE_TOTAL_BITS
#define MSDA_SAMPLE_TOTAL_BITS 48
#endif
#ifndef MSDA_SAMPLE_INT_BITS
#define MSDA_SAMPLE_INT_BITS 16
#endif
#ifndef MSDA_ACCUM_TOTAL_BITS
#define MSDA_ACCUM_TOTAL_BITS 48
#endif
#ifndef MSDA_ACCUM_INT_BITS
#define MSDA_ACCUM_INT_BITS 15
#endif

#define MSDA_FEATURE_BUS_BITS 512
#define MSDA_METADATA_SLOT_BITS 32
#ifndef MSDA_FEATURE_VALUES_PER_PACK
#define MSDA_FEATURE_VALUES_PER_PACK (MSDA_FEATURE_BUS_BITS / MSDA_FEATURE_VALUE_BITS)
#endif

static_assert(MSDA_FEATURE_VALUES_PER_PACK * MSDA_FEATURE_VALUE_BITS <= MSDA_FEATURE_BUS_BITS,
              "Feature values per pack must fit in the feature bus width.");
static_assert(MSDA_SL_TOTAL_BITS <= MSDA_METADATA_SLOT_BITS,
              "SL metadata must fit in one metadata slot.");
static_assert(MSDA_AW_TOTAL_BITS <= MSDA_METADATA_SLOT_BITS,
              "AW metadata must fit in one metadata slot.");

// ===== Data type definitions =====
using data_t = float;           // for compatibility with external data preparation
using ddr_float_t = float;      // same as data_t, used in the implementation
using value_t = ap_int<MSDA_FEATURE_VALUE_BITS>;
using index_t = int32_t;
const int CONST_FEATURE_VALUES_PER_PACK = MSDA_FEATURE_VALUES_PER_PACK;

// Internal calculation types. The default settings mirror the INT16 optimized
// source version. Changing these widths changes the numerical behavior and
// must be validated with the corresponding packed inputs and output checks.
using coord_t = ap_fixed<MSDA_SL_TOTAL_BITS, MSDA_SL_INT_BITS, AP_TRN, AP_WRAP>;
using weight_t = ap_fixed<MSDA_AW_TOTAL_BITS, MSDA_AW_INT_BITS, AP_TRN, AP_WRAP>;
using interp_t = ap_fixed<18, 1, AP_TRN, AP_WRAP>;   // interpolation [0,1): 1 integer (only sign)+17 fractional (DSP B input)
using sample_t = ap_fixed<MSDA_SAMPLE_TOTAL_BITS, MSDA_SAMPLE_INT_BITS, AP_TRN, AP_WRAP>;
using accum_t = ap_fixed<MSDA_ACCUM_TOTAL_BITS, MSDA_ACCUM_INT_BITS, AP_TRN, AP_WRAP>;

// Output type.
using output_t = accum_t;

// ===== HLS constant definitions =====
const int CONST_BATCH_SIZE = MSDA_BATCH_SIZE;
const int CONST_NUM_QUERIES = MSDA_NUM_QUERIES;
const int CONST_NUM_HEADS = MSDA_NUM_HEADS;
const int CONST_CHANNELS = MSDA_CHANNELS;
const int CONST_NUM_LEVELS = MSDA_NUM_LEVELS;
const int CONST_NUM_POINTS = MSDA_NUM_POINTS;
const int CONST_SPATIAL_SIZE = MSDA_SPATIAL_SIZE;

// ===== Fixed fractional array (from constant shapes/starting indices in data) =====
// Like [H0, W0, H1, W1, H2, W2, H3, W3]
static const index_t CONST_SPATIAL_SHAPES[CONST_NUM_LEVELS * 2] = {
    80, 200,
    40, 100,
    20,  50,
    10,  25
};

// Starting index for each level in value
// level starting position aligned to 32 (L2 padded to 1024, L3 padded to 256, total SPATIAL=21280)
static const index_t CONST_LEVEL_START[CONST_NUM_LEVELS] = {
    0, 16000, 20000, 21024
};

// ===== Data structure definitions =====
// (now using single function version, no complex data structures)

// ===== Function declarations =====

/**
 * @brief Main MSDA function with configurable feature and metadata precision.
 * Note: sampling_locations and attention_weights are pre-converted to fixed-point and packed.
 * v1.0.5: 8 SL + 8 AW interfaces + 2 output interfaces for 2-query parallel processing
 * v1.1.0: AW interfaces reduced from 8 to 4 by packing 2 levels into one 512-bit transaction
 * v1.2.0: 4-query parallel, 16 SL + 8 AW interfaces + 4 output interfaces
 */
void msda_top(
    const value_t* value_data,                    // quantized feature map (port A)
    const value_t* value_data_1,                  // duplicated m_axi port for value_data (port B)
    // Group 0: q % 4 == 0, Levels 0-3
    const ap_uint<512>* sampling_locations_data_0,
    const ap_uint<512>* sampling_locations_data_1,
    const ap_uint<512>* sampling_locations_data_2,
    const ap_uint<512>* sampling_locations_data_3,
    // Group 1: q % 4 == 1, Levels 0-3
    const ap_uint<512>* sampling_locations_data_4,
    const ap_uint<512>* sampling_locations_data_5,
    const ap_uint<512>* sampling_locations_data_6,
    const ap_uint<512>* sampling_locations_data_7,
    // Group 2: q % 4 == 2, Levels 0-3
    const ap_uint<512>* sampling_locations_data_8,
    const ap_uint<512>* sampling_locations_data_9,
    const ap_uint<512>* sampling_locations_data_10,
    const ap_uint<512>* sampling_locations_data_11,
    // Group 3: q % 4 == 3, Levels 0-3
    const ap_uint<512>* sampling_locations_data_12,
    const ap_uint<512>* sampling_locations_data_13,
    const ap_uint<512>* sampling_locations_data_14,
    const ap_uint<512>* sampling_locations_data_15,
    // Group 0: attention_weights, Levels 0&1, 2&3 combined
    const ap_uint<512>* attention_weights_data_0, // Levels 0 & 1
    const ap_uint<512>* attention_weights_data_1, // Levels 2 & 3
    // Group 1: attention_weights, Levels 0&1, 2&3 combined
    const ap_uint<512>* attention_weights_data_2, // Levels 0 & 1
    const ap_uint<512>* attention_weights_data_3, // Levels 2 & 3
    // Group 2: attention_weights, Levels 0&1, 2&3 combined
    const ap_uint<512>* attention_weights_data_4, // Levels 0 & 1
    const ap_uint<512>* attention_weights_data_5, // Levels 2 & 3
    // Group 3: attention_weights, Levels 0&1, 2&3 combined
    const ap_uint<512>* attention_weights_data_6, // Levels 0 & 1
    const ap_uint<512>* attention_weights_data_7, // Levels 2 & 3
    // Output: group 0 (q % 4 == 0)
    output_t* output_data_0,
    // Output: group 1 (q % 4 == 1)
    output_t* output_data_1,
    // Output: group 2 (q % 4 == 2)
    output_t* output_data_2,
    // Output: group 3 (q % 4 == 3)
    output_t* output_data_3
);

/**
 * @brief Bilinear interpolation function (ap_fixed version)
 */
sample_t bilinear_interpolation(
    const value_t v1, const value_t v2, const value_t v3, const value_t v4,
    const interp_t lh, const interp_t lw
);

#endif // MSDA_TOP_H
