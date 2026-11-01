/**
 * @file msda_top.h
 * @brief MSDA HLS Implementation Header
 */

#ifndef MSDA_TOP_H
#define MSDA_TOP_H

// HLS headers
#include <cstdint>
#include "ap_fixed.h"

#define MSDA_BATCH_SIZE     6
#ifndef MSDA_NUM_QUERIES
#define MSDA_NUM_QUERIES    5000
#endif
#define MSDA_NUM_HEADS      8
#define MSDA_CHANNELS       32
#define MSDA_NUM_LEVELS     4
#define MSDA_NUM_POINTS     8
// Every INT8 level and channel starts on a 64-byte boundary; logical sizes are unchanged.
#define MSDA_SPATIAL_SIZE   21312

using data_t = float;           // for testbench compatibility (sampling_loc, attn_weight, output)
using ddr_float_t = float;      // same as data_t, used in HLS implementation
using value_t = int8_t;         // quantized feature map (int8)
using index_t = int32_t;

using coord_t = ap_fixed<27, 11, AP_TRN, AP_WRAP>;
using weight_t = ap_fixed<27, 8, AP_TRN, AP_WRAP>;
using interp_t = ap_fixed<18, 1, AP_TRN, AP_WRAP>;
using sample_t = ap_fixed<48, 16, AP_TRN, AP_WRAP>;
using accum_t = ap_fixed<48, 15, AP_TRN, AP_WRAP>;

using output_t = accum_t;  // ap_fixed<48, 15>

const int CONST_BATCH_SIZE = MSDA_BATCH_SIZE;
const int CONST_NUM_QUERIES = MSDA_NUM_QUERIES;
const int CONST_NUM_HEADS = MSDA_NUM_HEADS;
const int CONST_CHANNELS = MSDA_CHANNELS;
const int CONST_NUM_LEVELS = MSDA_NUM_LEVELS;
const int CONST_NUM_POINTS = MSDA_NUM_POINTS;
const int CONST_SPATIAL_SIZE = MSDA_SPATIAL_SIZE;

static const index_t CONST_SPATIAL_SHAPES[CONST_NUM_LEVELS * 2] = {
    80, 200,
    40, 100,
    20,  50,
    10,  25
};

// HBM layout: L0=16000, L1 padded to4032, L2=1024, L3=256; stride21312 bytes.
static constexpr index_t CONST_LEVEL_START[CONST_NUM_LEVELS] = {
    0, 16000, 20032, 21056
};



/**
 * @brief Main MSDA function
 * Note: sampling_locations and attention_weights are pre-converted to ap_fixed and packed
 * v1.0.5: 8 SL + 8 AW interfaces + 2 output interfaces for 2-query parallel processing
 * v1.1.0: AW interfaces reduced from 8 to 4 by packing 2 levels into one 512-bit transaction
 * v1.2.0: 4-query parallel, 16 SL + 8 AW interfaces + 4 output interfaces
 */
extern "C" void msda_top(
    const value_t* value_data,                    // int8_t quantized feature map (port A)
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
    ap_uint<512>* output_data_0,
    // Output: group 1 (q % 4 == 1)
    ap_uint<512>* output_data_1,
    // Output: group 2 (q % 4 == 2)
    ap_uint<512>* output_data_2,
    // Output: group 3 (q % 4 == 3)
    ap_uint<512>* output_data_3
);

/**
 * @brief Bilinear interpolation function (ap_fixed version)
 */
sample_t bilinear_interpolation(
    const value_t v1, const value_t v2, const value_t v3, const value_t v4,
    const interp_t lh, const interp_t lw
);

#endif // MSDA_TOP_H