/**
 * @file msda_top.h
 * @brief MSDA HLS Implementation Header
 */

#ifndef MSDA_TOP_H
#define MSDA_TOP_H

// HLS headers
#include <cstdint>
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

// ===== Data type definitions =====
using data_t = float;           // for testbench compatibility (sampling_loc, attn_weight, output)
using ddr_float_t = float;      // same as data_t, used in HLS implementation
using value_t = int16_t;        // quantized feature map (int16)
using index_t = int32_t;

// Internal calculation types (ap_fixed for LUT optimization)
// Width design principles:
//   1. Conservative configuration based on actual data range analysis (prioritize precision)
//      - Theoretical maximum: sample_t needs 131K(18 integer), accum_t needs 1.57M(22 integer)
//      - 27-bit DSP native limit: 18 integer bits + 8 fractional bits, fractional precision is insufficient
//      - **Conclusion**: 32-bit total width is not native to DSP, but can balance integer and fractional bits
//   2. Key trade-off:
//      - 27×18 DSP native: LUT is optimal, but fractional bits are severely insufficient (8-9 bits)
//      - 32-bit solution: DSP concatenation (additional LUT) is required, but fractional bits are sufficient (12-13 bits)
//      - **Choose 32-bit**: prioritize precision, LUT can be optimized through other means
//   3. **Optimized configuration based on actual data analysis**:
//      - Use AP_RND (rounding) to ensure precision, although LUT is higher than AP_TRN but the precision loss is small
//      - AP_WRAP (wrap around) saves LUT for saturation logic
//   4. **Final 48-bit configuration (symbol bit + maximum fractional bit + DSP48E2 optimization)**:
//      - Key understanding: the I in ap_fixed<W, I> includes the sign bit! Range: [-2^(I-1), 2^(I-1)-1]
//      - coord_t: 11 integer + 15 fractional (max≈920, I=11 range [-1024,1023] is sufficient, 15 fractional bits ensure precision)
//      - weight_t: 1 integer + 22 fractional (max=0.37<1, 1 integer bit range [-1,0] can represent [0,1], 22 fractional bits balance precision and LUT) [Optimization: reduce 1 integer bit]
//      - interp_t: 1 integer + 17 fractional (max=1.0, only sign bit, 17 fractional bits!)
//      - sample_t: 16 integer + 31 fractional (test result: must remain 48 bits, reduced to 44 bits is insufficient)
//      - accum_t: 15 integer + 32 fractional (32 accumulations actually max≈406, based on runtime detection: 15 integer bits are sufficient, remain 48 bits)
//      - Design trade-off: coord_t allows a few values (1024-1026) to saturate, sacrificing 2x fractional precision improvement
//      - Quantization mode: AP_SAT prevents overflow wrapping (coord saturates a few values, others do not overflow); AP_RND rounding for precision
//      - Hardware mapping: coord(27)→DSP A, weight/interp(18)→DSP B, accum(48)→DSP P accumulator
using coord_t = ap_fixed<27, 11, AP_TRN, AP_WRAP>;   // loc*spatial maximum≈920: 11 integer (range ±1024)+15 fractional, AP_WRAP reduces LUT (DSP A input)
using weight_t = ap_fixed<23, 1, AP_TRN, AP_WRAP>;   // weights maximum 0.37: 1 integer (range -1 to 0, can represent 0-1)+22 fractional  [Optimization: reduce 1 integer bit, no precision loss]
using interp_t = ap_fixed<18, 1, AP_TRN, AP_WRAP>;   // interpolation [0,1): 1 integer (only sign)+17 fractional (DSP B input)
using sample_t = ap_fixed<48, 16, AP_TRN, AP_WRAP>;  // nearest neighbor interpolation: 16 integer (including sign, INT16 range)+31 fractional  [Test result: must remain 48 bits]
using accum_t = ap_fixed<48, 15, AP_TRN, AP_WRAP>;   // 32 accumulations actually maximum≈406: 15 integer (including sign, range ±16K)+32 fractional  [Based on runtime detection: 15 integer bits are sufficient, remain 48 bits]

// Output type (v1.0.2: output accum_t, converted in testbench)
using output_t = accum_t;  // ap_fixed<48, 15>

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
 * @brief Main MSDA function (Quantized INT16 + ap_fixed Version, 4-query parallel, q%4 grouped)
 * Note: sampling_locations and attention_weights are pre-converted to ap_fixed and packed
 * v1.0.5: 8 SL + 8 AW interfaces + 2 output interfaces for 2-query parallel processing
 * v1.1.0: AW interfaces reduced from 8 to 4 by packing 2 levels into one 512-bit transaction
 * v1.2.0: 4-query parallel, 16 SL + 8 AW interfaces + 4 output interfaces
 */
void msda_top(
    const value_t* value_data,                    // int16_t quantized feature map (port A)
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