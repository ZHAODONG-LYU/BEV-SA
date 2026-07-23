/**
 * @file msda_top.h
 * @brief MSDA HLS Implementation Header
 */

#ifndef MSDA_TOP_H
#define MSDA_TOP_H

// HLS headers
#include <ap_int.h>
#include <hls_stream.h>
#include <cstdint>

// ===== MSDA Configuration Parameters =====
#define MSDA_BATCH_SIZE     6
#define MSDA_NUM_QUERIES    9662
#define MSDA_NUM_HEADS      8
#define MSDA_CHANNELS       32
#define MSDA_NUM_LEVELS     4
#define MSDA_NUM_POINTS     8
#define MSDA_SPATIAL_SIZE   21250

// ===== Data Type Definition =====
using data_t = float;
using index_t = int32_t;

// ===== HLS Constants Definition =====
const int CONST_BATCH_SIZE = MSDA_BATCH_SIZE;
const int CONST_NUM_QUERIES = MSDA_NUM_QUERIES;
const int CONST_NUM_HEADS = MSDA_NUM_HEADS;
const int CONST_CHANNELS = MSDA_CHANNELS;
const int CONST_NUM_LEVELS = MSDA_NUM_LEVELS;
const int CONST_NUM_POINTS = MSDA_NUM_POINTS;
const int CONST_SPATIAL_SIZE = MSDA_SPATIAL_SIZE;

// ===== Fixed Small Arrays (from constant shapes/start indices in dataset) =====
// Like [H0, W0, H1, W1, H2, W2, H3, W3]
static const index_t CONST_SPATIAL_SHAPES[CONST_NUM_LEVELS * 2] = {
    80, 200,
    40, 100,
    20,  50,
    10,  25
};

// Start index for each level in value
static const index_t CONST_LEVEL_START[CONST_NUM_LEVELS] = {
    0, 16000, 20000, 21000
};

// ===== Data Structure Definition =====
// (Now using single function version, no complex data structures needed)

// ===== Function Declarations =====

/**
 * @brief Main MSDA function
 */
void msda_top(
    const data_t* value_data,
    const data_t* sampling_locations_data,
    const data_t* attention_weights_data,
    data_t* output_data
);

/**
 * @brief Bilinear interpolation function
 */
data_t bilinear_interpolation(
    const data_t v1, const data_t v2, const data_t v3, const data_t v4,
    const data_t lh, const data_t lw
);

#endif // MSDA_TOP_H