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

#ifndef MSDA_BATCH_SIZE
#define MSDA_BATCH_SIZE     6
#endif
#ifndef MSDA_NUM_QUERIES
#define MSDA_NUM_QUERIES    9662
#endif
#ifndef MSDA_NUM_QUERIES_RUN
#define MSDA_NUM_QUERIES_RUN MSDA_NUM_QUERIES
#endif
#ifndef MSDA_NUM_HEADS
#define MSDA_NUM_HEADS      8
#endif
#ifndef MSDA_CHANNELS
#define MSDA_CHANNELS       32
#endif
#ifndef MSDA_NUM_LEVELS
#define MSDA_NUM_LEVELS     4
#endif
#ifndef MSDA_NUM_POINTS
#define MSDA_NUM_POINTS     8
#endif
#ifndef MSDA_SPATIAL_SIZE
#define MSDA_SPATIAL_SIZE   21250
#endif

#ifndef USE_NEAREST_NEIGHBOR
#define USE_NEAREST_NEIGHBOR 0
#endif

#ifndef USE_16BIT_DATA
#define USE_16BIT_DATA 0
#endif

#ifndef USE_PINGPONG_BUFFER
#define USE_PINGPONG_BUFFER 0
#endif

#ifdef USE_16BIT_DATA
#include <ap_fixed.h>
// Paper chooses 13 total bits; Q6.7 is this adapted reproduction choice.
typedef ap_fixed<13, 6, AP_RND, AP_SAT> data_t;
typedef ap_fixed<24, 12, AP_RND, AP_SAT> accum_t;
#else
typedef float   data_t;
typedef float   accum_t;
#endif
typedef int32_t index_t;

static const int CONST_BATCH_SIZE  = MSDA_BATCH_SIZE;
static const int CONST_NUM_QUERIES = MSDA_NUM_QUERIES;
static const int CONST_NUM_QUERIES_RUN = MSDA_NUM_QUERIES_RUN;
static const int CONST_NUM_HEADS   = MSDA_NUM_HEADS;
static const int CONST_CHANNELS    = MSDA_CHANNELS;
static const int CONST_NUM_LEVELS  = MSDA_NUM_LEVELS;
static const int CONST_NUM_POINTS  = MSDA_NUM_POINTS;
static const int CONST_SPATIAL_SIZE = MSDA_SPATIAL_SIZE;

static const index_t CONST_SPATIAL_SHAPES[CONST_NUM_LEVELS * 2] = {
    80, 200,
    40, 100,
    20,  50,
    10,  25
};

static const index_t CONST_LEVEL_START[CONST_NUM_LEVELS] = {
    0, 16000, 20000, 21000
};
 
 
 
 /**
  * @brief Main MSDA function
  */
 void msda_top(
     const ap_uint<512>* value_data,
     const ap_uint<512>* sampling_locations_data,
     const ap_uint<512>* attention_weights_data,
     ap_uint<512>* output_data
 );
 
 /**
  * @brief Bilinear interpolation function
  */
 data_t bilinear_interpolation(
     const data_t v1, const data_t v2, const data_t v3, const data_t v4,
     const data_t lh, const data_t lw
 );
 
#endif // MSDA_TOP_H