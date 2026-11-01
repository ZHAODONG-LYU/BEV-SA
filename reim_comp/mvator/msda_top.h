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

#ifdef USE_16BIT_DATA
#include <ap_fixed.h>
typedef ap_fixed<16, 8, AP_RND, AP_SAT> data_t;
// Experimental Q8.0 feature format, scale 1; not a paper-specified Q format.
typedef ap_fixed<8, 8, AP_RND, AP_SAT> feature_t;
typedef ap_fixed<24, 12, AP_RND, AP_SAT> accum_t;
typedef ap_fixed<16, 8, AP_RND, AP_SAT> location_t;
typedef ap_fixed<24, 12, AP_RND, AP_SAT> pixel_coord_t;
#define AXI_VEC_SIZE 32
#else
typedef float   data_t;
typedef float   accum_t;
#define AXI_VEC_SIZE 16
#endif

#include <hls_vector.h>
typedef hls::vector<data_t, AXI_VEC_SIZE> axi_word_t;
typedef hls::vector<feature_t, 32> feature_word_t;

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
    56, 200, // Paper level 1 after removing its upper 24/80 rows (TV=0.3).
    40, 100,
    20,  50,
    10,  25
};

static const index_t CONST_LEVEL_START[CONST_NUM_LEVELS] = {
    0, 11200, 15200, 16200
};

#ifdef USE_4BANK
static const int CONST_BANK_SIZE[CONST_NUM_LEVELS][4] = {
    { 2800, 2800, 2800, 2800 },   // paper level 1 cropped to H=56,W=200
    { 1000, 1000, 1000, 1000 },   // paper level 2: H=40,W=100
    {  250,  250,  250,  250 },   // paper level 3: H=20,W=50
    {   65,   60,   65,   60 }    // paper level 4: H=10,W=25
};
static const int CONST_BANK_LEVEL_START[4][CONST_NUM_LEVELS + 1] = {
    { 0, 2800, 3800, 4050, 4115 },
    { 0, 2800, 3800, 4050, 4110 },
    { 0, 2800, 3800, 4050, 4115 },
    { 0, 2800, 3800, 4050, 4110 }
};
#endif
 
 
#ifdef USE_4BANK
 /**
  * @brief Main MSDA function (4-bank value ports)
  */
 void msda_top(
     const feature_word_t* value_bank0,
     const feature_word_t* value_bank1,
     const feature_word_t* value_bank2,
     const feature_word_t* value_bank3,
     const axi_word_t* sampling_locations_data,
     const axi_word_t* attention_weights_data,
     axi_word_t* output_data,
     int valid_count_0,
     int valid_count_1,
     int valid_count_2,
     int valid_count_3,
     int valid_count_4,
     int valid_count_5
 );
#else
 /**
  * @brief Main MSDA function (single value port)
  */
 void msda_top(
     const axi_word_t* value_data,
     const axi_word_t* sampling_locations_data,
     const axi_word_t* attention_weights_data,
     axi_word_t* output_data
 );
#endif
 
 /**
  * @brief Bilinear interpolation function
  */
 data_t bilinear_interpolation(
     const data_t v1, const data_t v2, const data_t v3, const data_t v4,
     const data_t lh, const data_t lw
 );
 
#endif // MSDA_TOP_H
