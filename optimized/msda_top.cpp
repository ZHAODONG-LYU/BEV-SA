/**
 * @file msda_top.cpp
 * @brief MSDA HLS Implementation - Complete Version
 */

 #include "msda_top.hpp"
 #include <cmath>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdio>      // for limited CSIM debug prints
#include <hls_stream.h>
#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 4096
#endif
#include <ap_int.h>

// ===== bilinear interpolation function (ap_fixed Version) =====
// Input: int16 values, interp_t weights, output: sample_t result
inline sample_t bilinear_interpolation(
    const value_t v1, const value_t v2, const value_t v3, const value_t v4,
    const interp_t lh, const interp_t lw) {
    
    #pragma HLS INLINE
    
    // Rewrite to reduce hardware latency and improve parallelism
    const interp_t ilh = interp_t(1.0) - lh;
    const interp_t ilw = interp_t(1.0) - lw;
    // Calculate the weights of the four corners, explicitly truncate each multiplication result to interp_t (32 bits)
    // interp_t (32 bits) * interp_t (32 bits) will automatically extend to 64 bits
    // Must be immediately truncated to 32 bits to avoid bit width extension propagation
    const interp_t w1 = interp_t(ilh * ilw);
    const interp_t w2 = interp_t(ilh * lw);
    const interp_t w3 = interp_t(lh * ilw);
    const interp_t w4 = interp_t(lh * lw);
    
    // Critical: explicitly truncate each multiplication result to sample_t, avoid HLS automatically extending the bit width to 60+ bits
    // Each intermediate multiplication result must be immediately truncated to 32 bits, otherwise an intermediate type of 60-64 bits will be generated
    // These wide bit widths will cause a large amount of LUT consumption (62-bit adder)
    sample_t term1 = sample_t(sample_t(v1) * w1);
    sample_t term2 = sample_t(sample_t(v2) * w2);
    sample_t term3 = sample_t(sample_t(v3) * w3);
    sample_t term4 = sample_t(sample_t(v4) * w4);
    sample_t result = term1 + term2 + term3 + term4;
    return result;
}

// Helper function to copy feature map to two banks concurrently (Quantized Version)
static void copy_feature_map_to_dual_banks(
    const value_t src_L0[4][16000], const value_t src_L1[4][4000], const value_t src_L2[4][1024], const value_t src_L3[4][256],
    value_t dst0_L0[4][16000], value_t dst0_L1[4][4000], value_t dst0_L2[4][1024], value_t dst0_L3[4][256],
    value_t dst1_L0[4][16000], value_t dst1_L1[4][4000], value_t dst1_L2[4][1024], value_t dst1_L3[4][256]
) {
    #pragma HLS INLINE off
    const int len0 = 16000, len1 = 4000, len2 = 1000, len3 = 250;

    COPY_L0: for (int i = 0; i < len0; ++i) {
        #pragma HLS PIPELINE
        for (int j = 0; j < 4; ++j) {
            #pragma HLS UNROLL
            value_t val = src_L0[j][i];
            dst0_L0[j][i] = val;
            dst1_L0[j][i] = val;
        }
    }
    COPY_L1: for (int i = 0; i < len1; ++i) {
        #pragma HLS PIPELINE
        for (int j = 0; j < 4; ++j) {
            #pragma HLS UNROLL
            value_t val = src_L1[j][i];
            dst0_L1[j][i] = val;
            dst1_L1[j][i] = val;
        }
    }
    COPY_L2: for (int i = 0; i < len2; ++i) {
        #pragma HLS PIPELINE
        for (int j = 0; j < 4; ++j) {
            #pragma HLS UNROLL
            value_t val = src_L2[j][i];
            dst0_L2[j][i] = val;
            dst1_L2[j][i] = val;
        }
    }
    COPY_L3: for (int i = 0; i < len3; ++i) {
        #pragma HLS PIPELINE
        for (int j = 0; j < 4; ++j) {
            #pragma HLS UNROLL
            value_t val = src_L3[j][i];
            dst0_L3[j][i] = val;
            dst1_L3[j][i] = val;
        }
    }
}
 
// Preload the entire 21250 feature map for (b, head, channel) to the on-chip buffer ping-pong loading function
// Note: value_data is layout as [batch][head][channel][spatial=21250] (row-major),
//       therefore the stride for continuous reading along spatial is 1 (spatial is the innermost layer).
// The parameter use_buf1 is true when writing to buf1, otherwise writing to buf0.
// QUANTIZED VERSION: value_data is int16, uses 512-bit reads (32 int16 per read)
// Scatter a 512-bit pack into 16 copies of the destination buffer (partial valid_elems supported)
template<int LEN>
static inline void store_pack_to_buf(
    const ap_uint<512>& pack_data,
    const int s_idx_base,
    const int valid_elems,
    value_t (*dst)[LEN]) {
    #pragma HLS INLINE
    for (int i = 0; i < 32; ++i) {
        #pragma HLS PIPELINE II=1
        if (i < valid_elems) {
            ap_uint<16> bits = pack_data.range(16*(i+1)-1, 16*i);
            ap_int<16> s_val(bits);
            value_t v = s_val.to_int();
            int s_idx = s_idx_base + i;
            for (int copy = 0; copy < 16; ++copy) {
                #pragma HLS UNROLL
                dst[copy][s_idx] = v;
            }
        }
    }
}

// Dual-AXI loader for one level: consumes two 512b ports per cycle (even/odd 32-element packs)
template<int LEN>
static void load_level_dual_axi(
    const value_t* value_data_0,
    const value_t* value_data_1,
    const long long base_linear,
    const int level_offset,
    value_t (*dst)[LEN]) {
    #pragma HLS INLINE off
    const int full_pairs = LEN / 64;        // two packs per pair
    const int mem_base = level_offset;      // offset in the flattened value tensor
    const int start_idx = 0;                // destination offset always starts at 0 per level

    PAIRS: for (int pair = 0; pair < full_pairs; ++pair) {
        #pragma HLS PIPELINE II=1
        const int idx_base = start_idx + pair * 64;
        const ap_uint<512>* pack_ptr0 = reinterpret_cast<const ap_uint<512>*>(value_data_0 + base_linear + (long long)mem_base + (long long)idx_base);
        const ap_uint<512>* pack_ptr1 = reinterpret_cast<const ap_uint<512>*>(value_data_1 + base_linear + (long long)mem_base + (long long)idx_base + 32);
        ap_uint<512> pack0 = *pack_ptr0;
        ap_uint<512> pack1 = *pack_ptr1;
        store_pack_to_buf<LEN>(pack0, idx_base, 32, dst);
        store_pack_to_buf<LEN>(pack1, idx_base + 32, 32, dst);
    }

    int offset = start_idx + full_pairs * 64;
    int remaining = LEN - full_pairs * 64;

    if (remaining >= 32) {
        const ap_uint<512>* pack_ptr = reinterpret_cast<const ap_uint<512>*>(value_data_0 + base_linear + (long long)mem_base + (long long)offset);
        ap_uint<512> pack_data = *pack_ptr;
        store_pack_to_buf<LEN>(pack_data, offset, 32, dst);
        offset += 32;
        remaining -= 32;
    }
    if (remaining > 0) {
        const ap_uint<512>* pack_ptr = reinterpret_cast<const ap_uint<512>*>(value_data_0 + base_linear + (long long)mem_base + (long long)offset);
        ap_uint<512> pack_data = *pack_ptr;
        store_pack_to_buf<LEN>(pack_data, offset, remaining, dst);
    }
}

static void value_reader_pingpong_levels(
    const value_t* value_data_0,  // int16, AXI port A
    const value_t* value_data_1,  // int16, AXI port B
    const int b_col,
    const int m_col,
    const int c_col,
    value_t buf0_L0[16][16000],
    value_t buf0_L1[16][4000],
    value_t buf0_L2[16][1024],
    value_t buf0_L3[16][256],
    value_t buf1_L0[16][16000],
    value_t buf1_L1[16][4000],
    value_t buf1_L2[16][1024],
    value_t buf1_L3[16][256],
    const ap_uint<1> use_buf1
) {
    #pragma HLS INLINE off
    if (c_col >= CONST_CHANNELS) return;

    // 新格式：[batch][head][channel][spatial]
    // 对于位置 (b_col, m_col, c_col, spatial_idx)：
    // index = b_col * (HEAD * CHANNEL * SPATIAL) + m_col * (CHANNEL * SPATIAL) + c_col * SPATIAL + spatial_idx
    const long long base_linear =
        (long long)b_col * (long long)CONST_NUM_HEADS * (long long)CONST_CHANNELS * (long long)CONST_SPATIAL_SIZE +
        (long long)m_col * (long long)CONST_CHANNELS * (long long)CONST_SPATIAL_SIZE +
        (long long)c_col * (long long)CONST_SPATIAL_SIZE;

    // 每个level的长度
    const int len0 = 16000; // 80*200 = 500 groups of 32
    const int len1 = 4000;  // 40*100 = 125 groups of 32
    const int len2 = 1024;  // 20*50 原1000，pad到32对齐
    const int len3 = 256;   // 10*25 原250，pad到32对齐

    // Dual-port loading: each iteration pulls two 512b bursts (even/odd) to halve latency
    if (use_buf1) {
        load_level_dual_axi<16000>(value_data_0, value_data_1, base_linear, 0, buf1_L0);
        load_level_dual_axi<4000>(value_data_0, value_data_1, base_linear, len0, buf1_L1);
        load_level_dual_axi<1024>(value_data_0, value_data_1, base_linear, len0 + len1, buf1_L2);
        load_level_dual_axi<256>(value_data_0, value_data_1, base_linear, len0 + len1 + len2, buf1_L3);
    } else {
        load_level_dual_axi<16000>(value_data_0, value_data_1, base_linear, 0, buf0_L0);
        load_level_dual_axi<4000>(value_data_0, value_data_1, base_linear, len0, buf0_L1);
        load_level_dual_axi<1024>(value_data_0, value_data_1, base_linear, len0 + len1, buf0_L2);
        load_level_dual_axi<256>(value_data_0, value_data_1, base_linear, len0 + len1 + len2, buf0_L3);
    }
}

// Fixed write-in buf0 loading package (convenient for static allocation under dataflow)
static void load_to_buf0(
    const value_t* value_data,
    const value_t* value_data_1,
    const int b_col,
    const int m_col,
    const int c_col,
    value_t buf0_L0[16][16000],
    value_t buf0_L1[16][4000],
    value_t buf0_L2[16][1024],
    value_t buf0_L3[16][256]
) {
    #pragma HLS INLINE off
    // Create a dummy buf1 pointer to avoid writing
    static value_t dummy1_L0[16][16000];
    static value_t dummy1_L1[16][4000];
    static value_t dummy1_L2[16][1024];
    static value_t dummy1_L3[16][256];
    value_reader_pingpong_levels(
        value_data, value_data_1, b_col, m_col, c_col,
        buf0_L0, buf0_L1, buf0_L2, buf0_L3,
        dummy1_L0, dummy1_L1, dummy1_L2, dummy1_L3,
        (ap_uint<1>)0
    );
}

// Fixed write-in buf1 loading package
static void load_to_buf1(
    const value_t* value_data,
    const value_t* value_data_1,
    const int b_col,
    const int m_col,
    const int c_col,
    value_t buf1_L0[16][16000],
    value_t buf1_L1[16][4000],
    value_t buf1_L2[16][1024],
    value_t buf1_L3[16][256]
) {
    #pragma HLS INLINE off
    static value_t dummy0_L0[16][16000];
    static value_t dummy0_L1[16][4000];
    static value_t dummy0_L2[16][1024];
    static value_t dummy0_L3[16][256];
    value_reader_pingpong_levels(
        value_data, value_data_1, b_col, m_col, c_col,
        dummy0_L0, dummy0_L1, dummy0_L2, dummy0_L3,
        buf1_L0, buf1_L1, buf1_L2, buf1_L3,
        (ap_uint<1>)1
    );
}

// Single fetch worker for one channel
// Note: sampling_locations and attention_weights are already converted to ap_fixed and packed
static void fetch_worker(
    const ap_uint<512>* sampling_locations_data,  // packed coord_t data
    const ap_uint<256>* attention_weights_data,   // packed weight_t data
    hls::stream<ap_uint<3072> >& packed_stream,
    const int b_col,
    const int m_col,
    const int num_queries_per_worker)
{
    #pragma HLS INLINE off
    const int SL_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2;
    const int AW_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS;

    LOOP_QUERIES_FETCH_WORKER:
    for (int i = 0; i < num_queries_per_worker; ++i) {
        #pragma HLS PIPELINE II=1
        ap_uint<3072> packed_metadata;
        int bit_offset = 0;

        LOOP_LEVELS_FETCH_WORKER:
        for (int l_col = 0; l_col < CONST_NUM_LEVELS; ++l_col) {
            #pragma HLS UNROLL
            // Calculate the index in the packed data
            // Note: the packed data is aligned by 512 bits/256 bits
            const int loc_base_float =
                b_col * num_queries_per_worker * SL_Q_STRIDE +
                i * SL_Q_STRIDE +
                m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
                l_col * CONST_NUM_POINTS * 2;
            const int weight_base_float =
                b_col * num_queries_per_worker * AW_Q_STRIDE +
                i * AW_Q_STRIDE +
                m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS +
                l_col * CONST_NUM_POINTS;

            // Convert to packed index: 16 floats = 1 512-bit package, 8 floats = 1 256-bit package
            const int loc_pack_idx = loc_base_float / 16;
            const int weight_pack_idx = weight_base_float / 8;

            ap_uint<512> loc_pack = sampling_locations_data[loc_pack_idx];
            ap_uint<256> w_pack   = attention_weights_data[weight_pack_idx];
            
            packed_metadata.range(bit_offset + 511, bit_offset) = loc_pack;
            bit_offset += 512;
            packed_metadata.range(bit_offset + 255, bit_offset) = w_pack;
            bit_offset += 256;
        }
        packed_stream.write(packed_metadata);
    }
}

// Stream merger
static void merge_streams(
    hls::stream<ap_uint<3072> >& s_in_0,
    hls::stream<ap_uint<3072> >& s_in_1,
    hls::stream<ap_uint<3072> >& s_in_2,
    hls::stream<ap_uint<3072> >& s_in_3,
    hls::stream<ap_uint<3072> >& s_out,
    const int num_q_w0,
    const int num_q_w1,
    const int num_q_w2,
    const int num_q_w3
) {
    #pragma HLS INLINE off
    MERGE_0: for (int i = 0; i < num_q_w0; ++i) {
        #pragma HLS PIPELINE
        s_out.write(s_in_0.read());
    }
    MERGE_1: for (int i = 0; i < num_q_w1; ++i) {
        #pragma HLS PIPELINE
        s_out.write(s_in_1.read());
    }
    MERGE_2: for (int i = 0; i < num_q_w2; ++i) {
        #pragma HLS PIPELINE
        s_out.write(s_in_2.read());
    }
    MERGE_3: for (int i = 0; i < num_q_w3; ++i) {
        #pragma HLS PIPELINE
        s_out.write(s_in_3.read());
    }
}

// Metadata fetch controller
static void fetch_metadata(
    const ap_uint<512>* sampling_locations_data_0,
    const ap_uint<512>* sampling_locations_data_1,
    const ap_uint<512>* sampling_locations_data_2,
    const ap_uint<512>* sampling_locations_data_3,
    const ap_uint<256>* attention_weights_data_0,
    const ap_uint<256>* attention_weights_data_1,
    const ap_uint<256>* attention_weights_data_2,
    const ap_uint<256>* attention_weights_data_3,
    hls::stream<ap_uint<3072> >& metadata_stream,
    const int b_col,
    const int m_col)
{
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    const int num_queries_per_worker = CONST_NUM_QUERIES / 4;
    const int num_q_w0 = num_queries_per_worker;
    const int num_q_w1 = num_queries_per_worker;
    const int num_q_w2 = num_queries_per_worker;
    const int num_q_w3 = CONST_NUM_QUERIES - (3 * num_queries_per_worker);

    hls::stream<ap_uint<3072>, 2> packed_stream_0;
    hls::stream<ap_uint<3072>, 2> packed_stream_1;
    hls::stream<ap_uint<3072>, 2> packed_stream_2;
    hls::stream<ap_uint<3072>, 2> packed_stream_3;

    fetch_worker(sampling_locations_data_0, attention_weights_data_0, packed_stream_0, b_col, m_col, num_q_w0);
    fetch_worker(sampling_locations_data_1, attention_weights_data_1, packed_stream_1, b_col, m_col, num_q_w1);
    fetch_worker(sampling_locations_data_2, attention_weights_data_2, packed_stream_2, b_col, m_col, num_q_w2);
    fetch_worker(sampling_locations_data_3, attention_weights_data_3, packed_stream_3, b_col, m_col, num_q_w3);
    
    merge_streams(packed_stream_0, packed_stream_1, packed_stream_2, packed_stream_3,
                  metadata_stream, num_q_w0, num_q_w1, num_q_w2, num_q_w3);
}

// ===== Level-parallel fetcher: 4 workers × 4 levels =====

// Worker for a single level (even query: 0, 2, 4, ...)
static void fetch_worker_level_even(
    const ap_uint<512>* sampling_locations_data,
    const ap_uint<512>* attention_weights_data,
    hls::stream<ap_uint<768> >& level_stream,
    const int b_col,
    const int m_col,
    const int level_idx)
{
    #pragma HLS INLINE off
    const int SL_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2;
    const int AW_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS;
    const int NUM_QUERIES_HALF = CONST_NUM_QUERIES / 2;

    LOOP_QUERIES_FETCH_WORKER_LEVEL_EVEN:
    for (int q_idx = 0; q_idx < NUM_QUERIES_HALF; ++q_idx) {
        #pragma HLS PIPELINE II=1
        const int q = q_idx * 2;  // 偶数 query: 0, 2, 4, ...

        const int loc_base_float =
            b_col * CONST_NUM_QUERIES * SL_Q_STRIDE +
            q     * SL_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
            level_idx * CONST_NUM_POINTS * 2;
        const int weight_base_float =
            b_col * CONST_NUM_QUERIES * AW_Q_STRIDE +
            q     * AW_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS +
            level_idx * CONST_NUM_POINTS;

        const int loc_pack_idx    = loc_base_float / 16;
        const int weight_pack_idx = weight_base_float / 16; // 16 weights per 512-bit pack (2 levels × 8 points)

        ap_uint<512> loc_pack = sampling_locations_data[loc_pack_idx];
        ap_uint<512> w_pack_512 = attention_weights_data[weight_pack_idx];
        // Even levels (0,2) use lower 256 bits, odd levels (1,3) use upper 256 bits
        ap_uint<256> w_pack = (level_idx & 1) ? w_pack_512.range(511, 256)
                                              : w_pack_512.range(255, 0);

        ap_uint<768> level_data;
        level_data.range(511,   0) = loc_pack;
        level_data.range(767, 512) = w_pack;

        level_stream.write(level_data);
    }
}

// Worker for a single level (odd query: 1, 3, 5, ...)
static void fetch_worker_level_odd(
    const ap_uint<512>* sampling_locations_data,
    const ap_uint<512>* attention_weights_data,
    hls::stream<ap_uint<768> >& level_stream,
    const int b_col,
    const int m_col,
    const int level_idx)
{
    #pragma HLS INLINE off
    const int SL_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2;
    const int AW_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS;
    const int NUM_QUERIES_HALF = CONST_NUM_QUERIES / 2;

    LOOP_QUERIES_FETCH_WORKER_LEVEL_ODD:
    for (int q_idx = 0; q_idx < NUM_QUERIES_HALF; ++q_idx) {
        #pragma HLS PIPELINE II=1
        const int q = q_idx * 2 + 1;  // Odd query: 1, 3, 5, ...

        const int loc_base_float =
            b_col * CONST_NUM_QUERIES * SL_Q_STRIDE +
            q     * SL_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
            level_idx * CONST_NUM_POINTS * 2;
        const int weight_base_float =
            b_col * CONST_NUM_QUERIES * AW_Q_STRIDE +
            q     * AW_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS +
            level_idx * CONST_NUM_POINTS;

        const int loc_pack_idx    = loc_base_float / 16;
        const int weight_pack_idx = weight_base_float / 16; // 16 weights per 512-bit pack (2 levels × 8 points)

        ap_uint<512> loc_pack = sampling_locations_data[loc_pack_idx];
        ap_uint<512> w_pack_512 = attention_weights_data[weight_pack_idx];
        ap_uint<256> w_pack = (level_idx & 1) ? w_pack_512.range(511, 256)
                                              : w_pack_512.range(255, 0);

        ap_uint<768> level_data;
        level_data.range(511,   0) = loc_pack;
        level_data.range(767, 512) = w_pack;

        level_stream.write(level_data);
    }
}

// Worker for a pair of levels (even query: 0, 2, 4, ...), read the weights of a pair of levels (same 512-bit package)
// level_idx_base must be 0 or 2, corresponding to these two levels (level_idx_base, level_idx_base+1)
static void fetch_worker_level_pair_even(
    const ap_uint<512>* sampling_locations_data_A,
    const ap_uint<512>* sampling_locations_data_B,
    const ap_uint<512>* attention_weights_data_pair,
    hls::stream<ap_uint<768> >& level_stream_A,
    hls::stream<ap_uint<768> >& level_stream_B,
    const int b_col,
    const int m_col,
    const int level_idx_base)
{
    #pragma HLS INLINE off
    const int SL_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2;
    const int AW_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS;
    const int NUM_QUERIES_HALF = CONST_NUM_QUERIES / 2;
    const int level_idx_A = level_idx_base;
    const int level_idx_B = level_idx_base + 1;

    LOOP_QUERIES_FETCH_WORKER_LEVEL_PAIR_EVEN:
    for (int q_idx = 0; q_idx < NUM_QUERIES_HALF; ++q_idx) {
        #pragma HLS PIPELINE II=1
        const int q = q_idx * 2;  // Even query: 0, 2, 4, ...

        // Level A sampling locations
        const int loc_base_float_A =
            b_col * CONST_NUM_QUERIES * SL_Q_STRIDE +
            q     * SL_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
            level_idx_A * CONST_NUM_POINTS * 2;
        const int loc_pack_idx_A = loc_base_float_A / 16;
        ap_uint<512> loc_pack_A = sampling_locations_data_A[loc_pack_idx_A];

        // Level B sampling locations
        const int loc_base_float_B =
            b_col * CONST_NUM_QUERIES * SL_Q_STRIDE +
            q     * SL_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
            level_idx_B * CONST_NUM_POINTS * 2;
        const int loc_pack_idx_B = loc_base_float_B / 16;
        ap_uint<512> loc_pack_B = sampling_locations_data_B[loc_pack_idx_B];

        // These two levels share a 512-bit weights package (lower 256 bits for level_idx_A, upper 256 bits for level_idx_B)
        const int weight_base_float_pair =
            b_col * CONST_NUM_QUERIES * AW_Q_STRIDE +
            q     * AW_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS +
            level_idx_A * CONST_NUM_POINTS;
        const int weight_pack_idx = weight_base_float_pair / 16; // 16 weights per 512-bit pack (2 levels × 8 points)

        ap_uint<512> w_pack_512 = attention_weights_data_pair[weight_pack_idx];
        ap_uint<256> w_pack_A = w_pack_512.range(255, 0);
        ap_uint<256> w_pack_B = w_pack_512.range(511, 256);

        ap_uint<768> level_data_A;
        level_data_A.range(511,   0) = loc_pack_A;
        level_data_A.range(767, 512) = w_pack_A;

        ap_uint<768> level_data_B;
        level_data_B.range(511,   0) = loc_pack_B;
        level_data_B.range(767, 512) = w_pack_B;

        level_stream_A.write(level_data_A);
        level_stream_B.write(level_data_B);
    }
}

// Worker for a pair of levels (odd query: 1, 3, 5, ...), read the weights of a pair of levels (same 512-bit package)
// level_idx_base must be 0 or 2, corresponding to these two levels (level_idx_base, level_idx_base+1)
static void fetch_worker_level_pair_odd(
    const ap_uint<512>* sampling_locations_data_A,
    const ap_uint<512>* sampling_locations_data_B,
    const ap_uint<512>* attention_weights_data_pair,
    hls::stream<ap_uint<768> >& level_stream_A,
    hls::stream<ap_uint<768> >& level_stream_B,
    const int b_col,
    const int m_col,
    const int level_idx_base)
{
    #pragma HLS INLINE off
    const int SL_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2;
    const int AW_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS;
    const int NUM_QUERIES_HALF = CONST_NUM_QUERIES / 2;
    const int level_idx_A = level_idx_base;
    const int level_idx_B = level_idx_base + 1;

    LOOP_QUERIES_FETCH_WORKER_LEVEL_PAIR_ODD:
    for (int q_idx = 0; q_idx < NUM_QUERIES_HALF; ++q_idx) {
        #pragma HLS PIPELINE II=1
        const int q = q_idx * 2 + 1;  // Odd query: 1, 3, 5, ...

        // Level A sampling locations
        const int loc_base_float_A =
            b_col * CONST_NUM_QUERIES * SL_Q_STRIDE +
            q     * SL_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
            level_idx_A * CONST_NUM_POINTS * 2;
        const int loc_pack_idx_A = loc_base_float_A / 16;
        ap_uint<512> loc_pack_A = sampling_locations_data_A[loc_pack_idx_A];

        // Level B sampling locations
        const int loc_base_float_B =
            b_col * CONST_NUM_QUERIES * SL_Q_STRIDE +
            q     * SL_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
            level_idx_B * CONST_NUM_POINTS * 2;
        const int loc_pack_idx_B = loc_base_float_B / 16;
        ap_uint<512> loc_pack_B = sampling_locations_data_B[loc_pack_idx_B];

        // These two levels share a 512-bit weights package (lower 256 bits for level_idx_A, upper 256 bits for level_idx_B)
        const int weight_base_float_pair =
            b_col * CONST_NUM_QUERIES * AW_Q_STRIDE +
            q     * AW_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS +
            level_idx_A * CONST_NUM_POINTS;
        const int weight_pack_idx = weight_base_float_pair / 16; // 16 weights per 512-bit pack (2 levels × 8 points)

        ap_uint<512> w_pack_512 = attention_weights_data_pair[weight_pack_idx];
        ap_uint<256> w_pack_A = w_pack_512.range(255, 0);
        ap_uint<256> w_pack_B = w_pack_512.range(511, 256);

        ap_uint<768> level_data_A;
        level_data_A.range(511,   0) = loc_pack_A;
        level_data_A.range(767, 512) = w_pack_A;

        ap_uint<768> level_data_B;
        level_data_B.range(511,   0) = loc_pack_B;
        level_data_B.range(767, 512) = w_pack_B;

        level_stream_A.write(level_data_A);
        level_stream_B.write(level_data_B);
    }
}

// Merge 4 level_streams into 3072-bit metadata (process half of the queries)
static void merge_level_streams_half(
    hls::stream<ap_uint<768> >& level_stream_0,
    hls::stream<ap_uint<768> >& level_stream_1,
    hls::stream<ap_uint<768> >& level_stream_2,
    hls::stream<ap_uint<768> >& level_stream_3,
    hls::stream<ap_uint<3072> >& metadata_stream)
{
    #pragma HLS INLINE off
    const int NUM_QUERIES_HALF = CONST_NUM_QUERIES / 2;

    LOOP_QUERIES_MERGE_LEVEL_HALF:
    for (int q = 0; q < NUM_QUERIES_HALF; ++q) {
        #pragma HLS PIPELINE II=1

        ap_uint<768> level_0 = level_stream_0.read();
        ap_uint<768> level_1 = level_stream_1.read();
        ap_uint<768> level_2 = level_stream_2.read();
        ap_uint<768> level_3 = level_stream_3.read();

        ap_uint<3072> packed_metadata;
        packed_metadata.range( 767,   0) = level_0;
        packed_metadata.range(1535,  768) = level_1;
        packed_metadata.range(2303, 1536) = level_2;
        packed_metadata.range(3071, 2304) = level_3;

        metadata_stream.write(packed_metadata);
    }
}

// Metadata fetch for even queries (0, 2, 4, ...)
static void fetch_metadata_even(
    const ap_uint<512>* sampling_locations_data_0,
    const ap_uint<512>* sampling_locations_data_1,
    const ap_uint<512>* sampling_locations_data_2,
    const ap_uint<512>* sampling_locations_data_3,
    const ap_uint<512>* attention_weights_data_0,
    const ap_uint<512>* attention_weights_data_1,
    hls::stream<ap_uint<3072> >& metadata_stream,
    const int b_col,
    const int m_col)
{
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    hls::stream<ap_uint<768>, 32> level_stream_0;
    hls::stream<ap_uint<768>, 32> level_stream_1;
    hls::stream<ap_uint<768>, 32> level_stream_2;
    hls::stream<ap_uint<768>, 32> level_stream_3;

    // Levels 0 & 1 share attention_weights_data_0 (packed as lower / upper 256 bits)
    fetch_worker_level_pair_even(
        sampling_locations_data_0, sampling_locations_data_1,
        attention_weights_data_0,
        level_stream_0, level_stream_1,
        b_col, m_col, 0);
    // Levels 2 & 3 share attention_weights_data_1
    fetch_worker_level_pair_even(
        sampling_locations_data_2, sampling_locations_data_3,
        attention_weights_data_1,
        level_stream_2, level_stream_3,
        b_col, m_col, 2);

    merge_level_streams_half(level_stream_0, level_stream_1, level_stream_2, level_stream_3, metadata_stream);
}

// Metadata fetch for odd queries (1, 3, 5, ...)
static void fetch_metadata_odd(
    const ap_uint<512>* sampling_locations_data_4,
    const ap_uint<512>* sampling_locations_data_5,
    const ap_uint<512>* sampling_locations_data_6,
    const ap_uint<512>* sampling_locations_data_7,
    const ap_uint<512>* attention_weights_data_4,
    const ap_uint<512>* attention_weights_data_5,
    hls::stream<ap_uint<3072> >& metadata_stream,
    const int b_col,
    const int m_col)
{
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    hls::stream<ap_uint<768>, 32> level_stream_0;
    hls::stream<ap_uint<768>, 32> level_stream_1;
    hls::stream<ap_uint<768>, 32> level_stream_2;
    hls::stream<ap_uint<768>, 32> level_stream_3;

    // Levels 0 & 1 share attention_weights_data_4
    fetch_worker_level_pair_odd(
        sampling_locations_data_4, sampling_locations_data_5,
        attention_weights_data_4,
        level_stream_0, level_stream_1,
        b_col, m_col, 0);
    // Levels 2 & 3 share attention_weights_data_5
    fetch_worker_level_pair_odd(
        sampling_locations_data_6, sampling_locations_data_7,
        attention_weights_data_5,
        level_stream_2, level_stream_3,
        b_col, m_col, 2);

    merge_level_streams_half(level_stream_0, level_stream_1, level_stream_2, level_stream_3, metadata_stream);
}

// Runtime data range detection variables (disabled)
// #ifndef __SYNTHESIS__
// static float g_weight_min = 1e10, g_weight_max = -1e10;
// static float g_sample_min = 1e10, g_sample_max = -1e10;
// static float g_weighted_min = 1e10, g_weighted_max = -1e10;
// static float g_point_sum_min = 1e10, g_point_sum_max = -1e10;
// static float g_accum_val_min = 1e10, g_accum_val_max = -1e10;
// static int g_sample_count = 0;
// static int g_accum_count = 0;
// #endif

// Computation module: process even queries (0, 2, 4, ...), using copy 0-3
static void execute_computation_even(
    hls::stream<ap_uint<3072> >& metadata_stream,
    output_t* output_data,
    const value_t cur_L0[16][16000], const value_t cur_L1[16][4000], const value_t cur_L2[16][1024], const value_t cur_L3[16][256],
    const int b_col,
    const int m_col,
    const int c_col
)
{
    #pragma HLS INLINE off
    const int NUM_QUERIES_HALF = CONST_NUM_QUERIES / 2;

    LOOP_NUM_QUERIES_EVEN:
    for (int query_idx = 0; query_idx < NUM_QUERIES_HALF; ++query_idx) {
        #pragma HLS PIPELINE II=1
        
        accum_t accumulated_value = 0;
        ap_uint<3072> packed_metadata = metadata_stream.read();
        int bit_offset = 0;

        LOOP_LEVELS:
        for (int l_col = 0; l_col < CONST_NUM_LEVELS; ++l_col) {
            #pragma HLS UNROLL
            
            coord_t loc_w_arr[CONST_NUM_POINTS];
            coord_t loc_h_arr[CONST_NUM_POINTS];
            weight_t weight_arr[CONST_NUM_POINTS];
            #pragma HLS ARRAY_PARTITION variable=loc_w_arr complete dim=1
            #pragma HLS ARRAY_PARTITION variable=loc_h_arr complete dim=1
            #pragma HLS ARRAY_PARTITION variable=weight_arr complete dim=1

            ap_uint<512> loc_pack = packed_metadata.range(bit_offset + 511, bit_offset);
            bit_offset += 512;
            ap_uint<256> w_pack = packed_metadata.range(bit_offset + 255, bit_offset);
            bit_offset += 256;

            for (int i = 0; i < 16; ++i) {
                #pragma HLS UNROLL
                ap_uint<32> word = loc_pack.range(32*(i+1)-1, 32*i);
                ap_uint<27> coord_bits = word.range(26, 0);
                coord_t coord_val;
                std::memcpy(&coord_val, &coord_bits, sizeof(coord_t));
                if ((i & 1) == 0) loc_w_arr[i>>1] = coord_val; else loc_h_arr[i>>1] = coord_val;
            }
            for (int i = 0; i < 8; ++i) {
                #pragma HLS UNROLL
                ap_uint<32> word = w_pack.range(32*(i+1)-1, 32*i);
                ap_uint<23> weight_bits = word.range(22, 0);
                weight_t weight_val;
                std::memcpy(&weight_val, &weight_bits, sizeof(weight_t));
                weight_arr[i] = weight_val;
            }

            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = CONST_SPATIAL_SHAPES[spatial_h_ptr];
            const int spatial_w = CONST_SPATIAL_SHAPES[spatial_h_ptr + 1];
            
            // Using copy 0-3
            const value_t* cur_buffers[4];
            #pragma HLS ARRAY_PARTITION variable=cur_buffers complete
            for(int i=0; i<4; ++i) {
                #pragma HLS UNROLL
                cur_buffers[i] = (l_col == 0) ? cur_L0[i] :
                                 (l_col == 1) ? cur_L1[i] :
                                 (l_col == 2) ? cur_L2[i] :
                                                cur_L3[i];
            }
            
            accum_t point_results[CONST_NUM_POINTS];
            #pragma HLS ARRAY_PARTITION variable=point_results complete

            LOOP_POINTS:
            for (int p_col = 0; p_col < CONST_NUM_POINTS; ++p_col) {
                #pragma HLS UNROLL
                const value_t* cur0 = cur_buffers[p_col % 4];
                
                const coord_t loc_w = loc_w_arr[p_col];
                const coord_t loc_h = loc_h_arr[p_col];
                const weight_t weight = weight_arr[p_col];

                coord_t h_im_temp = loc_h * coord_t(spatial_h);
                coord_t w_im_temp = loc_w * coord_t(spatial_w);
                #pragma HLS RESOURCE variable=h_im_temp core=Mul_DSP
                #pragma HLS RESOURCE variable=w_im_temp core=Mul_DSP
                const coord_t h_im = h_im_temp - coord_t(0.5);
                const coord_t w_im = w_im_temp - coord_t(0.5);

                int h_nearest, w_nearest;
                const coord_t h_im_plus_half = h_im + coord_t(0.5);
                const coord_t w_im_plus_half = w_im + coord_t(0.5);
                
                if (h_im >= coord_t(0.0)) {
                    h_nearest = (int)h_im_plus_half;
                } else {
                    coord_t h_im_frac = h_im - coord_t((int)h_im);
                    if (h_im_frac <= coord_t(-0.5)) {
                        h_nearest = (int)h_im - 1;
                    } else {
                        h_nearest = (int)h_im;
                    }
                }
                
                if (w_im >= coord_t(0.0)) {
                    w_nearest = (int)w_im_plus_half;
                } else {
                    coord_t w_im_frac = w_im - coord_t((int)w_im);
                    if (w_im_frac <= coord_t(-0.5)) {
                        w_nearest = (int)w_im - 1;
                    } else {
                        w_nearest = (int)w_im;
                    }
                }

                const int valid = (h_nearest >= 0) & (w_nearest >= 0) & (h_nearest < spatial_h) & (w_nearest < spatial_w);
                const int lin_idx = h_nearest * spatial_w + w_nearest;

                value_t sampled_value_int16 = 0;
                if (valid) {
                    sampled_value_int16 = cur0[lin_idx];
                }

                const sample_t sampled_value = sample_t(sampled_value_int16);
                sample_t weighted_sample = sampled_value * weight;
                #pragma HLS RESOURCE variable=weighted_sample core=Mul_DSP
                
                point_results[p_col] = accum_t(weighted_sample);
            }


            accum_t point_sum_for_level = 0;
            SUM_POINTS: for(int i=0; i<CONST_NUM_POINTS; ++i) {
                #pragma HLS UNROLL
                #pragma HLS RESOURCE variable=point_sum_for_level core=AddSub_DSP
                point_sum_for_level += point_results[i];
            }
            #pragma HLS RESOURCE variable=accumulated_value core=AddSub_DSP
            accumulated_value += point_sum_for_level;
        }

        // Write to output_data_0, the index is continuous (corresponding to query 0, 2, 4, ...)
        const int out_idx =
            b_col * (NUM_QUERIES_HALF * CONST_NUM_HEADS * CONST_CHANNELS) +
            query_idx * (CONST_NUM_HEADS * CONST_CHANNELS) +
            m_col * CONST_CHANNELS +
            c_col;
        output_data[out_idx] = accumulated_value;
    }
}

// Computation module: process odd queries (1, 3, 5, ...), using copy 4-7
static void execute_computation_odd(
    hls::stream<ap_uint<3072> >& metadata_stream,
    output_t* output_data,
    const value_t cur_L0[16][16000], const value_t cur_L1[16][4000], const value_t cur_L2[16][1024], const value_t cur_L3[16][256],
    const int b_col,
    const int m_col,
    const int c_col
)
{
    #pragma HLS INLINE off
    const int NUM_QUERIES_HALF = CONST_NUM_QUERIES / 2;

    LOOP_NUM_QUERIES_ODD:
    for (int query_idx = 0; query_idx < NUM_QUERIES_HALF; ++query_idx) {
        #pragma HLS PIPELINE II=1
        
        accum_t accumulated_value = 0;
        ap_uint<3072> packed_metadata = metadata_stream.read();
        int bit_offset = 0;

        LOOP_LEVELS:
        for (int l_col = 0; l_col < CONST_NUM_LEVELS; ++l_col) {
            #pragma HLS UNROLL
            
            coord_t loc_w_arr[CONST_NUM_POINTS];
            coord_t loc_h_arr[CONST_NUM_POINTS];
            weight_t weight_arr[CONST_NUM_POINTS];
            #pragma HLS ARRAY_PARTITION variable=loc_w_arr complete dim=1
            #pragma HLS ARRAY_PARTITION variable=loc_h_arr complete dim=1
            #pragma HLS ARRAY_PARTITION variable=weight_arr complete dim=1

            ap_uint<512> loc_pack = packed_metadata.range(bit_offset + 511, bit_offset);
            bit_offset += 512;
            ap_uint<256> w_pack = packed_metadata.range(bit_offset + 255, bit_offset);
            bit_offset += 256;

            for (int i = 0; i < 16; ++i) {
                #pragma HLS UNROLL
                ap_uint<32> word = loc_pack.range(32*(i+1)-1, 32*i);
                ap_uint<27> coord_bits = word.range(26, 0);
                coord_t coord_val;
                std::memcpy(&coord_val, &coord_bits, sizeof(coord_t));
                if ((i & 1) == 0) loc_w_arr[i>>1] = coord_val; else loc_h_arr[i>>1] = coord_val;
            }
            for (int i = 0; i < 8; ++i) {
                #pragma HLS UNROLL
                ap_uint<32> word = w_pack.range(32*(i+1)-1, 32*i);
                ap_uint<23> weight_bits = word.range(22, 0);
                weight_t weight_val;
                std::memcpy(&weight_val, &weight_bits, sizeof(weight_t));
                weight_arr[i] = weight_val;
            }

            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = CONST_SPATIAL_SHAPES[spatial_h_ptr];
            const int spatial_w = CONST_SPATIAL_SHAPES[spatial_h_ptr + 1];
            
            // Using copy 4-7
            const value_t* cur_buffers[4];
            #pragma HLS ARRAY_PARTITION variable=cur_buffers complete
            for(int i=0; i<4; ++i) {
                #pragma HLS UNROLL
                cur_buffers[i] = (l_col == 0) ? cur_L0[i+4] :
                                 (l_col == 1) ? cur_L1[i+4] :
                                 (l_col == 2) ? cur_L2[i+4] :
                                                cur_L3[i+4];
            }
            
            accum_t point_results[CONST_NUM_POINTS];
            #pragma HLS ARRAY_PARTITION variable=point_results complete

            LOOP_POINTS:
            for (int p_col = 0; p_col < CONST_NUM_POINTS; ++p_col) {
                #pragma HLS UNROLL
                const value_t* cur0 = cur_buffers[p_col % 4];
                
                const coord_t loc_w = loc_w_arr[p_col];
                const coord_t loc_h = loc_h_arr[p_col];
                const weight_t weight = weight_arr[p_col];

                coord_t h_im_temp = loc_h * coord_t(spatial_h);
                coord_t w_im_temp = loc_w * coord_t(spatial_w);
                #pragma HLS RESOURCE variable=h_im_temp core=Mul_DSP
                #pragma HLS RESOURCE variable=w_im_temp core=Mul_DSP
                const coord_t h_im = h_im_temp - coord_t(0.5);
                const coord_t w_im = w_im_temp - coord_t(0.5);

                int h_nearest, w_nearest;
                const coord_t h_im_plus_half = h_im + coord_t(0.5);
                const coord_t w_im_plus_half = w_im + coord_t(0.5);
                
                if (h_im >= coord_t(0.0)) {
                    h_nearest = (int)h_im_plus_half;
                } else {
                    coord_t h_im_frac = h_im - coord_t((int)h_im);
                    if (h_im_frac <= coord_t(-0.5)) {
                        h_nearest = (int)h_im - 1;
                    } else {
                        h_nearest = (int)h_im;
                    }
                }
                
                if (w_im >= coord_t(0.0)) {
                    w_nearest = (int)w_im_plus_half;
                } else {
                    coord_t w_im_frac = w_im - coord_t((int)w_im);
                    if (w_im_frac <= coord_t(-0.5)) {
                        w_nearest = (int)w_im - 1;
                    } else {
                        w_nearest = (int)w_im;
                    }
                }

                const int valid = (h_nearest >= 0) & (w_nearest >= 0) & (h_nearest < spatial_h) & (w_nearest < spatial_w);
                const int lin_idx = h_nearest * spatial_w + w_nearest;

                value_t sampled_value_int16 = 0;
                if (valid) {
                    sampled_value_int16 = cur0[lin_idx];
                }

                const sample_t sampled_value = sample_t(sampled_value_int16);
                sample_t weighted_sample = sampled_value * weight;
                #pragma HLS RESOURCE variable=weighted_sample core=Mul_DSP
                
                point_results[p_col] = accum_t(weighted_sample);
            }

            accum_t point_sum_for_level = 0;
            SUM_POINTS: for(int i=0; i<CONST_NUM_POINTS; ++i) {
                #pragma HLS UNROLL
                #pragma HLS RESOURCE variable=point_sum_for_level core=AddSub_DSP
                point_sum_for_level += point_results[i];
            }
            #pragma HLS RESOURCE variable=accumulated_value core=AddSub_DSP
            accumulated_value += point_sum_for_level;
        }

        // Write to output_data_1, the index is continuous (corresponding to query 1, 3, 5, ...)
        const int out_idx =
            b_col * (NUM_QUERIES_HALF * CONST_NUM_HEADS * CONST_CHANNELS) +
            query_idx * (CONST_NUM_HEADS * CONST_CHANNELS) +
            m_col * CONST_CHANNELS +
            c_col;
        output_data[out_idx] = accumulated_value;
    }
}

// ===== 4-query parallel: per-group metadata fetch (q % 4 == group_mod) =====
static void fetch_metadata_group(
    const ap_uint<512>* sampling_locations_data_0,
    const ap_uint<512>* sampling_locations_data_1,
    const ap_uint<512>* sampling_locations_data_2,
    const ap_uint<512>* sampling_locations_data_3,
    const ap_uint<512>* attention_weights_data_0,
    const ap_uint<512>* attention_weights_data_1,
    hls::stream<ap_uint<3072> >& metadata_stream,
    const int b_col,
    const int m_col,
    const int group_mod)  // 0,1,2,3
{
    #pragma HLS INLINE off

    const int SL_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2;
    const int AW_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS;
    // Support non-4 divisible: the number of queries per group is the ceiling of Q/4
    const int NUM_QUERIES_GROUP = (CONST_NUM_QUERIES + 3) >> 2; // ceil(Q/4)

    const ap_uint<512>* sl_ptrs[4];
    #pragma HLS ARRAY_PARTITION variable=sl_ptrs complete
    sl_ptrs[0] = sampling_locations_data_0;
    sl_ptrs[1] = sampling_locations_data_1;
    sl_ptrs[2] = sampling_locations_data_2;
    sl_ptrs[3] = sampling_locations_data_3;

    // Each group processes: q = group_mod + 4 * q_idx
    LOOP_QUERIES_FETCH_GROUP:
    for (int q_idx = 0; q_idx < NUM_QUERIES_GROUP; ++q_idx) {
        #pragma HLS PIPELINE II=1
        const int q = group_mod + (q_idx << 2);
        if (q >= CONST_NUM_QUERIES) break; // End early if q exceeds actual Q

        ap_uint<3072> packed_metadata;
        int bit_offset = 0;

        for (int l_col = 0; l_col < CONST_NUM_LEVELS; ++l_col) {
            #pragma HLS UNROLL

            const ap_uint<512>* sl_ptr =
                (l_col == 0) ? sl_ptrs[0] :
                (l_col == 1) ? sl_ptrs[1] :
                (l_col == 2) ? sl_ptrs[2] :
                               sl_ptrs[3];

            // sampling locations
            const int loc_base_float =
                b_col * CONST_NUM_QUERIES * SL_Q_STRIDE +
                q     * SL_Q_STRIDE +
                m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
                l_col * CONST_NUM_POINTS * 2;
            const int loc_pack_idx = loc_base_float / 16;
            ap_uint<512> loc_pack = sl_ptr[loc_pack_idx];

            // attention weights: two levels packed in the same 512-bit
            const ap_uint<512>* aw_pair_ptr;
            int base_level;
            if (l_col < 2) {
                aw_pair_ptr = attention_weights_data_0;
                base_level = 0;
            } else {
                aw_pair_ptr = attention_weights_data_1;
                base_level = 2;
            }
            const int weight_base_float_pair =
                b_col * CONST_NUM_QUERIES * AW_Q_STRIDE +
                q     * AW_Q_STRIDE +
                m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS +
                base_level * CONST_NUM_POINTS;
                const int weight_pack_idx = weight_base_float_pair / 16; // 16 weights (2 levels×8 points) per 512-bit
            ap_uint<512> w_pack_512 = aw_pair_ptr[weight_pack_idx];
            ap_uint<256> w_pack = ( (l_col & 1) == 0 )
                                  ? w_pack_512.range(255, 0)
                                  : w_pack_512.range(511, 256);

            packed_metadata.range(bit_offset + 511, bit_offset) = loc_pack;
            bit_offset += 512;
            packed_metadata.range(bit_offset + 255, bit_offset) = w_pack;
            bit_offset += 256;
        }

        metadata_stream.write(packed_metadata);
    }
}

// Computation module: each group processes CONST_NUM_QUERIES/4 queries, using 4 BRAM copies
static void execute_computation_group(
    hls::stream<ap_uint<3072> >& metadata_stream,
    output_t* output_data_group,
    const value_t cur_L0[16][16000], const value_t cur_L1[16][4000], const value_t cur_L2[16][1024], const value_t cur_L3[16][256],
    const int b_col,
    const int m_col,
    const int c_col,
    const int base_copy)  // 0,4,8,12
{
    #pragma HLS INLINE off
    const int NUM_QUERIES_GROUP = (CONST_NUM_QUERIES + 3) >> 2; // ceil(Q/4)

    LOOP_NUM_QUERIES_GROUP:
    for (int query_idx = 0; query_idx < NUM_QUERIES_GROUP; ++query_idx) {
        #pragma HLS PIPELINE II=1
        const int q_global = (query_idx << 2) + (base_copy >> 2); // base_copy=0/4/8/12 => group_mod
        if (q_global >= CONST_NUM_QUERIES) break; // End early if q_global exceeds actual Q

        accum_t accumulated_value = 0;
        ap_uint<3072> packed_metadata = metadata_stream.read();
        int bit_offset = 0;

        LOOP_LEVELS_GROUP:
        for (int l_col = 0; l_col < CONST_NUM_LEVELS; ++l_col) {
            #pragma HLS UNROLL

            coord_t loc_w_arr[CONST_NUM_POINTS];
            coord_t loc_h_arr[CONST_NUM_POINTS];
            weight_t weight_arr[CONST_NUM_POINTS];
            #pragma HLS ARRAY_PARTITION variable=loc_w_arr complete dim=1
            #pragma HLS ARRAY_PARTITION variable=loc_h_arr complete dim=1
            #pragma HLS ARRAY_PARTITION variable=weight_arr complete dim=1

            ap_uint<512> loc_pack = packed_metadata.range(bit_offset + 511, bit_offset);
            bit_offset += 512;
            ap_uint<256> w_pack = packed_metadata.range(bit_offset + 255, bit_offset);
            bit_offset += 256;

            for (int i = 0; i < 16; ++i) {
                #pragma HLS UNROLL
                ap_uint<32> word = loc_pack.range(32*(i+1)-1, 32*i);
                ap_uint<27> coord_bits = word.range(26, 0);
                coord_t coord_val;
                std::memcpy(&coord_val, &coord_bits, sizeof(coord_t));
                if ((i & 1) == 0) loc_w_arr[i>>1] = coord_val; else loc_h_arr[i>>1] = coord_val;
            }
            for (int i = 0; i < 8; ++i) {
                #pragma HLS UNROLL
                ap_uint<32> word = w_pack.range(32*(i+1)-1, 32*i);
                ap_uint<23> weight_bits = word.range(22, 0);
                weight_t weight_val;
                std::memcpy(&weight_val, &weight_bits, sizeof(weight_t));
                weight_arr[i] = weight_val;
            }

            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = CONST_SPATIAL_SHAPES[spatial_h_ptr];
            const int spatial_w = CONST_SPATIAL_SHAPES[spatial_h_ptr + 1];

            // Select 4 BRAM copies for this group
            const value_t* cur_buffers[4];
            #pragma HLS ARRAY_PARTITION variable=cur_buffers complete
            for (int i = 0; i < 4; ++i) {
                #pragma HLS UNROLL
                const int copy_idx = base_copy + i;
                cur_buffers[i] = (l_col == 0) ? cur_L0[copy_idx] :
                                 (l_col == 1) ? cur_L1[copy_idx] :
                                 (l_col == 2) ? cur_L2[copy_idx] :
                                                cur_L3[copy_idx];
            }

            accum_t point_results[CONST_NUM_POINTS];
            #pragma HLS ARRAY_PARTITION variable=point_results complete

            LOOP_POINTS_GROUP:
            for (int p_col = 0; p_col < CONST_NUM_POINTS; ++p_col) {
                #pragma HLS UNROLL
                const value_t* cur0 = cur_buffers[p_col % 4];

                const coord_t loc_w = loc_w_arr[p_col];
                const coord_t loc_h = loc_h_arr[p_col];
                const weight_t weight = weight_arr[p_col];

                coord_t h_im_temp = loc_h * coord_t(spatial_h);
                coord_t w_im_temp = loc_w * coord_t(spatial_w);
                #pragma HLS RESOURCE variable=h_im_temp core=Mul_DSP
                #pragma HLS RESOURCE variable=w_im_temp core=Mul_DSP
                const coord_t h_im = h_im_temp - coord_t(0.5);
                const coord_t w_im = w_im_temp - coord_t(0.5);

                int h_nearest, w_nearest;
                const coord_t h_im_plus_half = h_im + coord_t(0.5);
                const coord_t w_im_plus_half = w_im + coord_t(0.5);

                if (h_im >= coord_t(0.0)) {
                    h_nearest = (int)h_im_plus_half;
                } else {
                    coord_t h_im_frac = h_im - coord_t((int)h_im);
                    if (h_im_frac <= coord_t(-0.5)) {
                        h_nearest = (int)h_im - 1;
                    } else {
                        h_nearest = (int)h_im;
                    }
                }

                if (w_im >= coord_t(0.0)) {
                    w_nearest = (int)w_im_plus_half;
                } else {
                    coord_t w_im_frac = w_im - coord_t((int)w_im);
                    if (w_im_frac <= coord_t(-0.5)) {
                        w_nearest = (int)w_im - 1;
                    } else {
                        w_nearest = (int)w_im;
                    }
                }

                const int valid = (h_nearest >= 0) & (w_nearest >= 0) & (h_nearest < spatial_h) & (w_nearest < spatial_w);
                const int lin_idx = h_nearest * spatial_w + w_nearest;

                value_t sampled_value_int16 = 0;
                if (valid) {
                    sampled_value_int16 = cur0[lin_idx];
                }

                const sample_t sampled_value = sample_t(sampled_value_int16);
                sample_t weighted_sample = sampled_value * weight;
                #pragma HLS RESOURCE variable=weighted_sample core=Mul_DSP

                point_results[p_col] = accum_t(weighted_sample);
            }

            accum_t point_sum_for_level = 0;
            SUM_POINTS_GROUP:
            for (int i = 0; i < CONST_NUM_POINTS; ++i) {
                #pragma HLS UNROLL
                #pragma HLS RESOURCE variable=point_sum_for_level core=AddSub_DSP
                point_sum_for_level += point_results[i];
            }
            #pragma HLS RESOURCE variable=accumulated_value core=AddSub_DSP
            accumulated_value += point_sum_for_level;
        }

        // Write the output of this group, the index is continuous within this group
        const int Q_GROUP = (CONST_NUM_QUERIES + 3) >> 2;
        const int out_idx =
            b_col * (Q_GROUP * CONST_NUM_HEADS * CONST_CHANNELS) +
            query_idx * (CONST_NUM_HEADS * CONST_CHANNELS) +
            m_col * CONST_CHANNELS +
            c_col;
        output_data_group[out_idx] = accumulated_value;
    }
}

// 4-query parallel: 4 groups of metadata + compute (each group has independent SL/AW/output)
static void compute_channel_4q(
    // Group 0 (q % 4 == 0)
    const ap_uint<512>* sampling_locations_data_0,
    const ap_uint<512>* sampling_locations_data_1,
    const ap_uint<512>* sampling_locations_data_2,
    const ap_uint<512>* sampling_locations_data_3,
    // Group 1 (q % 4 == 1)
    const ap_uint<512>* sampling_locations_data_4,
    const ap_uint<512>* sampling_locations_data_5,
    const ap_uint<512>* sampling_locations_data_6,
    const ap_uint<512>* sampling_locations_data_7,
    // Group 2 (q % 4 == 2)
    const ap_uint<512>* sampling_locations_data_8,
    const ap_uint<512>* sampling_locations_data_9,
    const ap_uint<512>* sampling_locations_data_10,
    const ap_uint<512>* sampling_locations_data_11,
    // Group 3 (q % 4 == 3)
    const ap_uint<512>* sampling_locations_data_12,
    const ap_uint<512>* sampling_locations_data_13,
    const ap_uint<512>* sampling_locations_data_14,
    const ap_uint<512>* sampling_locations_data_15,
    // attention weights for 4 groups
    const ap_uint<512>* attention_weights_data_0,
    const ap_uint<512>* attention_weights_data_1,
    const ap_uint<512>* attention_weights_data_2,
    const ap_uint<512>* attention_weights_data_3,
    const ap_uint<512>* attention_weights_data_4,
    const ap_uint<512>* attention_weights_data_5,
    const ap_uint<512>* attention_weights_data_6,
    const ap_uint<512>* attention_weights_data_7,
    // outputs
    output_t* output_data_0,
    output_t* output_data_1,
    output_t* output_data_2,
    output_t* output_data_3,
    const value_t cur_L0[16][16000], const value_t cur_L1[16][4000], const value_t cur_L2[16][1024], const value_t cur_L3[16][256],
    const int b_col,
    const int m_col,
    const int c_col
) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    hls::stream<ap_uint<3072>, 4> metadata_stream_0;
    hls::stream<ap_uint<3072>, 4> metadata_stream_1;
    hls::stream<ap_uint<3072>, 4> metadata_stream_2;
    hls::stream<ap_uint<3072>, 4> metadata_stream_3;

    // metadata fetch for 4 groups (q%4)
    fetch_metadata_group(
        sampling_locations_data_0, sampling_locations_data_1, sampling_locations_data_2, sampling_locations_data_3,
        attention_weights_data_0, attention_weights_data_1,
        metadata_stream_0, b_col, m_col, 0);
    fetch_metadata_group(
        sampling_locations_data_4, sampling_locations_data_5, sampling_locations_data_6, sampling_locations_data_7,
        attention_weights_data_2, attention_weights_data_3,
        metadata_stream_1, b_col, m_col, 1);
    fetch_metadata_group(
        sampling_locations_data_8, sampling_locations_data_9, sampling_locations_data_10, sampling_locations_data_11,
        attention_weights_data_4, attention_weights_data_5,
        metadata_stream_2, b_col, m_col, 2);
    fetch_metadata_group(
        sampling_locations_data_12, sampling_locations_data_13, sampling_locations_data_14, sampling_locations_data_15,
        attention_weights_data_6, attention_weights_data_7,
        metadata_stream_3, b_col, m_col, 3);

    // 4 groups of compute (each uses 4 BRAM copies)
    execute_computation_group(metadata_stream_0, output_data_0,
        cur_L0, cur_L1, cur_L2, cur_L3,
        b_col, m_col, c_col,
        0);   // copies 0-3
    execute_computation_group(metadata_stream_1, output_data_1,
        cur_L0, cur_L1, cur_L2, cur_L3,
        b_col, m_col, c_col,
        4);   // copies 4-7
    execute_computation_group(metadata_stream_2, output_data_2,
        cur_L0, cur_L1, cur_L2, cur_L3,
        b_col, m_col, c_col,
        8);   // copies 8-11
    execute_computation_group(metadata_stream_3, output_data_3,
        cur_L0, cur_L1, cur_L2, cur_L3,
        b_col, m_col, c_col,
        12);  // copies 12-15
}

// (Removed, flattened call relationships to facilitate FUNCTION_INSTANTIATE)

// Unified df_pair function: select buffer based on use_buf1 at runtime (avoid duplicate hardware)
static void df_pair_unified(
    const value_t* value_data,
    const value_t* value_data_1,
    const ap_uint<512>* sampling_locations_data_0,
    const ap_uint<512>* sampling_locations_data_1,
    const ap_uint<512>* sampling_locations_data_2,
    const ap_uint<512>* sampling_locations_data_3,
    const ap_uint<512>* sampling_locations_data_4,
    const ap_uint<512>* sampling_locations_data_5,
    const ap_uint<512>* sampling_locations_data_6,
    const ap_uint<512>* sampling_locations_data_7,
    const ap_uint<512>* attention_weights_data_0,
    const ap_uint<512>* attention_weights_data_1,
    const ap_uint<512>* attention_weights_data_2,
    const ap_uint<512>* attention_weights_data_3,
    const ap_uint<512>* sampling_locations_data_8,
    const ap_uint<512>* sampling_locations_data_9,
    const ap_uint<512>* sampling_locations_data_10,
    const ap_uint<512>* sampling_locations_data_11,
    const ap_uint<512>* sampling_locations_data_12,
    const ap_uint<512>* sampling_locations_data_13,
    const ap_uint<512>* sampling_locations_data_14,
    const ap_uint<512>* sampling_locations_data_15,
    const ap_uint<512>* attention_weights_data_4,
    const ap_uint<512>* attention_weights_data_5,
    const ap_uint<512>* attention_weights_data_6,
    const ap_uint<512>* attention_weights_data_7,
    output_t* output_data_0,
    output_t* output_data_1,
    output_t* output_data_2,
    output_t* output_data_3,
    const int b_col,
    const int m_col,
    const int c_col,
    value_t buf0_L0[16][16000], value_t buf0_L1[16][4000], value_t buf0_L2[16][1024], value_t buf0_L3[16][256],
    value_t buf1_L0[16][16000], value_t buf1_L1[16][4000], value_t buf1_L2[16][1024], value_t buf1_L3[16][256],
    const ap_uint<1> use_buf1  // 0=使用buf0计算, 1=使用buf1计算
) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    
    // Select the current computation buffer and the next preload buffer based on use_buf1
    if (use_buf1 == (ap_uint<1>)0) {
        // Compute using buf0, preload to buf1
        compute_channel_4q(
            // SL groups
            sampling_locations_data_0, sampling_locations_data_1, sampling_locations_data_2, sampling_locations_data_3,
            sampling_locations_data_4, sampling_locations_data_5, sampling_locations_data_6, sampling_locations_data_7,
            sampling_locations_data_8,  sampling_locations_data_9,  sampling_locations_data_10, sampling_locations_data_11,
            sampling_locations_data_12, sampling_locations_data_13, sampling_locations_data_14, sampling_locations_data_15,
            // AW groups
            attention_weights_data_0, attention_weights_data_1,
            attention_weights_data_2, attention_weights_data_3,
            attention_weights_data_4, attention_weights_data_5,
            attention_weights_data_6, attention_weights_data_7,
            // outputs
            output_data_0, output_data_1, output_data_2, output_data_3,
            buf0_L0, buf0_L1, buf0_L2, buf0_L3,
            b_col, m_col, c_col);
        load_to_buf1(value_data, value_data_1, b_col, m_col, c_col + 1,
            buf1_L0, buf1_L1, buf1_L2, buf1_L3);
    } else {
        // Compute using buf1, preload to buf0
        compute_channel_4q(
            // SL groups
            sampling_locations_data_0, sampling_locations_data_1, sampling_locations_data_2, sampling_locations_data_3,
            sampling_locations_data_4, sampling_locations_data_5, sampling_locations_data_6, sampling_locations_data_7,
            sampling_locations_data_8,  sampling_locations_data_9,  sampling_locations_data_10, sampling_locations_data_11,
            sampling_locations_data_12, sampling_locations_data_13, sampling_locations_data_14, sampling_locations_data_15,
            // AW groups
            attention_weights_data_0, attention_weights_data_1,
            attention_weights_data_2, attention_weights_data_3,
            attention_weights_data_4, attention_weights_data_5,
            attention_weights_data_6, attention_weights_data_7,
            // outputs
            output_data_0, output_data_1, output_data_2, output_data_3,
            buf1_L0, buf1_L1, buf1_L2, buf1_L3,
            b_col, m_col, c_col);
        load_to_buf0(value_data, value_data_1, b_col, m_col, c_col + 1,
            buf0_L0, buf0_L1, buf0_L2, buf0_L3);
    }
}

// Long-term loading task across CHANNELS: load ping-pong for c=1..C-1, and send tokens to compute
// (Removed long-term task implementation across CHANNELS, restored to fixed channel-wise parallel)

void msda_top(
    const value_t* value_data,
    const value_t* value_data_1,
    // Group 0 (q % 4 == 0) - Level 0-3
    const ap_uint<512>* sampling_locations_data_0,
    const ap_uint<512>* sampling_locations_data_1,
    const ap_uint<512>* sampling_locations_data_2,
    const ap_uint<512>* sampling_locations_data_3,
    // Group 1 (q % 4 == 1) - Level 0-3
    const ap_uint<512>* sampling_locations_data_4,
    const ap_uint<512>* sampling_locations_data_5,
    const ap_uint<512>* sampling_locations_data_6,
    const ap_uint<512>* sampling_locations_data_7,
    // Group 2 (q % 4 == 2) - Level 0-3
    const ap_uint<512>* sampling_locations_data_8,
    const ap_uint<512>* sampling_locations_data_9,
    const ap_uint<512>* sampling_locations_data_10,
    const ap_uint<512>* sampling_locations_data_11,
    // Group 3 (q % 4 == 3) - Level 0-3
    const ap_uint<512>* sampling_locations_data_12,
    const ap_uint<512>* sampling_locations_data_13,
    const ap_uint<512>* sampling_locations_data_14,
    const ap_uint<512>* sampling_locations_data_15,
    // Group 0 - attention_weights (L0&1, L2&3)
    const ap_uint<512>* attention_weights_data_0, // Levels 0 & 1
    const ap_uint<512>* attention_weights_data_1, // Levels 2 & 3
    // Group 1 - attention_weights
    const ap_uint<512>* attention_weights_data_2, // Levels 0 & 1
    const ap_uint<512>* attention_weights_data_3, // Levels 2 & 3
    // Group 2 - attention_weights
    const ap_uint<512>* attention_weights_data_4, // Levels 0 & 1
    const ap_uint<512>* attention_weights_data_5, // Levels 2 & 3
    // Group 3 - attention_weights
    const ap_uint<512>* attention_weights_data_6, // Levels 0 & 1
    const ap_uint<512>* attention_weights_data_7, // Levels 2 & 3
    // Output: group 0
    output_t* output_data_0,
    // Output: group 1
    output_t* output_data_1,
    // Output: group 2
    output_t* output_data_2,
    // Output: group 3
    output_t* output_data_3
) {
    // ===== M_AXI Interface Setup =====
    // Calculate exact depth based on compile-time constants, avoid TV length mismatch with actual
    const int DEPTH_VALUE = CONST_BATCH_SIZE * CONST_NUM_HEADS * CONST_CHANNELS * CONST_SPATIAL_SIZE;
    const int TOTAL_COORDS = CONST_BATCH_SIZE * CONST_NUM_QUERIES * CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2;
    const int DEPTH_SL    = (TOTAL_COORDS + 15) / 16; // 16 coords per 512b pack
    const int TOTAL_WEIGHTS = CONST_BATCH_SIZE * CONST_NUM_QUERIES * CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS;
    const int DEPTH_AW    = (TOTAL_WEIGHTS + 15) / 16; // 16 weights (2 levels*8 points) per 512b pack
    const int DEPTH_OUT   = CONST_BATCH_SIZE * CONST_NUM_QUERIES * (CONST_NUM_HEADS * CONST_CHANNELS);

    // v1.2.0: 4-query parallel, 16 SL + 8 AW interfaces
    #pragma HLS INTERFACE m_axi port=value_data   offset=slave bundle=Gmem0  depth=DEPTH_VALUE max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=value_data_1 offset=slave bundle=Gmem29 depth=DEPTH_VALUE max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    
    // Group 0 - sampling_locations: 512-bit packs
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_0 offset=slave bundle=Gmem1 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_1 offset=slave bundle=Gmem2 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_2 offset=slave bundle=Gmem3 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_3 offset=slave bundle=Gmem4 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    
    // Group 1 - sampling_locations: 512-bit packs
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_4 offset=slave bundle=Gmem10 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_5 offset=slave bundle=Gmem11 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_6 offset=slave bundle=Gmem12 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_7 offset=slave bundle=Gmem13 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    
    // Group 2 - sampling_locations: 512-bit packs
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_8  offset=slave bundle=Gmem19 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_9  offset=slave bundle=Gmem20 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_10 offset=slave bundle=Gmem21 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_11 offset=slave bundle=Gmem22 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2

    // Group 3 - sampling_locations: 512-bit packs
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_12 offset=slave bundle=Gmem23 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_13 offset=slave bundle=Gmem24 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_14 offset=slave bundle=Gmem25 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=sampling_locations_data_15 offset=slave bundle=Gmem26 depth=DEPTH_SL max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2

    // Group 0 - attention_weights: 512-bit packs (L0&1, L2&3)
    #pragma HLS INTERFACE m_axi port=attention_weights_data_0 offset=slave bundle=Gmem5 depth=DEPTH_AW max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=attention_weights_data_1 offset=slave bundle=Gmem6 depth=DEPTH_AW max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    
    // Group 1 - attention_weights: 512-bit packs (L0&1, L2&3)
    #pragma HLS INTERFACE m_axi port=attention_weights_data_2 offset=slave bundle=Gmem14 depth=DEPTH_AW max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=attention_weights_data_3 offset=slave bundle=Gmem15 depth=DEPTH_AW max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2

    // Group 2 - attention_weights
    #pragma HLS INTERFACE m_axi port=attention_weights_data_4 offset=slave bundle=Gmem7  depth=DEPTH_AW max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=attention_weights_data_5 offset=slave bundle=Gmem8  depth=DEPTH_AW max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2

    // Group 3 - attention_weights
    #pragma HLS INTERFACE m_axi port=attention_weights_data_6 offset=slave bundle=Gmem16 depth=DEPTH_AW max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2
    #pragma HLS INTERFACE m_axi port=attention_weights_data_7 offset=slave bundle=Gmem17 depth=DEPTH_AW max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=2

    // Output interfaces for 4-query parallel (each handles 1/4 of queries)
    #pragma HLS INTERFACE m_axi port=output_data_0 offset=slave bundle=Gmem9  depth=DEPTH_OUT max_write_burst_length=64 max_widen_bitwidth=512 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=output_data_1 offset=slave bundle=Gmem18 depth=DEPTH_OUT max_write_burst_length=64 max_widen_bitwidth=512 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=output_data_2 offset=slave bundle=Gmem27 depth=DEPTH_OUT max_write_burst_length=64 max_widen_bitwidth=512 num_write_outstanding=2
    #pragma HLS INTERFACE m_axi port=output_data_3 offset=slave bundle=Gmem28 depth=DEPTH_OUT max_write_burst_length=64 max_widen_bitwidth=512 num_write_outstanding=2
    
    // Control Interface
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    
    // Read directly from header file constants (no local copy)
    
    int output_index = 0; // Reserved but no longer used, calculate out_idx yourself to maintain [b,q,h,c] output order

    // New data format [batch, head, channel, spatial] - spatial is continuous
    // batch stride = HEAD * CHANNEL * SPATIAL
    const int batch_stride = CONST_NUM_HEADS * CONST_CHANNELS * CONST_SPATIAL_SIZE;
    
    // ===== Define ping-pong buffers at the top of the main function (on-chip BRAM) =====
    // Ping-pong buffers for each LEVEL (Quantized: int16)
    // IMPORTANT: Initialize to 0 to avoid garbage data from out-of-bounds prefetch
    // 16 copies: support 4-query parallel, each group uses 4 copies
    value_t value_buf0_L0[16][16000] = {0};  // group0:0-3, group1:4-7, group2:8-11, group3:12-15
    value_t value_buf0_L1[16][4000] = {0};
    value_t value_buf0_L2[16][1024] = {0};
    value_t value_buf0_L3[16][256] = {0};
    value_t value_buf1_L0[16][16000] = {0};
    value_t value_buf1_L1[16][4000] = {0};
    value_t value_buf1_L2[16][1024] = {0};
    value_t value_buf1_L3[16][256] = {0};
    // Put the largest blocks L0/L1 into URAM to reduce BRAM usage
    // Timing @200MHz: URAM EN/CE nets tend to become long high-fanout cross-SLR routes.
    // Try mapping L0 ping-pong buffers to BRAM to give placer more flexibility and reduce SLR-crossing control nets.
    #pragma HLS BIND_STORAGE variable=value_buf0_L0 type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=value_buf0_L1 type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=value_buf0_L2 type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=value_buf0_L3 type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=value_buf1_L0 type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=value_buf1_L1 type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=value_buf1_L2 type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=value_buf1_L3 type=ram_t2p impl=bram
    // Partition by column to support bilinear interpolation parallel read
    #pragma HLS ARRAY_PARTITION variable=value_buf0_L0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_buf0_L0 cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=value_buf1_L0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_buf1_L0 cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=value_buf0_L1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_buf0_L1 cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=value_buf1_L1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_buf1_L1 cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=value_buf0_L2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_buf0_L2 cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=value_buf1_L2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_buf1_L2 cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=value_buf0_L3 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_buf0_L3 cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=value_buf1_L3 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_buf1_L3 cyclic factor=4 dim=2

    #pragma HLS DEPENDENCE variable=value_buf0_L0 inter false
    #pragma HLS DEPENDENCE variable=value_buf0_L1 inter false
    #pragma HLS DEPENDENCE variable=value_buf0_L2 inter false
    #pragma HLS DEPENDENCE variable=value_buf0_L3 inter false
    #pragma HLS DEPENDENCE variable=value_buf1_L0 inter false
    #pragma HLS DEPENDENCE variable=value_buf1_L1 inter false
    #pragma HLS DEPENDENCE variable=value_buf1_L2 inter false
    #pragma HLS DEPENDENCE variable=value_buf1_L3 inter false

    // Global initialization stage: before entering any loop, prefetch the first frame (b=0,h=0,c=0) to buf0
    value_reader_pingpong_levels(value_data, value_data_1, 0, 0, 0,
        value_buf0_L0, value_buf0_L1, value_buf0_L2, value_buf0_L3,
        value_buf1_L0, value_buf1_L1, value_buf1_L2, value_buf1_L3,
        (ap_uint<1>)0);
    ap_uint<1> prepared_sel = 0; // Buffer selection for the next (b,h) group (0->buf0, 1->buf1)

    // ===== Correct loop order: reference computation_core in the backup =====
    LOOP_BATCH:
    for (int b_col = 0; b_col < CONST_BATCH_SIZE; ++b_col) {
        // Starting offset in the value tensor for this batch (new format: [batch, head, channel, spatial])
        const int data_value_ptr_init_offset = b_col * batch_stride;
        LOOP_NUM_HEADS:
        for (int m_col = 0; m_col < CONST_NUM_HEADS; ++m_col) {
                // Use the previously prepared buffer as the source for the first channel
                ap_uint<1> start_sel = prepared_sel;
                // Loop by channel: parallel "compute current + preload next" within a single step
                LOOP_CHANNELS:
                for (int c_col = 0; c_col < CONST_CHANNELS; ++c_col) {
                    ap_uint<1> cur_sel = (ap_uint<1>)(start_sel ^ (ap_uint<1>)(c_col & 1));
                    // Unified call to df_pair_unified, avoid duplicate hardware
                    df_pair_unified(
                        value_data, value_data_1,
                        // SL groups
                        sampling_locations_data_0, sampling_locations_data_1, sampling_locations_data_2, sampling_locations_data_3,
                        sampling_locations_data_4, sampling_locations_data_5, sampling_locations_data_6, sampling_locations_data_7,
                        attention_weights_data_0, attention_weights_data_1, attention_weights_data_2, attention_weights_data_3,
                        sampling_locations_data_8,  sampling_locations_data_9,  sampling_locations_data_10, sampling_locations_data_11,
                        sampling_locations_data_12, sampling_locations_data_13, sampling_locations_data_14, sampling_locations_data_15,
                        attention_weights_data_4, attention_weights_data_5, attention_weights_data_6, attention_weights_data_7,
                        // outputs
                        output_data_0, output_data_1, output_data_2, output_data_3,
                        b_col, m_col, c_col,
                        value_buf0_L0, value_buf0_L1, value_buf0_L2, value_buf0_L3,
                        value_buf1_L0, value_buf1_L1, value_buf1_L2, value_buf1_L3,
                        cur_sel  // Select which buffer to use at runtime
                    );
                }

                // At channel end: prefetch c=0 to the correct phase buffer for the next (b,m) group
                // This logic is moved here from inside LOOP_CHANNELS to make the loop body uniform.
                int next_b = b_col;
                int next_m = m_col + 1;
                if (next_m == CONST_NUM_HEADS) { next_m = 0; next_b = b_col + 1; }
                if (next_b < CONST_BATCH_SIZE) {
                    ap_uint<1> next_group_sel = (ap_uint<1>)(start_sel ^ (ap_uint<1>)(CONST_CHANNELS & 1));
                    value_reader_pingpong_levels(value_data, value_data_1, next_b, next_m, 0,
                        value_buf0_L0, value_buf0_L1, value_buf0_L2, value_buf0_L3,
                        value_buf1_L0, value_buf1_L1, value_buf1_L2, value_buf1_L3,
                        next_group_sel);
                }
                
                // Update the starting phase of the first channel for the next group
                prepared_sel = (ap_uint<1>)(start_sel ^ (ap_uint<1>)(CONST_CHANNELS & 1));
        } // End of LOOP_NUM_HEADS
    } // End of LOOP_BATCH
}
