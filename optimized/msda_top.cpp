/**
 * @file msda_top.cpp
 * @brief MSDA HLS Implementation - Complete Version
 */

 #include "../INCLUDE/msda_top.h"
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

// Internal feature representation only; HBM remains signed INT8.
using feature_cache_word_t = ap_uint<16>;

template<int BANKS>
static inline value_t read_feature_cache_k2(
    const feature_cache_word_t* cache, const int linear_idx) {
    #pragma HLS INLINE
    static_assert(BANKS == 16 || BANKS == 8 || BANKS == 4,
                  "K2 layout preserves the measured spatial bank factors");
    // Called only under the preserved valid branch: the full signed32 index is >=0.
    static_assert(sizeof(unsigned int) == 4, "K2 cache address conversion retains all32 bits");
    const unsigned int unsigned_idx = static_cast<unsigned int>(linear_idx);
    const unsigned int packed_idx = unsigned_idx % BANKS + BANKS * (unsigned_idx / (2 * BANKS));
    const unsigned int lane = (unsigned_idx / BANKS) % 2U;
    const feature_cache_word_t packed = cache[packed_idx];
    ap_uint<8> raw_bits = packed.range(lane * 8 + 7, lane * 8);
    ap_int<8> signed_byte;
    signed_byte.range(7, 0) = raw_bits;
    return value_t(signed_byte.to_int());
}

// Input: int8 values, interp_t weights, output: sample_t result
inline sample_t bilinear_interpolation(
    const value_t v1, const value_t v2, const value_t v3, const value_t v4,
    const interp_t lh, const interp_t lw) {
    
    #pragma HLS INLINE
    
    const interp_t ilh = interp_t(1.0) - lh;
    const interp_t ilw = interp_t(1.0) - lw;
    const interp_t w1 = interp_t(ilh * ilw);
    const interp_t w2 = interp_t(ilh * lw);
    const interp_t w3 = interp_t(lh * ilw);
    const interp_t w4 = interp_t(lh * lw);
    
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
 
// QUANTIZED VERSION: value_data uses signed INT8 features (64 per 512-bit read)
// Scatter a 512-bit pack into 16 copies of the destination buffer (partial valid_elems supported)
template<int LEN>
static inline void store_pack_to_buf(
    const ap_uint<512>& pack_data,
    const int s_idx_base,
    const int valid_elems,
    feature_cache_word_t (*dst)[LEN / 2]) {
    #pragma HLS INLINE
    static_assert(LEN == 16000 || LEN == 4000 || LEN == 1024 || LEN == 256,
                  "K2 producer is restricted to the measured level lengths");
    constexpr int BANKS = LEN == 16000 ? 16 : (LEN == 4000 ? 8 : 4);
    static_assert(LEN % (2 * BANKS) == 0 && 64 % (2 * BANKS) == 0,
                  "Every loaded pack must contain complete two-lane bank words");
    // s_idx_base is a multiple of64. valid_elems is64, or the complete L1 tail32.
    for (int word_idx = 0; word_idx < 32; ++word_idx) {
        #pragma HLS PIPELINE II=1
        const int bank = word_idx % BANKS;
        const int block = word_idx / BANKS;
        const int byte0 = block * (2 * BANKS) + bank;
        const int byte1 = byte0 + BANKS;
        if (byte1 < valid_elems) {
            feature_cache_word_t packed_word;
            packed_word.range(7, 0) = pack_data.range(8 * (byte0 + 1) - 1, 8 * byte0);
            packed_word.range(15, 8) = pack_data.range(8 * (byte1 + 1) - 1, 8 * byte1);
            const int packed_idx = (s_idx_base / (2 * BANKS)) * BANKS + word_idx;
            for (int copy = 0; copy < 16; ++copy) {
                #pragma HLS UNROLL
                dst[copy][packed_idx] = packed_word;
            }
        }
    }
}

// K: one feature AXI port. Continuous 512-bit reads are decoupled from
// the unchanged replicated-cache writer. INT8 padding and valid tails stay exact.
template<int LEN>
static void read_level_single_axi(
    const value_t* value_data,
    const long long base_linear,
    const int level_offset,
    hls::stream<ap_uint<512>>& packets) {
    #pragma HLS INLINE off
    const ap_uint<512>* src = reinterpret_cast<const ap_uint<512>*>(
        value_data + base_linear + (long long)level_offset);
    constexpr int PACKS = (LEN + 63) / 64;
    READ_LEVEL_PACKETS: for (int packet = 0; packet < PACKS; ++packet) {
        #pragma HLS PIPELINE II=1
        packets.write(src[packet]);
    }
}

template<int LEN>
static void scatter_level_packets(
    hls::stream<ap_uint<512>>& packets,
    feature_cache_word_t (*dst)[LEN / 2]) {
    #pragma HLS INLINE off
    constexpr int PACKS = (LEN + 63) / 64;
    SCATTER_LEVEL_PACKETS: for (int packet = 0; packet < PACKS; ++packet) {
        #pragma HLS PIPELINE II=1
        const int offset = packet * 64;
        const int valid_elems = (LEN - offset < 64) ? LEN - offset : 64;
        const ap_uint<512> data = packets.read();
        store_pack_to_buf<LEN>(data, offset, valid_elems, dst);
    }
}

template<int LEN>
static void load_level_single_axi(
    const value_t* value_data,
    const long long base_linear,
    const int level_offset,
    feature_cache_word_t (*dst)[LEN / 2]) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    hls::stream<ap_uint<512>> packets("feature_packets");
    #pragma HLS STREAM variable=packets depth=64
    read_level_single_axi<LEN>(value_data, base_linear, level_offset, packets);
    scatter_level_packets<LEN>(packets, dst);
}

static void value_reader_pingpong_levels(
    const value_t* value_data_0,  // INT8, single feature AXI port
    const int b_col,
    const int m_col,
    const int c_col,
    feature_cache_word_t buf0_L0[16][8000],
    feature_cache_word_t buf0_L1[16][2000],
    feature_cache_word_t buf0_L2[16][512],
    feature_cache_word_t buf0_L3[16][128],
    feature_cache_word_t buf1_L0[16][8000],
    feature_cache_word_t buf1_L1[16][2000],
    feature_cache_word_t buf1_L2[16][512],
    feature_cache_word_t buf1_L3[16][128],
    const ap_uint<1> use_buf1
) {
    #pragma HLS INLINE off
    if (c_col >= CONST_CHANNELS) return;

    // index = b_col * (HEAD * CHANNEL * SPATIAL) + m_col * (CHANNEL * SPATIAL) + c_col * SPATIAL + spatial_idx
    const long long base_linear =
        (long long)b_col * (long long)CONST_NUM_HEADS * (long long)CONST_CHANNELS * (long long)CONST_SPATIAL_SIZE +
        (long long)m_col * (long long)CONST_CHANNELS * (long long)CONST_SPATIAL_SIZE +
        (long long)c_col * (long long)CONST_SPATIAL_SIZE;

    static_assert(CONST_SPATIAL_SIZE % 64 == 0, "INT8 channel stride must be64-byte aligned");
    static_assert(CONST_LEVEL_START[0] % 64 == 0 && CONST_LEVEL_START[1] % 64 == 0 &&
                  CONST_LEVEL_START[2] % 64 == 0 && CONST_LEVEL_START[3] % 64 == 0,
                  "All INT8 level starts must be64-byte aligned");
    const int len0 = 16000; // Logical cache lengths remain unchanged.
    const int len1 = 4000;  // 40*100 = 125 groups of 32
    const int len2 = 1024;
    const int len3 = 256;

    // K: load each level through the single continuous-address feature port.
    if (use_buf1) {
        load_level_single_axi<16000>(value_data_0, base_linear, 0, buf1_L0);
        load_level_single_axi<4000>(value_data_0, base_linear, CONST_LEVEL_START[1], buf1_L1);
        load_level_single_axi<1024>(value_data_0, base_linear, CONST_LEVEL_START[2], buf1_L2);
        load_level_single_axi<256>(value_data_0, base_linear, CONST_LEVEL_START[3], buf1_L3);
    } else {
        load_level_single_axi<16000>(value_data_0, base_linear, 0, buf0_L0);
        load_level_single_axi<4000>(value_data_0, base_linear, CONST_LEVEL_START[1], buf0_L1);
        load_level_single_axi<1024>(value_data_0, base_linear, CONST_LEVEL_START[2], buf0_L2);
        load_level_single_axi<256>(value_data_0, base_linear, CONST_LEVEL_START[3], buf0_L3);
    }
}

static void load_to_buf0(
    const value_t* value_data,
    const int b_col,
    const int m_col,
    const int c_col,
    feature_cache_word_t buf0_L0[16][8000],
    feature_cache_word_t buf0_L1[16][2000],
    feature_cache_word_t buf0_L2[16][512],
    feature_cache_word_t buf0_L3[16][128]
) {
    #pragma HLS INLINE off
    static feature_cache_word_t dummy1_L0[16][8000];
    static feature_cache_word_t dummy1_L1[16][2000];
    static feature_cache_word_t dummy1_L2[16][512];
    static feature_cache_word_t dummy1_L3[16][128];
    value_reader_pingpong_levels(
        value_data, b_col, m_col, c_col,
        buf0_L0, buf0_L1, buf0_L2, buf0_L3,
        dummy1_L0, dummy1_L1, dummy1_L2, dummy1_L3,
        (ap_uint<1>)0
    );
}

static void load_to_buf1(
    const value_t* value_data,
    const int b_col,
    const int m_col,
    const int c_col,
    feature_cache_word_t buf1_L0[16][8000],
    feature_cache_word_t buf1_L1[16][2000],
    feature_cache_word_t buf1_L2[16][512],
    feature_cache_word_t buf1_L3[16][128]
) {
    #pragma HLS INLINE off
    static feature_cache_word_t dummy0_L0[16][8000];
    static feature_cache_word_t dummy0_L1[16][2000];
    static feature_cache_word_t dummy0_L2[16][512];
    static feature_cache_word_t dummy0_L3[16][128];
    value_reader_pingpong_levels(
        value_data, b_col, m_col, c_col,
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

    hls::stream<ap_uint<3072> > packed_stream_0;
    hls::stream<ap_uint<3072> > packed_stream_1;
    hls::stream<ap_uint<3072> > packed_stream_2;
    hls::stream<ap_uint<3072> > packed_stream_3;
    #pragma HLS STREAM variable=packed_stream_0 depth=2
    #pragma HLS STREAM variable=packed_stream_1 depth=2
    #pragma HLS STREAM variable=packed_stream_2 depth=2
    #pragma HLS STREAM variable=packed_stream_3 depth=2

    fetch_worker(sampling_locations_data_0, attention_weights_data_0, packed_stream_0, b_col, m_col, num_q_w0);
    fetch_worker(sampling_locations_data_1, attention_weights_data_1, packed_stream_1, b_col, m_col, num_q_w1);
    fetch_worker(sampling_locations_data_2, attention_weights_data_2, packed_stream_2, b_col, m_col, num_q_w2);
    fetch_worker(sampling_locations_data_3, attention_weights_data_3, packed_stream_3, b_col, m_col, num_q_w3);
    
    merge_streams(packed_stream_0, packed_stream_1, packed_stream_2, packed_stream_3,
                  metadata_stream, num_q_w0, num_q_w1, num_q_w2, num_q_w3);
}


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
        const int q = q_idx * 2;

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
        const int q = q_idx * 2 + 1;

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
        const int q = q_idx * 2;

        const int loc_base_float_A =
            b_col * CONST_NUM_QUERIES * SL_Q_STRIDE +
            q     * SL_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
            level_idx_A * CONST_NUM_POINTS * 2;
        const int loc_pack_idx_A = loc_base_float_A / 16;
        ap_uint<512> loc_pack_A = sampling_locations_data_A[loc_pack_idx_A];

        const int loc_base_float_B =
            b_col * CONST_NUM_QUERIES * SL_Q_STRIDE +
            q     * SL_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
            level_idx_B * CONST_NUM_POINTS * 2;
        const int loc_pack_idx_B = loc_base_float_B / 16;
        ap_uint<512> loc_pack_B = sampling_locations_data_B[loc_pack_idx_B];

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
        const int q = q_idx * 2 + 1;

        const int loc_base_float_A =
            b_col * CONST_NUM_QUERIES * SL_Q_STRIDE +
            q     * SL_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
            level_idx_A * CONST_NUM_POINTS * 2;
        const int loc_pack_idx_A = loc_base_float_A / 16;
        ap_uint<512> loc_pack_A = sampling_locations_data_A[loc_pack_idx_A];

        const int loc_base_float_B =
            b_col * CONST_NUM_QUERIES * SL_Q_STRIDE +
            q     * SL_Q_STRIDE +
            m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
            level_idx_B * CONST_NUM_POINTS * 2;
        const int loc_pack_idx_B = loc_base_float_B / 16;
        ap_uint<512> loc_pack_B = sampling_locations_data_B[loc_pack_idx_B];

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

    hls::stream<ap_uint<768> > level_stream_0;
    hls::stream<ap_uint<768> > level_stream_1;
    hls::stream<ap_uint<768> > level_stream_2;
    hls::stream<ap_uint<768> > level_stream_3;
    #pragma HLS STREAM variable=level_stream_0 depth=32
    #pragma HLS STREAM variable=level_stream_1 depth=32
    #pragma HLS STREAM variable=level_stream_2 depth=32
    #pragma HLS STREAM variable=level_stream_3 depth=32

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

    hls::stream<ap_uint<768> > level_stream_0;
    hls::stream<ap_uint<768> > level_stream_1;
    hls::stream<ap_uint<768> > level_stream_2;
    hls::stream<ap_uint<768> > level_stream_3;
    #pragma HLS STREAM variable=level_stream_0 depth=32
    #pragma HLS STREAM variable=level_stream_1 depth=32
    #pragma HLS STREAM variable=level_stream_2 depth=32
    #pragma HLS STREAM variable=level_stream_3 depth=32

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

// #ifndef __SYNTHESIS__
// static float g_weight_min = 1e10, g_weight_max = -1e10;
// static float g_sample_min = 1e10, g_sample_max = -1e10;
// static float g_weighted_min = 1e10, g_weighted_max = -1e10;
// static float g_point_sum_min = 1e10, g_point_sum_max = -1e10;
// static float g_accum_val_min = 1e10, g_accum_val_max = -1e10;
// static int g_sample_count = 0;
// static int g_accum_count = 0;
// #endif

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
                ap_uint<27> weight_bits = word.range(26, 0);
                weight_t weight_val;
                std::memcpy(&weight_val, &weight_bits, sizeof(weight_t));
                weight_arr[i] = weight_val;
            }

            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = CONST_SPATIAL_SHAPES[spatial_h_ptr];
            const int spatial_w = CONST_SPATIAL_SHAPES[spatial_h_ptr + 1];
            
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

                ap_int<8> sampled_value_int8 = 0;
                if (valid) {
                    sampled_value_int8 = cur0[lin_idx];
                }

                // 8-bit * 27-bit fits perfectly in 1 DSP48E2.
                // 8 integer bits + (8 integer + 19 fractional) bits = 16 integer + 19 fractional = 35 bits total.
                ap_fixed<35, 19> weighted_sample = sampled_value_int8 * weight;
                #pragma HLS BIND_OP variable=weighted_sample op=mul impl=dsp
                
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

        const int out_idx =
            b_col * (NUM_QUERIES_HALF * CONST_NUM_HEADS * CONST_CHANNELS) +
            query_idx * (CONST_NUM_HEADS * CONST_CHANNELS) +
            m_col * CONST_CHANNELS +
            c_col;
        output_data[out_idx] = accumulated_value;
    }
}

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
                ap_uint<27> weight_bits = word.range(26, 0);
                weight_t weight_val;
                std::memcpy(&weight_val, &weight_bits, sizeof(weight_t));
                weight_arr[i] = weight_val;
            }

            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = CONST_SPATIAL_SHAPES[spatial_h_ptr];
            const int spatial_w = CONST_SPATIAL_SHAPES[spatial_h_ptr + 1];
            
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

                ap_int<8> sampled_value_int8 = 0;
                if (valid) {
                    sampled_value_int8 = cur0[lin_idx];
                }

                // 8-bit * 27-bit fits perfectly in 1 DSP48E2.
                // 8 integer bits + (8 integer + 19 fractional) bits = 16 integer + 19 fractional = 35 bits total.
                ap_fixed<35, 19> weighted_sample = sampled_value_int8 * weight;
                #pragma HLS BIND_OP variable=weighted_sample op=mul impl=dsp
                
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
    const int NUM_QUERIES_GROUP = (CONST_NUM_QUERIES + 3) >> 2; // ceil(Q/4)

    const ap_uint<512>* sl_ptrs[4];
    #pragma HLS ARRAY_PARTITION variable=sl_ptrs complete
    sl_ptrs[0] = sampling_locations_data_0;
    sl_ptrs[1] = sampling_locations_data_1;
    sl_ptrs[2] = sampling_locations_data_2;
    sl_ptrs[3] = sampling_locations_data_3;

    LOOP_QUERIES_FETCH_GROUP:
    for (int q_idx = 0; q_idx < NUM_QUERIES_GROUP; ++q_idx) {
        #pragma HLS PIPELINE II=1
        const int q = group_mod + (q_idx << 2);
        if (q >= CONST_NUM_QUERIES) break;

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

// Raw query metadata is immutable across all channels for one (view, head).
static const int METADATA_QUERIES_PER_GROUP = (CONST_NUM_QUERIES + 3) >> 2;

// Channel-invariant point descriptors: int32 index +1 valid +27 raw weight.
struct nearest_weight_point_t {
    int lin_idx;
    int valid;
    weight_t weight;
};
static_assert(sizeof(int) == 4, "The original lin_idx representation must remain32 bits");

static ap_uint<1920> precompute_nearest_weight_metadata(const ap_uint<3072>& packed_metadata) {
    #pragma HLS INLINE
    ap_uint<1920> cached_metadata;
    int bit_offset = 0;
    PRECOMPUTE_LEVELS:
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
                ap_uint<27> weight_bits = word.range(26, 0);
                weight_t weight_val;
                std::memcpy(&weight_val, &weight_bits, sizeof(weight_t));
                weight_arr[i] = weight_val;
            }

            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = CONST_SPATIAL_SHAPES[spatial_h_ptr];
            const int spatial_w = CONST_SPATIAL_SHAPES[spatial_h_ptr + 1];

            PRECOMPUTE_POINTS:
            for (int p_col = 0; p_col < CONST_NUM_POINTS; ++p_col) {
                #pragma HLS UNROLL
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
                const int cache_offset = (l_col * CONST_NUM_POINTS + p_col) * 60;
                cached_metadata.range(cache_offset + 31, cache_offset) = ap_uint<32>(lin_idx);
                cached_metadata[cache_offset + 32] = valid;
                cached_metadata.range(cache_offset + 59, cache_offset + 33) = weight.range(26, 0);
            }
    }
    return cached_metadata;
}

static nearest_weight_point_t decode_cached_nearest_weight_point(
    const ap_uint<1920>& cached_metadata, const int point_slot) {
    #pragma HLS INLINE
    const int cache_offset = point_slot * 60;
    ap_uint<32> index_bits = cached_metadata.range(cache_offset + 31, cache_offset);
    ap_int<32> signed_index;
    signed_index.range(31, 0) = index_bits;
    nearest_weight_point_t point;
    point.lin_idx = signed_index.to_int();
    ap_uint<1> valid_bit = cached_metadata[cache_offset + 32];
    point.valid = valid_bit.to_uint();
    ap_uint<27> weight_bits = cached_metadata.range(cache_offset + 59, cache_offset + 33);
    std::memcpy(&point.weight, &weight_bits, sizeof(weight_t));
    return point;
}

static void fill_metadata_cache_group(
    const ap_uint<512>* sampling_locations_data_0,
    const ap_uint<512>* sampling_locations_data_1,
    const ap_uint<512>* sampling_locations_data_2,
    const ap_uint<512>* sampling_locations_data_3,
    const ap_uint<512>* attention_weights_data_0,
    const ap_uint<512>* attention_weights_data_1,
    ap_uint<1920> metadata_cache[METADATA_QUERIES_PER_GROUP],
    const int b_col,
    const int m_col,
    const int group_mod)  // 0,1,2,3
{
    #pragma HLS INLINE off

    const int SL_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2;
    const int AW_Q_STRIDE = CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS;
    const int NUM_QUERIES_GROUP = (CONST_NUM_QUERIES + 3) >> 2; // ceil(Q/4)

    const ap_uint<512>* sl_ptrs[4];
    #pragma HLS ARRAY_PARTITION variable=sl_ptrs complete
    sl_ptrs[0] = sampling_locations_data_0;
    sl_ptrs[1] = sampling_locations_data_1;
    sl_ptrs[2] = sampling_locations_data_2;
    sl_ptrs[3] = sampling_locations_data_3;

    LOOP_QUERIES_FETCH_GROUP:
    for (int q_idx = 0; q_idx < NUM_QUERIES_GROUP; ++q_idx) {
        #pragma HLS PIPELINE II=1
        const int q = group_mod + (q_idx << 2);
        if (q >= CONST_NUM_QUERIES) break;

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
            const int loc_pack_idx =
                (b_col * CONST_NUM_HEADS + m_col) * NUM_QUERIES_GROUP + q_idx;
            ap_uint<512> loc_pack = sl_ptr[loc_pack_idx];

            const ap_uint<512>* aw_pair_ptr;
            int base_level;
            if (l_col < 2) {
                aw_pair_ptr = attention_weights_data_0;
                base_level = 0;
            } else {
                aw_pair_ptr = attention_weights_data_1;
                base_level = 2;
            }
            const int weight_pack_idx =
                (b_col * CONST_NUM_HEADS + m_col) * NUM_QUERIES_GROUP + q_idx;
            ap_uint<512> w_pack_512 = aw_pair_ptr[weight_pack_idx];
            ap_uint<256> w_pack = ( (l_col & 1) == 0 )
                                  ? w_pack_512.range(255, 0)
                                  : w_pack_512.range(511, 256);

            packed_metadata.range(bit_offset + 511, bit_offset) = loc_pack;
            bit_offset += 512;
            packed_metadata.range(bit_offset + 255, bit_offset) = w_pack;
            bit_offset += 256;
        }

        metadata_cache[q_idx] = precompute_nearest_weight_metadata(packed_metadata);
    }
}

static void replay_metadata_cache_group(
    const ap_uint<1920> metadata_cache[METADATA_QUERIES_PER_GROUP],
    hls::stream<ap_uint<1920> >& metadata_stream,
    const int group_mod) {
    #pragma HLS INLINE off
    LOOP_QUERIES_REPLAY_GROUP:
    for (int q_idx = 0; q_idx < METADATA_QUERIES_PER_GROUP; ++q_idx) {
        #pragma HLS PIPELINE II=1
        const int q = group_mod + (q_idx << 2);
        if (q >= CONST_NUM_QUERIES) break;
        metadata_stream.write(metadata_cache[q_idx]);
    }
}

static void fill_metadata_cache_4q(
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
    ap_uint<1920> metadata_cache_0[METADATA_QUERIES_PER_GROUP],
    ap_uint<1920> metadata_cache_1[METADATA_QUERIES_PER_GROUP],
    ap_uint<1920> metadata_cache_2[METADATA_QUERIES_PER_GROUP],
    ap_uint<1920> metadata_cache_3[METADATA_QUERIES_PER_GROUP],
    const int b_col,
    const int m_col
) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    fill_metadata_cache_group(
        sampling_locations_data_0, sampling_locations_data_1, sampling_locations_data_2, sampling_locations_data_3,
        attention_weights_data_0, attention_weights_data_1,
        metadata_cache_0, b_col, m_col, 0);
    fill_metadata_cache_group(
        sampling_locations_data_4, sampling_locations_data_5, sampling_locations_data_6, sampling_locations_data_7,
        attention_weights_data_2, attention_weights_data_3,
        metadata_cache_1, b_col, m_col, 1);
    fill_metadata_cache_group(
        sampling_locations_data_8, sampling_locations_data_9, sampling_locations_data_10, sampling_locations_data_11,
        attention_weights_data_4, attention_weights_data_5,
        metadata_cache_2, b_col, m_col, 2);
    fill_metadata_cache_group(
        sampling_locations_data_12, sampling_locations_data_13, sampling_locations_data_14, sampling_locations_data_15,
        attention_weights_data_6, attention_weights_data_7,
        metadata_cache_3, b_col, m_col, 3);
}


static const int OUTPUT_TILE_BANKS = 8;
static const int OUTPUT_WORDS_PER_QUERY = CONST_CHANNELS / OUTPUT_TILE_BANKS;
static const int OUTPUT_GROUP_QUERIES = (CONST_NUM_QUERIES + 3) >> 2;
static const int OUTPUT_TILE_DEPTH = OUTPUT_GROUP_QUERIES * OUTPUT_WORDS_PER_QUERY;
static_assert(CONST_CHANNELS == 32, "Output packing requires all 32 channels");
static_assert(CONST_NUM_QUERIES == 5000 && (CONST_NUM_QUERIES % 4) == 0,
              "Measured P=4 output packing requires 5000 queries/view");
static_assert(CONST_BATCH_SIZE == 6 && CONST_NUM_HEADS == 8,
              "Measured output packing requires six views/eight heads");

static void execute_computation_group(
    hls::stream<ap_uint<1920> >& metadata_stream,
    ap_uint<48> output_tile_group[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    const feature_cache_word_t cur_L0[16][8000], const feature_cache_word_t cur_L1[16][2000], const feature_cache_word_t cur_L2[16][512], const feature_cache_word_t cur_L3[16][128],
    const int b_col,
    const int m_col,
    const int c_col,
    const int base_copy)  // 0,4,8,12
{
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=output_tile_group complete dim=1
    const int NUM_QUERIES_GROUP = (CONST_NUM_QUERIES + 3) >> 2; // ceil(Q/4)

    LOOP_NUM_QUERIES_GROUP:
    for (int query_idx = 0; query_idx < NUM_QUERIES_GROUP; ++query_idx) {
        #pragma HLS PIPELINE II=1
        const int q_global = (query_idx << 2) + (base_copy >> 2); // base_copy=0/4/8/12 => group_mod
        if (q_global >= CONST_NUM_QUERIES) break;

        accum_t accumulated_value = 0;
        ap_uint<1920> packed_metadata = metadata_stream.read();

        LOOP_LEVELS_GROUP:
        for (int l_col = 0; l_col < CONST_NUM_LEVELS; ++l_col) {
            #pragma HLS UNROLL

            const feature_cache_word_t* cur_buffers[4];
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
                const feature_cache_word_t* cur0 = cur_buffers[p_col % 4];

                const nearest_weight_point_t metadata_point = decode_cached_nearest_weight_point(
                    packed_metadata, l_col * CONST_NUM_POINTS + p_col);
                const weight_t weight = metadata_point.weight;
                const int valid = metadata_point.valid;
                const int lin_idx = metadata_point.lin_idx;

                value_t sampled_value_int16 = 0;
                if (valid) {
                    sampled_value_int16 = (l_col == 0) ? read_feature_cache_k2<16>(cur0, lin_idx) :
                                          (l_col == 1) ? read_feature_cache_k2<8>(cur0, lin_idx) :
                                                         read_feature_cache_k2<4>(cur0, lin_idx);
                }

                ap_fixed<35, 16, AP_TRN, AP_WRAP> weighted_product = ap_int<8>(sampled_value_int16) * weight;
                #pragma HLS BIND_OP variable=weighted_product op=mul impl=dsp
                const sample_t weighted_sample = sample_t(weighted_product);

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

        // Bank c%8, address q*4+c/8; every head writes all 32 channels before flush.
        output_tile_group[c_col & 7]
                         [query_idx * OUTPUT_WORDS_PER_QUERY + (c_col >> 3)] =
            accumulated_value.range(47, 0);
    }
}

static void flush_output_group(
    const ap_uint<48> output_tile_group[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<512>* output_data_group,
    const int b_col,
    const int m_col)
{
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=output_tile_group complete dim=1
    OUTPUT_FLUSH_QUERIES:
    for (int query_idx = 0; query_idx < OUTPUT_GROUP_QUERIES; ++query_idx) {
        const int base_word =
            b_col * (OUTPUT_GROUP_QUERIES * CONST_NUM_HEADS * OUTPUT_WORDS_PER_QUERY) +
            query_idx * (CONST_NUM_HEADS * OUTPUT_WORDS_PER_QUERY) +
            m_col * OUTPUT_WORDS_PER_QUERY;
        OUTPUT_FLUSH_BURST4:
        for (int word_idx = 0; word_idx < OUTPUT_WORDS_PER_QUERY; ++word_idx) {
            #pragma HLS PIPELINE II=1
            ap_uint<512> packed_output = 0;
            OUTPUT_FLUSH_LANES:
            for (int lane = 0; lane < OUTPUT_TILE_BANKS; ++lane) {
                #pragma HLS UNROLL
                // Bits [63:48] in each original eight-byte ABI slot stay zero.
                packed_output.range(lane * 64 + 47, lane * 64) =
                    output_tile_group[lane]
                                     [query_idx * OUTPUT_WORDS_PER_QUERY + word_idx];
            }
            output_data_group[base_word + word_idx] = packed_output;
        }
    }
}

static void flush_output_head(
    const ap_uint<48> output_tile_0[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    const ap_uint<48> output_tile_1[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    const ap_uint<48> output_tile_2[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    const ap_uint<48> output_tile_3[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<512>* output_data_0,
    ap_uint<512>* output_data_1,
    ap_uint<512>* output_data_2,
    ap_uint<512>* output_data_3,
    const int b_col,
    const int m_col)
{
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    flush_output_group(output_tile_0, output_data_0, b_col, m_col);
    flush_output_group(output_tile_1, output_data_1, b_col, m_col);
    flush_output_group(output_tile_2, output_data_2, b_col, m_col);
    flush_output_group(output_tile_3, output_data_3, b_col, m_col);
}

static void compute_channel_4q(
    const ap_uint<1920> metadata_cache_0[METADATA_QUERIES_PER_GROUP],
    const ap_uint<1920> metadata_cache_1[METADATA_QUERIES_PER_GROUP],
    const ap_uint<1920> metadata_cache_2[METADATA_QUERIES_PER_GROUP],
    const ap_uint<1920> metadata_cache_3[METADATA_QUERIES_PER_GROUP],
    // outputs
    ap_uint<48> output_tile_0[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_1[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_2[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_3[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    const feature_cache_word_t cur_L0[16][8000], const feature_cache_word_t cur_L1[16][2000], const feature_cache_word_t cur_L2[16][512], const feature_cache_word_t cur_L3[16][128],
    const int b_col,
    const int m_col,
    const int c_col
) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    hls::stream<ap_uint<1920> > metadata_stream_0;
    hls::stream<ap_uint<1920> > metadata_stream_1;
    hls::stream<ap_uint<1920> > metadata_stream_2;
    hls::stream<ap_uint<1920> > metadata_stream_3;
    #pragma HLS STREAM variable=metadata_stream_0 depth=4
    #pragma HLS STREAM variable=metadata_stream_1 depth=4
    #pragma HLS STREAM variable=metadata_stream_2 depth=4
    #pragma HLS STREAM variable=metadata_stream_3 depth=4

    // Replay the same raw query words for each channel; no SL/AW AXI access.
    replay_metadata_cache_group(metadata_cache_0, metadata_stream_0, 0);
    replay_metadata_cache_group(metadata_cache_1, metadata_stream_1, 1);
    replay_metadata_cache_group(metadata_cache_2, metadata_stream_2, 2);
    replay_metadata_cache_group(metadata_cache_3, metadata_stream_3, 3);

    execute_computation_group(metadata_stream_0, output_tile_0,
        cur_L0, cur_L1, cur_L2, cur_L3,
        b_col, m_col, c_col,
        0);   // copies 0-3
    execute_computation_group(metadata_stream_1, output_tile_1,
        cur_L0, cur_L1, cur_L2, cur_L3,
        b_col, m_col, c_col,
        4);   // copies 4-7
    execute_computation_group(metadata_stream_2, output_tile_2,
        cur_L0, cur_L1, cur_L2, cur_L3,
        b_col, m_col, c_col,
        8);   // copies 8-11
    execute_computation_group(metadata_stream_3, output_tile_3,
        cur_L0, cur_L1, cur_L2, cur_L3,
        b_col, m_col, c_col,
        12);  // copies 12-15
}


static void df_pair_unified(
    const value_t* value_data,
    const ap_uint<1920> metadata_cache_0[METADATA_QUERIES_PER_GROUP],
    const ap_uint<1920> metadata_cache_1[METADATA_QUERIES_PER_GROUP],
    const ap_uint<1920> metadata_cache_2[METADATA_QUERIES_PER_GROUP],
    const ap_uint<1920> metadata_cache_3[METADATA_QUERIES_PER_GROUP],
    ap_uint<48> output_tile_0[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_1[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_2[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_3[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    const int b_col,
    const int m_col,
    const int c_col,
    feature_cache_word_t buf0_L0[16][8000], feature_cache_word_t buf0_L1[16][2000], feature_cache_word_t buf0_L2[16][512], feature_cache_word_t buf0_L3[16][128],
    feature_cache_word_t buf1_L0[16][8000], feature_cache_word_t buf1_L1[16][2000], feature_cache_word_t buf1_L2[16][512], feature_cache_word_t buf1_L3[16][128],
    const ap_uint<1> use_buf1
) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    
    if (use_buf1 == (ap_uint<1>)0) {
        compute_channel_4q(
            metadata_cache_0, metadata_cache_1, metadata_cache_2, metadata_cache_3,
            // outputs
            output_tile_0, output_tile_1, output_tile_2, output_tile_3,
            buf0_L0, buf0_L1, buf0_L2, buf0_L3,
            b_col, m_col, c_col);
        load_to_buf1(value_data, b_col, m_col, c_col + 1,
            buf1_L0, buf1_L1, buf1_L2, buf1_L3);
    } else {
        compute_channel_4q(
            metadata_cache_0, metadata_cache_1, metadata_cache_2, metadata_cache_3,
            // outputs
            output_tile_0, output_tile_1, output_tile_2, output_tile_3,
            buf1_L0, buf1_L1, buf1_L2, buf1_L3,
            b_col, m_col, c_col);
        load_to_buf0(value_data, b_col, m_col, c_col + 1,
            buf0_L0, buf0_L1, buf0_L2, buf0_L3);
    }
}



// H control only: two raw48 output tiles, one current-head compute actor.
static void prepare_compute_current_head(
    const value_t* value_data,
    const ap_uint<512>* sampling_locations_data_0,
    const ap_uint<512>* sampling_locations_data_1,
    const ap_uint<512>* sampling_locations_data_2,
    const ap_uint<512>* sampling_locations_data_3,
    const ap_uint<512>* sampling_locations_data_4,
    const ap_uint<512>* sampling_locations_data_5,
    const ap_uint<512>* sampling_locations_data_6,
    const ap_uint<512>* sampling_locations_data_7,
    const ap_uint<512>* sampling_locations_data_8,
    const ap_uint<512>* sampling_locations_data_9,
    const ap_uint<512>* sampling_locations_data_10,
    const ap_uint<512>* sampling_locations_data_11,
    const ap_uint<512>* sampling_locations_data_12,
    const ap_uint<512>* sampling_locations_data_13,
    const ap_uint<512>* sampling_locations_data_14,
    const ap_uint<512>* sampling_locations_data_15,
    const ap_uint<512>* attention_weights_data_0,
    const ap_uint<512>* attention_weights_data_1,
    const ap_uint<512>* attention_weights_data_2,
    const ap_uint<512>* attention_weights_data_3,
    const ap_uint<512>* attention_weights_data_4,
    const ap_uint<512>* attention_weights_data_5,
    const ap_uint<512>* attention_weights_data_6,
    const ap_uint<512>* attention_weights_data_7,
    ap_uint<1920> metadata_cache_0[METADATA_QUERIES_PER_GROUP],
    ap_uint<1920> metadata_cache_1[METADATA_QUERIES_PER_GROUP],
    ap_uint<1920> metadata_cache_2[METADATA_QUERIES_PER_GROUP],
    ap_uint<1920> metadata_cache_3[METADATA_QUERIES_PER_GROUP],
    ap_uint<48> output_tile_0[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_1[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_2[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_3[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    feature_cache_word_t value_buf0_L0[16][8000],
    feature_cache_word_t value_buf0_L1[16][2000],
    feature_cache_word_t value_buf0_L2[16][512],
    feature_cache_word_t value_buf0_L3[16][128],
    feature_cache_word_t value_buf1_L0[16][8000],
    feature_cache_word_t value_buf1_L1[16][2000],
    feature_cache_word_t value_buf1_L2[16][512],
    feature_cache_word_t value_buf1_L3[16][128],
    const int b_col,
    const int m_col,
    const bool current_valid
) {
    #pragma HLS INLINE off
    static_assert(CONST_CHANNELS == 32 && (CONST_CHANNELS & 1) == 0, "H phase proof requires the frozen 32-channel configuration");
    if (!current_valid) return;
    fill_metadata_cache_4q(
        sampling_locations_data_0, sampling_locations_data_1, sampling_locations_data_2, sampling_locations_data_3, sampling_locations_data_4, sampling_locations_data_5, sampling_locations_data_6, sampling_locations_data_7, sampling_locations_data_8, sampling_locations_data_9, sampling_locations_data_10, sampling_locations_data_11, sampling_locations_data_12, sampling_locations_data_13, sampling_locations_data_14, sampling_locations_data_15, attention_weights_data_0, attention_weights_data_1, attention_weights_data_2, attention_weights_data_3, attention_weights_data_4, attention_weights_data_5, attention_weights_data_6, attention_weights_data_7, metadata_cache_0, metadata_cache_1, metadata_cache_2, metadata_cache_3, b_col, m_col
    );
    value_reader_pingpong_levels(
        value_data, b_col, m_col, 0, value_buf0_L0, value_buf0_L1, value_buf0_L2, value_buf0_L3, value_buf1_L0, value_buf1_L1, value_buf1_L2, value_buf1_L3, (ap_uint<1>)0
    );
    const ap_uint<1> start_sel = 0; // G prepared_sel remains zero for all C=32 heads.
                LOOP_CHANNELS:
                for (int c_col = 0; c_col < CONST_CHANNELS; ++c_col) {
                    ap_uint<1> cur_sel = (ap_uint<1>)(start_sel ^ (ap_uint<1>)(c_col & 1));
                    df_pair_unified(
                        value_data,
                        metadata_cache_0, metadata_cache_1, metadata_cache_2, metadata_cache_3,
                        // outputs
                        output_tile_0, output_tile_1, output_tile_2, output_tile_3,
                        b_col, m_col, c_col,
                        value_buf0_L0, value_buf0_L1, value_buf0_L2, value_buf0_L3,
                        value_buf1_L0, value_buf1_L1, value_buf1_L2, value_buf1_L3,
                        cur_sel
                    );
                }
}

static void flush_previous_head_if_valid(
    ap_uint<48> output_tile_0[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_1[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_2[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_3[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<512>* output_data_0,
    ap_uint<512>* output_data_1,
    ap_uint<512>* output_data_2,
    ap_uint<512>* output_data_3,
    const int b_col,
    const int m_col,
    const bool previous_valid
) {
    #pragma HLS INLINE off
    if (!previous_valid) return;
    flush_output_head(
        output_tile_0, output_tile_1, output_tile_2, output_tile_3, output_data_0, output_data_1, output_data_2, output_data_3, b_col, m_col
    );
}

static void head_stage_unified(
    const value_t* value_data,
    const ap_uint<512>* sampling_locations_data_0,
    const ap_uint<512>* sampling_locations_data_1,
    const ap_uint<512>* sampling_locations_data_2,
    const ap_uint<512>* sampling_locations_data_3,
    const ap_uint<512>* sampling_locations_data_4,
    const ap_uint<512>* sampling_locations_data_5,
    const ap_uint<512>* sampling_locations_data_6,
    const ap_uint<512>* sampling_locations_data_7,
    const ap_uint<512>* sampling_locations_data_8,
    const ap_uint<512>* sampling_locations_data_9,
    const ap_uint<512>* sampling_locations_data_10,
    const ap_uint<512>* sampling_locations_data_11,
    const ap_uint<512>* sampling_locations_data_12,
    const ap_uint<512>* sampling_locations_data_13,
    const ap_uint<512>* sampling_locations_data_14,
    const ap_uint<512>* sampling_locations_data_15,
    const ap_uint<512>* attention_weights_data_0,
    const ap_uint<512>* attention_weights_data_1,
    const ap_uint<512>* attention_weights_data_2,
    const ap_uint<512>* attention_weights_data_3,
    const ap_uint<512>* attention_weights_data_4,
    const ap_uint<512>* attention_weights_data_5,
    const ap_uint<512>* attention_weights_data_6,
    const ap_uint<512>* attention_weights_data_7,
    ap_uint<1920> metadata_cache_0[METADATA_QUERIES_PER_GROUP],
    ap_uint<1920> metadata_cache_1[METADATA_QUERIES_PER_GROUP],
    ap_uint<1920> metadata_cache_2[METADATA_QUERIES_PER_GROUP],
    ap_uint<1920> metadata_cache_3[METADATA_QUERIES_PER_GROUP],
    ap_uint<48> output_tile_0[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_1[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_2[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_3[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_b_0[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_b_1[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_b_2[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<48> output_tile_b_3[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH],
    ap_uint<512>* output_data_0,
    ap_uint<512>* output_data_1,
    ap_uint<512>* output_data_2,
    ap_uint<512>* output_data_3,
    feature_cache_word_t value_buf0_L0[16][8000],
    feature_cache_word_t value_buf0_L1[16][2000],
    feature_cache_word_t value_buf0_L2[16][512],
    feature_cache_word_t value_buf0_L3[16][128],
    feature_cache_word_t value_buf1_L0[16][8000],
    feature_cache_word_t value_buf1_L1[16][2000],
    feature_cache_word_t value_buf1_L2[16][512],
    feature_cache_word_t value_buf1_L3[16][128],
    const int stage
) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION instances=prepare_compute_current_head limit=1 function
    #pragma HLS ALLOCATION instances=flush_previous_head_if_valid limit=1 function
    #pragma HLS DATAFLOW
    const int head_count = CONST_BATCH_SIZE * CONST_NUM_HEADS;
    const bool current_valid = stage < head_count;
    const bool previous_valid = stage > 0;
    const int current_head = current_valid ? stage : head_count - 1;
    const int previous_head = previous_valid ? stage - 1 : 0;
    const int current_b = current_head / CONST_NUM_HEADS;
    const int current_m = current_head % CONST_NUM_HEADS;
    const int previous_b = previous_head / CONST_NUM_HEADS;
    const int previous_m = previous_head % CONST_NUM_HEADS;
    // Named opposite physical arrays in each mutually exclusive branch.
    // This function returns only after BOTH selected tasks complete.
    if ((stage & 1) == 0) {
        flush_previous_head_if_valid(
            output_tile_b_0, output_tile_b_1, output_tile_b_2, output_tile_b_3, output_data_0, output_data_1, output_data_2, output_data_3, previous_b, previous_m, previous_valid
        );
        prepare_compute_current_head(
            value_data, sampling_locations_data_0, sampling_locations_data_1, sampling_locations_data_2, sampling_locations_data_3, sampling_locations_data_4, sampling_locations_data_5, sampling_locations_data_6, sampling_locations_data_7, sampling_locations_data_8, sampling_locations_data_9, sampling_locations_data_10, sampling_locations_data_11, sampling_locations_data_12, sampling_locations_data_13, sampling_locations_data_14, sampling_locations_data_15, attention_weights_data_0, attention_weights_data_1, attention_weights_data_2, attention_weights_data_3, attention_weights_data_4, attention_weights_data_5, attention_weights_data_6, attention_weights_data_7, metadata_cache_0, metadata_cache_1, metadata_cache_2, metadata_cache_3, output_tile_0, output_tile_1, output_tile_2, output_tile_3, value_buf0_L0, value_buf0_L1, value_buf0_L2, value_buf0_L3, value_buf1_L0, value_buf1_L1, value_buf1_L2, value_buf1_L3, current_b, current_m, current_valid
        );
    } else {
        flush_previous_head_if_valid(
            output_tile_0, output_tile_1, output_tile_2, output_tile_3, output_data_0, output_data_1, output_data_2, output_data_3, previous_b, previous_m, previous_valid
        );
        prepare_compute_current_head(
            value_data, sampling_locations_data_0, sampling_locations_data_1, sampling_locations_data_2, sampling_locations_data_3, sampling_locations_data_4, sampling_locations_data_5, sampling_locations_data_6, sampling_locations_data_7, sampling_locations_data_8, sampling_locations_data_9, sampling_locations_data_10, sampling_locations_data_11, sampling_locations_data_12, sampling_locations_data_13, sampling_locations_data_14, sampling_locations_data_15, attention_weights_data_0, attention_weights_data_1, attention_weights_data_2, attention_weights_data_3, attention_weights_data_4, attention_weights_data_5, attention_weights_data_6, attention_weights_data_7, metadata_cache_0, metadata_cache_1, metadata_cache_2, metadata_cache_3, output_tile_b_0, output_tile_b_1, output_tile_b_2, output_tile_b_3, value_buf0_L0, value_buf0_L1, value_buf0_L2, value_buf0_L3, value_buf1_L0, value_buf1_L1, value_buf1_L2, value_buf1_L3, current_b, current_m, current_valid
        );
    }
}

extern "C" void msda_top(
    const value_t* value_data,
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
    ap_uint<512>* output_data_0,
    // Output: group 1
    ap_uint<512>* output_data_1,
    // Output: group 2
    ap_uint<512>* output_data_2,
    // Output: group 3
    ap_uint<512>* output_data_3
) {
    // ===== M_AXI Interface Setup =====
    const int DEPTH_VALUE = CONST_BATCH_SIZE * CONST_NUM_HEADS * CONST_CHANNELS * CONST_SPATIAL_SIZE;
    const int TOTAL_COORDS = CONST_BATCH_SIZE * CONST_NUM_QUERIES * CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2;
    const int DEPTH_SL    = CONST_BATCH_SIZE * CONST_NUM_HEADS * METADATA_QUERIES_PER_GROUP; // compact [view][head][q_idx] words
    const int TOTAL_WEIGHTS = CONST_BATCH_SIZE * CONST_NUM_QUERIES * CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS;
    const int DEPTH_AW    = CONST_BATCH_SIZE * CONST_NUM_HEADS * METADATA_QUERIES_PER_GROUP; // compact [view][head][q_idx] words
    const int DEPTH_OUT   = CONST_BATCH_SIZE * OUTPUT_GROUP_QUERIES * CONST_NUM_HEADS * OUTPUT_WORDS_PER_QUERY; // 512-bit words per group

    // v1.2.0: 4-query parallel, 16 SL + 8 AW interfaces
    #pragma HLS INTERFACE m_axi port=value_data   offset=slave bundle=Gmem0  depth=DEPTH_VALUE max_read_burst_length=64 max_widen_bitwidth=512 num_read_outstanding=8
    
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
    #pragma HLS INTERFACE m_axi port=output_data_0 offset=slave bundle=Gmem9  depth=DEPTH_OUT max_write_burst_length=4 max_widen_bitwidth=512 num_write_outstanding=64
    #pragma HLS INTERFACE m_axi port=output_data_1 offset=slave bundle=Gmem18 depth=DEPTH_OUT max_write_burst_length=4 max_widen_bitwidth=512 num_write_outstanding=64
    #pragma HLS INTERFACE m_axi port=output_data_2 offset=slave bundle=Gmem27 depth=DEPTH_OUT max_write_burst_length=4 max_widen_bitwidth=512 num_write_outstanding=64
    #pragma HLS INTERFACE m_axi port=output_data_3 offset=slave bundle=Gmem28 depth=DEPTH_OUT max_write_burst_length=4 max_widen_bitwidth=512 num_write_outstanding=64
    
    // Control Interface
        #pragma HLS INTERFACE s_axilite port=value_data bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_0 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_1 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_2 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_3 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_4 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_5 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_6 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_7 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_8 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_9 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_10 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_11 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_12 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_13 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_14 bundle=control
    #pragma HLS INTERFACE s_axilite port=sampling_locations_data_15 bundle=control
    #pragma HLS INTERFACE s_axilite port=attention_weights_data_0 bundle=control
    #pragma HLS INTERFACE s_axilite port=attention_weights_data_1 bundle=control
    #pragma HLS INTERFACE s_axilite port=attention_weights_data_2 bundle=control
    #pragma HLS INTERFACE s_axilite port=attention_weights_data_3 bundle=control
    #pragma HLS INTERFACE s_axilite port=attention_weights_data_4 bundle=control
    #pragma HLS INTERFACE s_axilite port=attention_weights_data_5 bundle=control
    #pragma HLS INTERFACE s_axilite port=attention_weights_data_6 bundle=control
    #pragma HLS INTERFACE s_axilite port=attention_weights_data_7 bundle=control
    #pragma HLS INTERFACE s_axilite port=output_data_0 bundle=control
    #pragma HLS INTERFACE s_axilite port=output_data_1 bundle=control
    #pragma HLS INTERFACE s_axilite port=output_data_2 bundle=control
    #pragma HLS INTERFACE s_axilite port=output_data_3 bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    #ifndef __SYNTHESIS__
    printf("DEBUG: msda_top called. CONST_NUM_QUERIES=%d\n", CONST_NUM_QUERIES);
    fflush(stdout);
    #endif
    
    
    int output_index = 0;

    const int batch_stride = CONST_NUM_HEADS * CONST_CHANNELS * CONST_SPATIAL_SIZE;
    
    // OUTPUT_PACK: raw48 storage only, no numeric conversion or recomputation.
    // Eight banks/group; 5000 addresses/bank; estimated two URAMs/bank.
    ap_uint<48> output_tile_0[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH];
    ap_uint<48> output_tile_1[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH];
    ap_uint<48> output_tile_2[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH];
    ap_uint<48> output_tile_3[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH];
    #pragma HLS ARRAY_PARTITION variable=output_tile_0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output_tile_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output_tile_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output_tile_3 complete dim=1
    #pragma HLS BIND_STORAGE variable=output_tile_0 type=ram_t2p impl=uram latency=2
    #pragma HLS BIND_STORAGE variable=output_tile_1 type=ram_t2p impl=uram latency=2
    #pragma HLS BIND_STORAGE variable=output_tile_2 type=ram_t2p impl=uram latency=2
    #pragma HLS BIND_STORAGE variable=output_tile_3 type=ram_t2p impl=uram latency=2

    // H second output tile: independent raw48 storage, no numeric conversion.
    // Eight banks/group; 5000 addresses/bank; estimated two URAMs/bank.
    ap_uint<48> output_tile_b_0[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH];
    ap_uint<48> output_tile_b_1[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH];
    ap_uint<48> output_tile_b_2[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH];
    ap_uint<48> output_tile_b_3[OUTPUT_TILE_BANKS][OUTPUT_TILE_DEPTH];
    #pragma HLS ARRAY_PARTITION variable=output_tile_b_0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output_tile_b_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output_tile_b_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output_tile_b_3 complete dim=1
    #pragma HLS BIND_STORAGE variable=output_tile_b_0 type=ram_t2p impl=uram latency=2
    #pragma HLS BIND_STORAGE variable=output_tile_b_1 type=ram_t2p impl=uram latency=2
    #pragma HLS BIND_STORAGE variable=output_tile_b_2 type=ram_t2p impl=uram latency=2
    #pragma HLS BIND_STORAGE variable=output_tile_b_3 type=ram_t2p impl=uram latency=2

    // IMPORTANT: Initialize to 0 to avoid garbage data from out-of-bounds prefetch
    feature_cache_word_t value_buf0_L0[16][8000];  // group0:0-3, group1:4-7, group2:8-11, group3:12-15
    feature_cache_word_t value_buf0_L1[16][2000];
    feature_cache_word_t value_buf0_L2[16][512];
    feature_cache_word_t value_buf0_L3[16][128];
    feature_cache_word_t value_buf1_L0[16][8000];
    feature_cache_word_t value_buf1_L1[16][2000];
    feature_cache_word_t value_buf1_L2[16][512];
    feature_cache_word_t value_buf1_L3[16][128];
    // Timing @200MHz: URAM EN/CE nets tend to become long high-fanout cross-SLR routes.
    // Try mapping L0 ping-pong buffers to BRAM to give placer more flexibility and reduce SLR-crossing control nets.
    #pragma HLS BIND_STORAGE variable=value_buf0_L0 type=ram_t2p impl=bram latency=2
    #pragma HLS BIND_STORAGE variable=value_buf0_L1 type=ram_t2p impl=bram latency=2
    #pragma HLS BIND_STORAGE variable=value_buf0_L2 type=ram_t2p impl=bram latency=2
    #pragma HLS BIND_STORAGE variable=value_buf0_L3 type=ram_t2p impl=bram latency=2
    #pragma HLS BIND_STORAGE variable=value_buf1_L0 type=ram_t2p impl=bram latency=2
    #pragma HLS BIND_STORAGE variable=value_buf1_L1 type=ram_t2p impl=bram latency=2
    #pragma HLS BIND_STORAGE variable=value_buf1_L2 type=ram_t2p impl=bram latency=2
    #pragma HLS BIND_STORAGE variable=value_buf1_L3 type=ram_t2p impl=bram latency=2
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

    // One complete raw-word cache for each P=4 query group, refilled per (view, head).
    ap_uint<1920> metadata_cache_0[METADATA_QUERIES_PER_GROUP];
    ap_uint<1920> metadata_cache_1[METADATA_QUERIES_PER_GROUP];
    ap_uint<1920> metadata_cache_2[METADATA_QUERIES_PER_GROUP];
    ap_uint<1920> metadata_cache_3[METADATA_QUERIES_PER_GROUP];
    #pragma HLS BIND_STORAGE variable=metadata_cache_0 type=ram_2p impl=uram latency=2
    #pragma HLS BIND_STORAGE variable=metadata_cache_1 type=ram_2p impl=uram latency=2
    #pragma HLS BIND_STORAGE variable=metadata_cache_2 type=ram_2p impl=uram latency=2
    #pragma HLS BIND_STORAGE variable=metadata_cache_3 type=ram_2p impl=uram latency=2

    // H: stage0 initialize current head; stage1..47 overlap previous flush
    // with current prepare+compute; stage48 drains the final head.
    // No PIPELINE/DATAFLOW across stages: each stage waits for both tasks.
    LOOP_HEAD_STAGES_H:
    for (int stage = 0; stage <= CONST_BATCH_SIZE * CONST_NUM_HEADS; ++stage) {
        head_stage_unified(
            value_data, sampling_locations_data_0, sampling_locations_data_1, sampling_locations_data_2, sampling_locations_data_3, sampling_locations_data_4, sampling_locations_data_5, sampling_locations_data_6, sampling_locations_data_7, sampling_locations_data_8, sampling_locations_data_9, sampling_locations_data_10, sampling_locations_data_11, sampling_locations_data_12, sampling_locations_data_13, sampling_locations_data_14, sampling_locations_data_15, attention_weights_data_0, attention_weights_data_1, attention_weights_data_2, attention_weights_data_3, attention_weights_data_4, attention_weights_data_5, attention_weights_data_6, attention_weights_data_7, metadata_cache_0, metadata_cache_1, metadata_cache_2, metadata_cache_3, output_tile_0, output_tile_1, output_tile_2, output_tile_3, output_tile_b_0, output_tile_b_1, output_tile_b_2, output_tile_b_3, output_data_0, output_data_1, output_data_2, output_data_3, value_buf0_L0, value_buf0_L1, value_buf0_L2, value_buf0_L3, value_buf1_L0, value_buf1_L1, value_buf1_L2, value_buf1_L3, stage
        );
    }
}
