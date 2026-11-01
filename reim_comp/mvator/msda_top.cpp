/**
 * @file msda_top.cpp
 * @brief MSDA HLS Implementation - Complete Version
 * USE_4BANK: 4 value ports (4-bank), same BI/II-P logic, align with Step1 golden.
 */

#include "../INCLUDE/msda_top.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cassert>
#ifndef USE_8WAY_FIFO_POLLING
#error "This deployment variant requires the retained eight-FIFO topology"
#endif

#ifndef II_P_THRESHOLD
#define II_P_THRESHOLD 0.05f
#endif
#ifndef IQ_P_THRESHOLD
#define IQ_P_THRESHOLD 0.0f
#endif

inline data_t bilinear_interpolation(
    const data_t v1, const data_t v2, const data_t v3, const data_t v4,
    const data_t lh, const data_t lw) {
    const data_t dv2 = v2 - v1;
    const data_t dv3 = v3 - v1;
    const data_t dv4 = v4 - v2 - v3 + v1;
    const data_t term1 = lw * dv2;
    const data_t term2 = lh * dv3;
    const data_t term3 = lh * lw;
    return v1 + term1 + term2 + term3 * dv4;
}

#ifdef USE_4BANK
static inline int floor_div2(int x) {
    return (x >= 0) ? (x / 2) : ((x - 1) / 2);
}

static inline int bank_local_pixel(int h, int w, int W, int k) {
    if (k == 0) return floor_div2(h) * ((W + 1) / 2) + floor_div2(w);
    if (k == 1) return floor_div2(h) * (W / 2) + floor_div2(w - 1);
    if (k == 2) return floor_div2(h - 1) * ((W + 1) / 2) + floor_div2(w);
    return floor_div2(h - 1) * (W / 2) + floor_div2(w - 1);
}

struct SampleToken {
    int l_col;
    location_t loc_w;
    location_t loc_h;
    data_t weight;
    bool is_end;
};

static void msda_producer(
    const axi_word_t* sampling_locations_data,
    const axi_word_t* attention_weights_data,
    int b_col, int query_idx, int m_col,
    hls::stream<SampleToken>& sample_fifo
) {
    const int loc_elements = CONST_NUM_LEVELS * CONST_NUM_POINTS * 2; // 64
    const int weight_elements = CONST_NUM_LEVELS * CONST_NUM_POINTS; // 32
    
    const int base_loc_word = (b_col * CONST_NUM_QUERIES * CONST_NUM_HEADS + query_idx * CONST_NUM_HEADS + m_col) * loc_elements / AXI_VEC_SIZE;
    const int base_weight_word = (b_col * CONST_NUM_QUERIES * CONST_NUM_HEADS + query_idx * CONST_NUM_HEADS + m_col) * weight_elements / AXI_VEC_SIZE;
    
    location_t loc_buf[64];
    data_t weight_buf[32];
#pragma HLS ARRAY_PARTITION variable=loc_buf cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=weight_buf cyclic factor=32 dim=1

    READ_LOC:
    for (int i = 0; i < 64 / AXI_VEC_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        axi_word_t word = sampling_locations_data[base_loc_word + i];
        for (int j = 0; j < AXI_VEC_SIZE; ++j) {
#pragma HLS UNROLL
            loc_buf[i * AXI_VEC_SIZE + j] = word[j];
        }
    }
    READ_WEIGHT:
    for (int i = 0; i < 32 / AXI_VEC_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        axi_word_t word = attention_weights_data[base_weight_word + i];
        for (int j = 0; j < AXI_VEC_SIZE; ++j) {
#pragma HLS UNROLL
            weight_buf[i * AXI_VEC_SIZE + j] = word[j];
        }
    }

    SPATIAL_PRUNER:
    for (int lp = 0; lp < CONST_NUM_LEVELS * CONST_NUM_POINTS + 1; ++lp) {
#pragma HLS PIPELINE II=1
        if (lp == CONST_NUM_LEVELS * CONST_NUM_POINTS) {
            SampleToken end_token;
            end_token.is_end = true;
            sample_fifo.write(end_token);
        } else {
            const int l_col = lp / CONST_NUM_POINTS;
            const data_t w = weight_buf[lp];
#ifndef IQ_P_DISABLE
            const data_t T_IQ = (data_t)IQ_P_THRESHOLD;
            if (w >= T_IQ) {
#endif
                SampleToken token;
                token.l_col = l_col;
                token.loc_w = loc_buf[lp * 2];
                token.loc_h = loc_buf[lp * 2 + 1];
                token.weight = w;
                token.is_end = false;
                sample_fifo.write(token);
#ifndef IQ_P_DISABLE
            }
#endif
        }
    }
}

#ifdef USE_8WAY_FIFO_POLLING
static void msda_producer_8way(
    const axi_word_t* sampling_locations_data,
    const axi_word_t* attention_weights_data,
    int b_col, int query_idx, int m_col,
    hls::stream<SampleToken>& fifo0, hls::stream<SampleToken>& fifo1,
    hls::stream<SampleToken>& fifo2, hls::stream<SampleToken>& fifo3,
    hls::stream<SampleToken>& fifo4, hls::stream<SampleToken>& fifo5,
    hls::stream<SampleToken>& fifo6, hls::stream<SampleToken>& fifo7
) {
    const int loc_elements = CONST_NUM_LEVELS * CONST_NUM_POINTS * 2;
    const int weight_elements = CONST_NUM_LEVELS * CONST_NUM_POINTS;
    
    const int base_loc_word = query_idx * loc_elements / AXI_VEC_SIZE;
    const int base_weight_word = query_idx * weight_elements / AXI_VEC_SIZE;
    
    location_t loc_buf[64];
    data_t weight_buf[32];
#pragma HLS ARRAY_PARTITION variable=loc_buf cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=weight_buf cyclic factor=32 dim=1

    READ_LOC_8W:
    for (int i = 0; i < 64 / AXI_VEC_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        axi_word_t word = sampling_locations_data[base_loc_word + i];
        for (int j = 0; j < AXI_VEC_SIZE; ++j) {
#pragma HLS UNROLL
            loc_buf[i * AXI_VEC_SIZE + j] = word[j];
        }
    }
    READ_WEIGHT_8W:
    for (int i = 0; i < 32 / AXI_VEC_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        axi_word_t word = attention_weights_data[base_weight_word + i];
        for (int j = 0; j < AXI_VEC_SIZE; ++j) {
#pragma HLS UNROLL
            weight_buf[i * AXI_VEC_SIZE + j] = word[j];
        }
    }

    int token_count = 0;
    SPATIAL_PRUNER_8W:
    for (int lp = 0; lp < CONST_NUM_LEVELS * CONST_NUM_POINTS + 1; ++lp) {
#pragma HLS PIPELINE II=1
        if (lp == CONST_NUM_LEVELS * CONST_NUM_POINTS) {
            SampleToken end_token;
            end_token.is_end = true;
            // The polling unit reads fifo0..fifo7 cyclically.  If pruning
            // emitted n tokens, it next reads fifo[n % 8]; writing the end
            // marker to fifo0 would deadlock whenever n is not a multiple of
            // eight (the poller would block on an empty intermediate FIFO).
            switch (token_count % 8) {
                case 0: fifo0.write(end_token); break;
                case 1: fifo1.write(end_token); break;
                case 2: fifo2.write(end_token); break;
                case 3: fifo3.write(end_token); break;
                case 4: fifo4.write(end_token); break;
                case 5: fifo5.write(end_token); break;
                case 6: fifo6.write(end_token); break;
                default: fifo7.write(end_token); break;
            }
        } else {
            const int l_col = lp / CONST_NUM_POINTS;
            const data_t w = weight_buf[lp];
#ifndef IQ_P_DISABLE
            const data_t T_IQ = (data_t)IQ_P_THRESHOLD;
            if (w >= T_IQ) {
#endif
                SampleToken token;
                token.l_col = l_col;
                token.loc_w = loc_buf[lp * 2];
                token.loc_h = loc_buf[lp * 2 + 1];
                token.weight = w;
                token.is_end = false;
                switch (token_count % 8) {
                    case 0: fifo0.write(token); break;
                    case 1: fifo1.write(token); break;
                    case 2: fifo2.write(token); break;
                    case 3: fifo3.write(token); break;
                    case 4: fifo4.write(token); break;
                    case 5: fifo5.write(token); break;
                    case 6: fifo6.write(token); break;
                    default: fifo7.write(token); break;
                }
                token_count++;
#ifndef IQ_P_DISABLE
            }
#endif
        }
    }
}

static void msda_polling_unit(
    hls::stream<SampleToken>& fifo0, hls::stream<SampleToken>& fifo1,
    hls::stream<SampleToken>& fifo2, hls::stream<SampleToken>& fifo3,
    hls::stream<SampleToken>& fifo4, hls::stream<SampleToken>& fifo5,
    hls::stream<SampleToken>& fifo6, hls::stream<SampleToken>& fifo7,
    hls::stream<SampleToken>& consumer_fifo
) {
    SampleToken t;
    bool done = false;
    int k = 0;
    POLL_LOOP:
    while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=1 max=33
        switch (k) {
            case 0: t = fifo0.read(); break;
            case 1: t = fifo1.read(); break;
            case 2: t = fifo2.read(); break;
            case 3: t = fifo3.read(); break;
            case 4: t = fifo4.read(); break;
            case 5: t = fifo5.read(); break;
            case 6: t = fifo6.read(); break;
            default: t = fifo7.read(); break;
        }
        consumer_fifo.write(t);
        done = t.is_end;
        k = (k + 1) % 8;
    }
}

// Keep one producer, one polling stage, and one consumer for the full query
// tile. End markers delimit each query; the existing 8-way routing restarts
// at FIFO0 for every query.
static void msda_producer_tile(
    const axi_word_t* loc_tile, const axi_word_t* weight_tile, int n,
    hls::stream<SampleToken>& fifo0, hls::stream<SampleToken>& fifo1,
    hls::stream<SampleToken>& fifo2, hls::stream<SampleToken>& fifo3,
    hls::stream<SampleToken>& fifo4, hls::stream<SampleToken>& fifo5,
    hls::stream<SampleToken>& fifo6, hls::stream<SampleToken>& fifo7
) {
#pragma HLS INLINE off
    for (int q = 0; q < n; ++q) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=64
        msda_producer_8way(loc_tile, weight_tile, 0, q, 0,
            fifo0, fifo1, fifo2, fifo3, fifo4, fifo5, fifo6, fifo7);
    }
}

static void msda_polling_tile(
    int n,
    hls::stream<SampleToken>& fifo0, hls::stream<SampleToken>& fifo1,
    hls::stream<SampleToken>& fifo2, hls::stream<SampleToken>& fifo3,
    hls::stream<SampleToken>& fifo4, hls::stream<SampleToken>& fifo5,
    hls::stream<SampleToken>& fifo6, hls::stream<SampleToken>& fifo7,
    hls::stream<SampleToken>& consumer_fifo
) {
#pragma HLS INLINE off
    for (int q = 0; q < n; ++q) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=64
        msda_polling_unit(fifo0, fifo1, fifo2, fifo3, fifo4, fifo5, fifo6, fifo7,
            consumer_fifo);
    }
}
#endif

static void msda_consumer(
    hls::stream<SampleToken>& sample_fifo,
    const feature_t v_buffer_bank0[4115][32],
    const feature_t v_buffer_bank1[4115][32],
    const feature_t v_buffer_bank2[4115][32],
    const feature_t v_buffer_bank3[4115][32],
    axi_word_t* output_data,
    int b_col, int query_idx, int m_col
) {
    accum_t accumulated_values[32];
#pragma HLS ARRAY_PARTITION variable=accumulated_values complete
    for (int c = 0; c < 32; ++c) {
#pragma HLS UNROLL
        accumulated_values[c] = 0;
    }

    bool done = false;
    LOOP_VALID_SAMPLES:
    while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=4 max=32 avg=16
        SampleToken token = sample_fifo.read();
        if (token.is_end) {
            done = true;
        } else {
            const int l_col = token.l_col;
            const location_t loc_w = token.loc_w;
            const location_t loc_h = token.loc_h;
            const data_t weight = token.weight;
            
            const int spatial_h = CONST_SPATIAL_SHAPES[l_col << 1];
            const int spatial_w = CONST_SPATIAL_SHAPES[(l_col << 1) + 1];
            
            // SAFP keeps the lower 56 rows of the paper's level 1 (H=80).
            // Preserve original normalized coordinates by subtracting the 24-row
            // crop offset before applying the standard half-pixel convention.
            pixel_coord_t h_im;
            if (l_col == 0) {
                // Compute the original-coordinate sample first, then shift it
                // into the compacted feature map. The 24-row shift is exact
                // in this fixed-point format.
                h_im = loc_h * spatial_h - (data_t)0.5f - (data_t)24;
            } else {
                h_im = loc_h * spatial_h - (data_t)0.5f;
            }
            const pixel_coord_t w_im = loc_w * spatial_w - (pixel_coord_t)0.5f;
            const int h_low = ((ap_int<12>)h_im.range(23, 12)).to_int();
            const int w_low = ((ap_int<12>)w_im.range(23, 12)).to_int();
            const int h_high = h_low + 1;
            const int w_high = w_low + 1;
            const data_t lh = h_im - h_low;
            const data_t lw = w_im - w_low;
            
            const int valid1 = (h_low >= 0) & (w_low >= 0) & (h_low < spatial_h) & (w_low < spatial_w);
            const int valid2 = (h_low >= 0) & (w_high >= 0) & (h_low < spatial_h) & (w_high < spatial_w);
            const int valid3 = (h_high >= 0) & (w_low >= 0) & (h_high < spatial_h) & (w_low < spatial_w);
            const int valid4 = (h_high >= 0) & (w_high >= 0) & (h_high < spatial_h) & (w_high < spatial_w);
            
            const int k1 = (((h_low % 2) + 2) % 2) * 2 + (((w_low % 2) + 2) % 2);
            const int k2 = (((h_low % 2) + 2) % 2) * 2 + (((w_high % 2) + 2) % 2);
            const int k3 = (((h_high % 2) + 2) % 2) * 2 + (((w_low % 2) + 2) % 2);
            const int k4 = (((h_high % 2) + 2) % 2) * 2 + (((w_high % 2) + 2) % 2);
            
            const int lp1 = bank_local_pixel(h_low, w_low, spatial_w, k1);
            const int lp2 = bank_local_pixel(h_low, w_high, spatial_w, k2);
            const int lp3 = bank_local_pixel(h_high, w_low, spatial_w, k3);
            const int lp4 = bank_local_pixel(h_high, w_high, spatial_w, k4);
            
            const int o1 = CONST_BANK_LEVEL_START[k1][l_col] + lp1;
            const int o2 = CONST_BANK_LEVEL_START[k2][l_col] + lp2;
            const int o3 = CONST_BANK_LEVEL_START[k3][l_col] + lp3;
            const int o4 = CONST_BANK_LEVEL_START[k4][l_col] + lp4;

            int ob0 = (k1 == 0) ? o1 : (k2 == 0) ? o2 : (k3 == 0) ? o3 : o4;
            int ob1 = (k1 == 1) ? o1 : (k2 == 1) ? o2 : (k3 == 1) ? o3 : o4;
            int ob2 = (k1 == 2) ? o1 : (k2 == 2) ? o2 : (k3 == 2) ? o3 : o4;
            int ob3 = (k1 == 3) ? o1 : (k2 == 3) ? o2 : (k3 == 3) ? o3 : o4;

            ob0 = (ob0 >= 0 && ob0 < 4115) ? ob0 : 0;
            ob1 = (ob1 >= 0 && ob1 < 4115) ? ob1 : 0;
            ob2 = (ob2 >= 0 && ob2 < 4115) ? ob2 : 0;
            ob3 = (ob3 >= 0 && ob3 < 4115) ? ob3 : 0;

            for (int c = 0; c < 32; ++c) {
#pragma HLS UNROLL
                data_t r0_val = v_buffer_bank0[ob0][c];
                data_t r1_val = v_buffer_bank1[ob1][c];
                data_t r2_val = v_buffer_bank2[ob2][c];
                data_t r3_val = v_buffer_bank3[ob3][c];

                data_t v1 = (k1 == 0) ? r0_val : (k1 == 1) ? r1_val : (k1 == 2) ? r2_val : r3_val;
                data_t v2 = (k2 == 0) ? r0_val : (k2 == 1) ? r1_val : (k2 == 2) ? r2_val : r3_val;
                data_t v3 = (k3 == 0) ? r0_val : (k3 == 1) ? r1_val : (k3 == 2) ? r2_val : r3_val;
                data_t v4 = (k4 == 0) ? r0_val : (k4 == 1) ? r1_val : (k4 == 2) ? r2_val : r3_val;

                if (!valid1) v1 = 0.0f;
                if (!valid2) v2 = 0.0f;
                if (!valid3) v3 = 0.0f;
                if (!valid4) v4 = 0.0f;

                data_t sampled_value;
#ifndef II_P_DISABLE
                const data_t T_II = (data_t)II_P_THRESHOLD;
                const data_t one = (data_t)1.0f;
                const int keep_p1 = ((one - lh) > T_II) || ((one - lw) > T_II);
                const int keep_p2 = ((one - lh) > T_II) || (lw > T_II);
                const int keep_p3 = (lh > T_II) || ((one - lw) > T_II);
                const int keep_p4 = (lh > T_II) || (lw > T_II);
                const int e1 = valid1 & keep_p1, e2 = valid2 & keep_p2, e3 = valid3 & keep_p3, e4 = valid4 & keep_p4;
                const int cnt = e1 + e2 + e3 + e4;
                if (cnt >= 4)
                    sampled_value = bilinear_interpolation(v1, v2, v3, v4, lh, lw);
                else if (cnt == 0)
                    sampled_value = bilinear_interpolation(v1, v2, v3, v4, lh, lw);
                else {
                    const data_t sum_v = (e1 ? v1 : (data_t)0.0f) + (e2 ? v2 : (data_t)0.0f) + (e3 ? v3 : (data_t)0.0f) + (e4 ? v4 : (data_t)0.0f);
                    const data_t inv_n = (cnt == 1) ? (data_t)1.0f : (cnt == 2) ? (data_t)0.5f : (data_t)0.33333334f;
                    sampled_value = sum_v * inv_n;
                }
#else
                sampled_value = bilinear_interpolation(v1, v2, v3, v4, lh, lw);
#endif
                accumulated_values[c] += sampled_value * weight;
            }
        }
    }
    
    axi_word_t out_word;
    for (int c = 0; c < 32; ++c) {
#pragma HLS UNROLL
        out_word[c % AXI_VEC_SIZE] = (data_t)accumulated_values[c];
    }
    // One result per local query; external layout is restored by DRAIN_OUTPUT_TILE.
    output_data[query_idx] = out_word;
}

#ifdef USE_8WAY_FIFO_POLLING
static void msda_consumer_tile(
    hls::stream<SampleToken>& sample_fifo,
    const feature_t v_buffer_bank0[4115][32],
    const feature_t v_buffer_bank1[4115][32],
    const feature_t v_buffer_bank2[4115][32],
    const feature_t v_buffer_bank3[4115][32],
    axi_word_t* output_tile, int n
) {
    accum_t accumulated_values[32];
#pragma HLS ARRAY_PARTITION variable=accumulated_values complete
    for (int c = 0; c < 32; ++c) {
#pragma HLS UNROLL
        accumulated_values[c] = 0;
    }

    int query_idx = 0;
    LOOP_TILE_SAMPLES:
    while (query_idx < n) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=1 max=2112 avg=128
        SampleToken token = sample_fifo.read();
        if (token.is_end) {
            axi_word_t out_word;
            for (int c = 0; c < 32; ++c) {
#pragma HLS UNROLL
                out_word[c % AXI_VEC_SIZE] = (data_t)accumulated_values[c];
                accumulated_values[c] = 0;
            }
            output_tile[query_idx] = out_word;
            query_idx++;
        } else {
            const int l_col = token.l_col;
            const location_t loc_w = token.loc_w;
            const location_t loc_h = token.loc_h;
            const data_t weight = token.weight;
            
            const int spatial_h = CONST_SPATIAL_SHAPES[l_col << 1];
            const int spatial_w = CONST_SPATIAL_SHAPES[(l_col << 1) + 1];
            
            // SAFP keeps the lower 56 rows of the paper's level 1 (H=80).
            // Preserve original normalized coordinates by subtracting the 24-row
            // crop offset before applying the standard half-pixel convention.
            pixel_coord_t h_im;
            if (l_col == 0) {
                // Compute the original-coordinate sample first, then shift it
                // into the compacted feature map. The 24-row shift is exact
                // in this fixed-point format.
                h_im = loc_h * spatial_h - (data_t)0.5f - (data_t)24;
            } else {
                h_im = loc_h * spatial_h - (data_t)0.5f;
            }
            const pixel_coord_t w_im = loc_w * spatial_w - (pixel_coord_t)0.5f;
            const int h_low = ((ap_int<12>)h_im.range(23, 12)).to_int();
            const int w_low = ((ap_int<12>)w_im.range(23, 12)).to_int();
            const int h_high = h_low + 1;
            const int w_high = w_low + 1;
            const data_t lh = h_im - h_low;
            const data_t lw = w_im - w_low;
            
            const int valid1 = (h_low >= 0) & (w_low >= 0) & (h_low < spatial_h) & (w_low < spatial_w);
            const int valid2 = (h_low >= 0) & (w_high >= 0) & (h_low < spatial_h) & (w_high < spatial_w);
            const int valid3 = (h_high >= 0) & (w_low >= 0) & (h_high < spatial_h) & (w_low < spatial_w);
            const int valid4 = (h_high >= 0) & (w_high >= 0) & (h_high < spatial_h) & (w_high < spatial_w);
            
            const int k1 = (((h_low % 2) + 2) % 2) * 2 + (((w_low % 2) + 2) % 2);
            const int k2 = (((h_low % 2) + 2) % 2) * 2 + (((w_high % 2) + 2) % 2);
            const int k3 = (((h_high % 2) + 2) % 2) * 2 + (((w_low % 2) + 2) % 2);
            const int k4 = (((h_high % 2) + 2) % 2) * 2 + (((w_high % 2) + 2) % 2);
            
            const int lp1 = bank_local_pixel(h_low, w_low, spatial_w, k1);
            const int lp2 = bank_local_pixel(h_low, w_high, spatial_w, k2);
            const int lp3 = bank_local_pixel(h_high, w_low, spatial_w, k3);
            const int lp4 = bank_local_pixel(h_high, w_high, spatial_w, k4);
            
            const int o1 = CONST_BANK_LEVEL_START[k1][l_col] + lp1;
            const int o2 = CONST_BANK_LEVEL_START[k2][l_col] + lp2;
            const int o3 = CONST_BANK_LEVEL_START[k3][l_col] + lp3;
            const int o4 = CONST_BANK_LEVEL_START[k4][l_col] + lp4;

            int ob0 = (k1 == 0) ? o1 : (k2 == 0) ? o2 : (k3 == 0) ? o3 : o4;
            int ob1 = (k1 == 1) ? o1 : (k2 == 1) ? o2 : (k3 == 1) ? o3 : o4;
            int ob2 = (k1 == 2) ? o1 : (k2 == 2) ? o2 : (k3 == 2) ? o3 : o4;
            int ob3 = (k1 == 3) ? o1 : (k2 == 3) ? o2 : (k3 == 3) ? o3 : o4;

            ob0 = (ob0 >= 0 && ob0 < 4115) ? ob0 : 0;
            ob1 = (ob1 >= 0 && ob1 < 4115) ? ob1 : 0;
            ob2 = (ob2 >= 0 && ob2 < 4115) ? ob2 : 0;
            ob3 = (ob3 >= 0 && ob3 < 4115) ? ob3 : 0;

            for (int c = 0; c < 32; ++c) {
#pragma HLS UNROLL
                data_t r0_val = v_buffer_bank0[ob0][c];
                data_t r1_val = v_buffer_bank1[ob1][c];
                data_t r2_val = v_buffer_bank2[ob2][c];
                data_t r3_val = v_buffer_bank3[ob3][c];

                data_t v1 = (k1 == 0) ? r0_val : (k1 == 1) ? r1_val : (k1 == 2) ? r2_val : r3_val;
                data_t v2 = (k2 == 0) ? r0_val : (k2 == 1) ? r1_val : (k2 == 2) ? r2_val : r3_val;
                data_t v3 = (k3 == 0) ? r0_val : (k3 == 1) ? r1_val : (k3 == 2) ? r2_val : r3_val;
                data_t v4 = (k4 == 0) ? r0_val : (k4 == 1) ? r1_val : (k4 == 2) ? r2_val : r3_val;

                if (!valid1) v1 = 0.0f;
                if (!valid2) v2 = 0.0f;
                if (!valid3) v3 = 0.0f;
                if (!valid4) v4 = 0.0f;

                data_t sampled_value;
#ifndef II_P_DISABLE
                const data_t T_II = (data_t)II_P_THRESHOLD;
                const data_t one = (data_t)1.0f;
                const int keep_p1 = ((one - lh) > T_II) || ((one - lw) > T_II);
                const int keep_p2 = ((one - lh) > T_II) || (lw > T_II);
                const int keep_p3 = (lh > T_II) || ((one - lw) > T_II);
                const int keep_p4 = (lh > T_II) || (lw > T_II);
                const int e1 = valid1 & keep_p1, e2 = valid2 & keep_p2, e3 = valid3 & keep_p3, e4 = valid4 & keep_p4;
                const int cnt = e1 + e2 + e3 + e4;
                if (cnt >= 4)
                    sampled_value = bilinear_interpolation(v1, v2, v3, v4, lh, lw);
                else if (cnt == 0)
                    sampled_value = bilinear_interpolation(v1, v2, v3, v4, lh, lw);
                else {
                    const data_t sum_v = (e1 ? v1 : (data_t)0.0f) + (e2 ? v2 : (data_t)0.0f) + (e3 ? v3 : (data_t)0.0f) + (e4 ? v4 : (data_t)0.0f);
                    const data_t inv_n = (cnt == 1) ? (data_t)1.0f : (cnt == 2) ? (data_t)0.5f : (data_t)0.33333334f;
                    sampled_value = sum_v * inv_n;
                }
#else
                sampled_value = bilinear_interpolation(v1, v2, v3, v4, lh, lw);
#endif
                accumulated_values[c] += sampled_value * weight;
            }
        }
    }

}

static void msda_process_query_tile(
    const axi_word_t* loc_tile, const axi_word_t* weight_tile,
    axi_word_t* output_tile, int n,
    const feature_t v_buffer_bank0[4115][32],
    const feature_t v_buffer_bank1[4115][32],
    const feature_t v_buffer_bank2[4115][32],
    const feature_t v_buffer_bank3[4115][32]
) {
#pragma HLS INLINE off
    hls::stream<SampleToken> fifo0("fifo0"), fifo1("fifo1"), fifo2("fifo2"), fifo3("fifo3");
    hls::stream<SampleToken> fifo4("fifo4"), fifo5("fifo5"), fifo6("fifo6"), fifo7("fifo7");
    hls::stream<SampleToken> consumer_fifo("consumer_fifo");
#pragma HLS STREAM variable=fifo0 depth=16
#pragma HLS STREAM variable=fifo1 depth=16
#pragma HLS STREAM variable=fifo2 depth=16
#pragma HLS STREAM variable=fifo3 depth=16
#pragma HLS STREAM variable=fifo4 depth=16
#pragma HLS STREAM variable=fifo5 depth=16
#pragma HLS STREAM variable=fifo6 depth=16
#pragma HLS STREAM variable=fifo7 depth=16
#pragma HLS STREAM variable=consumer_fifo depth=64
#pragma HLS DATAFLOW
    msda_producer_tile(loc_tile, weight_tile, n,
        fifo0, fifo1, fifo2, fifo3, fifo4, fifo5, fifo6, fifo7);
    msda_polling_tile(n, fifo0, fifo1, fifo2, fifo3, fifo4, fifo5, fifo6, fifo7,
        consumer_fifo);
    msda_consumer_tile(consumer_fifo,
        v_buffer_bank0, v_buffer_bank1, v_buffer_bank2, v_buffer_bank3,
        output_tile, n);
}
#endif

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
) {
    #pragma HLS INTERFACE m_axi port=value_bank0 offset=slave bundle=Gmem0 depth=265625 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=value_bank1 offset=slave bundle=Gmem1 depth=265625 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=value_bank2 offset=slave bundle=Gmem2 depth=265625 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=value_bank3 offset=slave bundle=Gmem3 depth=265625 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=sampling_locations_data offset=slave bundle=Gmem4 depth=927552 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=attention_weights_data offset=slave bundle=Gmem5 depth=463776 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=output_data offset=slave bundle=Gmem6 depth=463776 max_write_burst_length=256 num_write_outstanding=16
    #pragma HLS INTERFACE s_axilite port=valid_count_0 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=valid_count_1 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=valid_count_2 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=valid_count_3 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=valid_count_4 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=valid_count_5 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    const int qid_stride = CONST_NUM_HEADS * CONST_CHANNELS;
    int valid_counts[CONST_BATCH_SIZE];
#pragma HLS ARRAY_PARTITION variable=valid_counts complete dim=1
    valid_counts[0] = valid_count_0;
    valid_counts[1] = valid_count_1;
    valid_counts[2] = valid_count_2;
    valid_counts[3] = valid_count_3;
    valid_counts[4] = valid_count_4;
    valid_counts[5] = valid_count_5;

    LOOP_BATCH:
    for (int b_col = 0; b_col < CONST_BATCH_SIZE; ++b_col) {
        // IV-P: indices are dense from zero and each camera switches to the
        // next one when its host-configured valid prefix is exhausted.
        const int raw_valid_count = valid_counts[b_col];
        const int camera_valid_count =
            (raw_valid_count > CONST_NUM_QUERIES) ? CONST_NUM_QUERIES
                                                  : raw_valid_count;
        if (camera_valid_count <= 0) {
            continue;
        }

        LOOP_NUM_HEADS:
        for (int m_col = 0; m_col < CONST_NUM_HEADS; ++m_col) {
            
            feature_t v_buffer_bank0[4115][32];
            feature_t v_buffer_bank1[4115][32];
            feature_t v_buffer_bank2[4115][32];
            feature_t v_buffer_bank3[4115][32];
            #pragma HLS ARRAY_PARTITION variable=v_buffer_bank0 dim=2 complete
            #pragma HLS ARRAY_PARTITION variable=v_buffer_bank1 dim=2 complete
            #pragma HLS ARRAY_PARTITION variable=v_buffer_bank2 dim=2 complete
            #pragma HLS ARRAY_PARTITION variable=v_buffer_bank3 dim=2 complete

            LOAD_SUB_HEAD:
            for (int bank_lp = 0; bank_lp < 4115; ++bank_lp) {
#pragma HLS PIPELINE II=1
                int mc_base = m_col * CONST_CHANNELS;
                // Head-major physical feature layout: [batch][head][bank_pixel][channel].
                // For a fixed head, consecutive bank_lp values now address
                // consecutive 256-bit feature words, while preserving the same feature
                // values and checkerboard-bank mapping.
                
                if (bank_lp < CONST_BANK_LEVEL_START[0][CONST_NUM_LEVELS]) {
                    int total0 = CONST_BANK_LEVEL_START[0][CONST_NUM_LEVELS];
                    int base0 = (b_col * total0 * qid_stride + mc_base * total0 + bank_lp * CONST_CHANNELS) / AXI_VEC_SIZE;
                    feature_word_t word = value_bank0[base0];
                    for (int c = 0; c < 32; ++c) {
#pragma HLS UNROLL
                        v_buffer_bank0[bank_lp][c] = word[c % AXI_VEC_SIZE];
                    }
                }
                if (bank_lp < CONST_BANK_LEVEL_START[1][CONST_NUM_LEVELS]) {
                    int total1 = CONST_BANK_LEVEL_START[1][CONST_NUM_LEVELS];
                    int base1 = (b_col * total1 * qid_stride + mc_base * total1 + bank_lp * CONST_CHANNELS) / AXI_VEC_SIZE;
                    feature_word_t word = value_bank1[base1];
                    for (int c = 0; c < 32; ++c) {
#pragma HLS UNROLL
                        v_buffer_bank1[bank_lp][c] = word[c % AXI_VEC_SIZE];
                    }
                }
                if (bank_lp < CONST_BANK_LEVEL_START[2][CONST_NUM_LEVELS]) {
                    int total2 = CONST_BANK_LEVEL_START[2][CONST_NUM_LEVELS];
                    int base2 = (b_col * total2 * qid_stride + mc_base * total2 + bank_lp * CONST_CHANNELS) / AXI_VEC_SIZE;
                    feature_word_t word = value_bank2[base2];
                    for (int c = 0; c < 32; ++c) {
#pragma HLS UNROLL
                        v_buffer_bank2[bank_lp][c] = word[c % AXI_VEC_SIZE];
                    }
                }
                if (bank_lp < CONST_BANK_LEVEL_START[3][CONST_NUM_LEVELS]) {
                    int total3 = CONST_BANK_LEVEL_START[3][CONST_NUM_LEVELS];
                    int base3 = (b_col * total3 * qid_stride + mc_base * total3 + bank_lp * CONST_CHANNELS) / AXI_VEC_SIZE;
                    feature_word_t word = value_bank3[base3];
                    for (int c = 0; c < 32; ++c) {
#pragma HLS UNROLL
                        v_buffer_bank3[bank_lp][c] = word[c % AXI_VEC_SIZE];
                    }
                }
            }

#ifdef USE_8WAY_FIFO_POLLING
            // Physical metadata/output layout: [view][head][query][...].
            // Keep the original 9662-slot span, but touch only the valid
            // prefix supplied by the host. All stages below share n.
            const int TILE_Q = 64;
            LOOP_QUERY_TILES:
            for (int q_base = 0; q_base < camera_valid_count; q_base += TILE_Q) {
                axi_word_t loc_tile[TILE_Q * 2];
                axi_word_t weight_tile[TILE_Q];
                axi_word_t output_tile[TILE_Q];
#pragma HLS BIND_STORAGE variable=loc_tile type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=weight_tile type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=output_tile type=ram_2p impl=bram
                const int n = (q_base + TILE_Q <= camera_valid_count)
                            ? TILE_Q : camera_valid_count - q_base;
                PREFETCH_LOC_TILE:
                for (int i = 0; i < n * 2; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=2 max=128 avg=128
                    const int addr = ((b_col * CONST_NUM_HEADS + m_col)
                                   * CONST_NUM_QUERIES + q_base) * 2 + i;
                    loc_tile[i] = sampling_locations_data[addr];
                }
                PREFETCH_WEIGHT_TILE:
                for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=64
                    const int addr = ((b_col * CONST_NUM_HEADS + m_col)
                                   * CONST_NUM_QUERIES + q_base + i);
                    weight_tile[i] = attention_weights_data[addr];
                }
                // One dataflow graph serves the whole tile, amortizing
                // per-query process start/stop while retaining one sampler.
                msda_process_query_tile(loc_tile, weight_tile, output_tile, n,
                    v_buffer_bank0, v_buffer_bank1, v_buffer_bank2, v_buffer_bank3);
                DRAIN_OUTPUT_TILE:
                for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=64
                    const int addr = ((b_col * CONST_NUM_HEADS + m_col)
                                   * CONST_NUM_QUERIES + q_base + i);
                    output_data[addr] = output_tile[i];
                }
            }
#else
            LOOP_NUM_QUERIES:
            for (int query_idx = 0; query_idx < CONST_NUM_QUERIES_RUN; ++query_idx) {
#pragma HLS DATAFLOW
                hls::stream<SampleToken> sample_fifo("sample_fifo");
#pragma HLS STREAM variable=sample_fifo depth=64

                msda_producer(sampling_locations_data, attention_weights_data, b_col, query_idx, m_col, sample_fifo);
                msda_consumer(
                    sample_fifo, 
                    v_buffer_bank0, v_buffer_bank1, v_buffer_bank2, v_buffer_bank3,
                    output_data, b_col, query_idx, m_col
                );
            }
#endif
        }
    }
}
#else
// ... Original serial MSDA code
#endif
