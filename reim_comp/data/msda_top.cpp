/**
 * @file msda_top.cpp
 * @brief MSDA HLS Implementation - Complete Version
 */

#include "../INCLUDE/msda_top.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cassert>

#ifndef USE_NEAREST_NEIGHBOR
#define USE_NEAREST_NEIGHBOR 0
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

inline data_t nearest_neighbor_sample(
    const data_t v1, const data_t v2, const data_t v3, const data_t v4,
    const data_t lh, const data_t lw) {
    
    const bool h_high = (lh >= (data_t)0.5f);
    const bool w_high = (lw >= (data_t)0.5f);
    
    if (!h_high && !w_high) {
        return v1;
    } else if (!h_high && w_high) {
        return v2;
    } else if (h_high && !w_high) {
        return v3;
    } else {
        return v4;
    }
}

#if USE_PINGPONG_BUFFER
inline void load_value_to_buffer_all_channels(
    const ap_uint<512>* value_data,
    const int b_col,
    const int m_col,
    data_t buffer[CONST_CHANNELS][CONST_SPATIAL_SIZE],
    const int data_value_ptr_init_offset,
    const int qid_stride) {
    
    CACHE_LOAD_SPATIAL:
    for (int s = 0; s < CONST_SPATIAL_SIZE; ++s) {
        #pragma HLS PIPELINE II=1
        // ((b*heads + m)*spatial + s).
        (void)data_value_ptr_init_offset;
        (void)qid_stride;
        const int addr_512 = (b_col * CONST_NUM_HEADS + m_col) * CONST_SPATIAL_SIZE + s;
        ap_uint<512> word = value_data[addr_512];
        
        CACHE_LOAD_CHANNEL:
        for (int c_col = 0; c_col < CONST_CHANNELS; ++c_col) {
            #pragma HLS UNROLL
            ap_uint<16> bits = word.range(c_col * 16 + 15, c_col * 16);
            data_t val;
            val.range(12, 0) = bits.range(12, 0);
            buffer[c_col][s] = val;
        }
    }
}
#endif

void msda_top(
    const ap_uint<512>* value_data,
    const ap_uint<512>* sampling_locations_data,
    const ap_uint<512>* attention_weights_data,
    ap_uint<512>* output_data
) {
    #pragma HLS INTERFACE m_axi port=value_data offset=slave bundle=Gmem0 depth=1020000 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=sampling_locations_data offset=slave bundle=Gmem1 depth=927552 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=attention_weights_data offset=slave bundle=Gmem2 depth=463776 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=output_data offset=slave bundle=Gmem3 depth=463776 max_write_burst_length=256 num_write_outstanding=16
    
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    
    const int qid_stride = CONST_NUM_HEADS * CONST_CHANNELS;
    
    #if USE_PINGPONG_BUFFER
    data_t value_buf0[CONST_CHANNELS][CONST_SPATIAL_SIZE];
    data_t value_buf1[CONST_CHANNELS][CONST_SPATIAL_SIZE];
    #pragma HLS ARRAY_PARTITION variable=value_buf0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_buf1 complete dim=1
    #pragma HLS BIND_STORAGE variable=value_buf0 type=ram_t2p impl=bram
    #pragma HLS BIND_STORAGE variable=value_buf1 type=ram_t2p impl=bram
    #endif
    
    LOOP_BATCH:
    for (int b_col = 0; b_col < CONST_BATCH_SIZE; ++b_col) {
        const int data_value_ptr_init_offset = b_col * CONST_SPATIAL_SIZE * qid_stride;
        
        LOOP_NUM_HEADS:
        for (int m_col = 0; m_col < CONST_NUM_HEADS; ++m_col) {
            
            #if USE_PINGPONG_BUFFER
            if (m_col % 2 == 0) {
                load_value_to_buffer_all_channels(value_data, b_col, m_col, value_buf0, data_value_ptr_init_offset, qid_stride);
            } else {
                load_value_to_buffer_all_channels(value_data, b_col, m_col, value_buf1, data_value_ptr_init_offset, qid_stride);
            }
            #endif
            
            LOOP_NUM_QUERIES:
            for (int query_idx = 0; query_idx < CONST_NUM_QUERIES_RUN; ++query_idx) {
                // Cache the complete metadata for this (batch, query, head).
                // The original RTL reread the same 512-bit words for every
                // sampling point: 64 location elements occupy two words and
                // 32 weights occupy one word.  These three reads preserve the
                // same element ordering while removing redundant HBM traffic.
                // Head-major physical packing makes fixed-head queries contiguous in HBM:
                // [batch][head][query][level][point].  Element order within each query
                // remains [level][point][(w,h)] / [level][point] for identical semantics.
                const int metadata_base =
                    (b_col * CONST_NUM_HEADS + m_col) * CONST_NUM_QUERIES +
                    query_idx;
                const int loc_base = metadata_base * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2;
                const int weight_base = metadata_base * CONST_NUM_LEVELS * CONST_NUM_POINTS;
                ap_uint<512> loc_word0 = sampling_locations_data[loc_base / 32];
                ap_uint<512> loc_word1 = sampling_locations_data[loc_base / 32 + 1];
                ap_uint<512> weight_word0 = attention_weights_data[weight_base / 32];
                data_t loc_buf[CONST_NUM_LEVELS * CONST_NUM_POINTS * 2];
                data_t weight_buf[CONST_NUM_LEVELS * CONST_NUM_POINTS];
                #pragma HLS ARRAY_PARTITION variable=loc_buf complete dim=1
                #pragma HLS ARRAY_PARTITION variable=weight_buf complete dim=1
                for (int j = 0; j < 32; ++j) {
                    #pragma HLS UNROLL
                    loc_buf[j].range(12, 0) = loc_word0.range(j * 16 + 12, j * 16);
                    loc_buf[32 + j].range(12, 0) = loc_word1.range(j * 16 + 12, j * 16);
                    weight_buf[j].range(12, 0) = weight_word0.range(j * 16 + 12, j * 16);
                }
                
                data_t accumulated_value[CONST_CHANNELS];
                #pragma HLS ARRAY_PARTITION variable=accumulated_value complete dim=1
                for (int c = 0; c < CONST_CHANNELS; ++c) {
                    #pragma HLS UNROLL
                    accumulated_value[c] = 0.0f;
                }
                
                LOOP_LEVELS:
                for (int l_col = 0; l_col < CONST_NUM_LEVELS; ++l_col) {
                    
                    const int spatial_h_ptr = l_col << 1;
                    const int spatial_h = CONST_SPATIAL_SHAPES[spatial_h_ptr];
                    const int spatial_w = CONST_SPATIAL_SHAPES[spatial_h_ptr + 1];
                    const int level_start_id = CONST_LEVEL_START[l_col];
                    
                    LOOP_POINTS:
                    for (int p_col = 0; p_col < CONST_NUM_POINTS; ++p_col) {
                        #pragma HLS PIPELINE
                        
                        // Metadata is read from the per-query/head cache above.
                        const int metadata_point = l_col * CONST_NUM_POINTS + p_col;
                        const data_t loc_w = loc_buf[metadata_point * 2];
                        const data_t loc_h = loc_buf[metadata_point * 2 + 1];
                        const data_t weight = weight_buf[metadata_point];
                        
                        const data_t h_im = loc_h * spatial_h - (data_t)0.5;
                        const data_t w_im = loc_w * spatial_w - (data_t)0.5;
                        
                        const int h_low = (h_im >= 0) ? (int)h_im : (int)(h_im - 1);
                        const int w_low = (w_im >= 0) ? (int)w_im : (int)(w_im - 1);
                        const int h_high = h_low + 1;
                        const int w_high = w_low + 1;

                        const data_t lh = h_im - h_low;
                        const data_t lw = w_im - w_low;
                        
                        const int valid1 = (h_low >= 0) & (w_low >= 0) & (h_low < spatial_h) & (w_low < spatial_w);
                        const int valid2 = (h_low >= 0) & (w_high >= 0) & (h_low < spatial_h) & (w_high < spatial_w);
                        const int valid3 = (h_high >= 0) & (w_low >= 0) & (h_high < spatial_h) & (w_low < spatial_w);
                        const int valid4 = (h_high >= 0) & (w_high >= 0) & (h_high < spatial_h) & (w_high < spatial_w);
                        
                        // DATA III-B: round coordinates and read the selected point directly.
                        // Preserve this adapted baseline's coordinate arithmetic exactly.
                        const int yy=h_low+(lh>=(data_t)0.5);
                        const int xx=w_low+(lw>=(data_t)0.5);
                        const bool selected_valid=yy>=0&&yy<spatial_h&&xx>=0&&xx<spatial_w;
                        const int selected_idx=selected_valid?level_start_id+yy*spatial_w+xx:0;
                        LOOP_CHANNELS_INNER:
                        for(int c_col=0;c_col<CONST_CHANNELS;++c_col){
#pragma HLS UNROLL
                            const data_t sampled_value=selected_valid?((m_col%2==0)?value_buf0[c_col][selected_idx]:value_buf1[c_col][selected_idx]):(data_t)0;
#if DATA_DSP_ADD
                            // Exact product and sum widths: no intermediate rounding or saturation.
                            // Map existing arithmetic to DSP; no extra MAC lanes or query parallelism.
                            ap_fixed<26,12> weighted=sampled_value*weight;
#pragma HLS BIND_OP variable=weighted op=mul impl=dsp latency=3
                            ap_fixed<27,13> total=weighted+accumulated_value[c_col];
#pragma HLS BIND_OP variable=total op=add impl=dsp latency=1
                            accumulated_value[c_col]=total;
#else
                            accumulated_value[c_col]+=sampled_value*weight;
#endif
                        }
                    } // End of LOOP_POINTS
                } // End of LOOP_LEVELS
                
                const int out_base_offset = 
                    b_col * CONST_NUM_QUERIES * CONST_NUM_HEADS * CONST_CHANNELS +
                    query_idx * CONST_NUM_HEADS * CONST_CHANNELS +
                    m_col * CONST_CHANNELS;
                    
                ap_uint<512> out_word = 0;
                LOOP_OUTPUT:
                for (int c_col = 0; c_col < CONST_CHANNELS; ++c_col) {
                    #pragma HLS UNROLL
                    data_t val = accumulated_value[c_col];
                    out_word.range(c_col * 16 + 15, c_col * 16) = (ap_uint<16>)val.range(12, 0);
                }
                output_data[out_base_offset / 32] = out_word;
            } // End of LOOP_NUM_QUERIES
        } // End of LOOP_NUM_HEADS
    } // End of LOOP_BATCH
}
