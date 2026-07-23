/**
 * @file msda_top.cpp
 * @brief MSDA HLS Implementation - Complete Version
 */

 #include "msda_top.hpp"
 #include <cmath>
 #include <iostream>
 #include <algorithm>
 #include <cassert>

// ===== Bilinear Interpolation Function =====
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
 
void msda_top(
    const data_t* value_data,
    const data_t* sampling_locations_data,
    const data_t* attention_weights_data,
    data_t* output_data
) {
    // ===== M_AXI Interface Settings =====
    #pragma HLS INTERFACE m_axi port=value_data offset=slave bundle=Gmem0 depth=32640000 max_read_burst_length=256 max_widen_bitwidth=512 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=sampling_locations_data offset=slave bundle=Gmem1 depth=29681664 max_read_burst_length=256 max_widen_bitwidth=512 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=attention_weights_data offset=slave bundle=Gmem2 depth=14840832 max_read_burst_length=256 max_widen_bitwidth=512 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=output_data offset=slave bundle=Gmem3 depth=14840832 max_write_burst_length=256 max_widen_bitwidth=512 num_write_outstanding=16
    
    // Control interface
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    
    // Read constants directly from header file (no local copy needed)
    
    int output_index = 0;

    // Only compute once the fixed stride related to channels/heads
    const int qid_stride = CONST_NUM_HEADS * CONST_CHANNELS;
    
    // ===== Correct loop order =====
    LOOP_BATCH:
    for (int b_col = 0; b_col < CONST_BATCH_SIZE; ++b_col) {
        // Start offset in value tensor for this batch (independent of queries/channels)
        const int data_value_ptr_init_offset = b_col * CONST_SPATIAL_SIZE * qid_stride;
        LOOP_NUM_QUERIES:
        for (int query_idx = 0; query_idx < CONST_NUM_QUERIES; ++query_idx) {
            LOOP_NUM_HEADS:
            for (int m_col = 0; m_col < CONST_NUM_HEADS; ++m_col) {
                LOOP_CHANNELS:  // Channel loop is in the 4th layer, not the outermost layer!
                for (int c_col = 0; c_col < CONST_CHANNELS; ++c_col) {
                        #pragma HLS PIPELINE II=1
                        
                    // Initialize accumulator for current channel
                    data_t accumulated_value = 0.0f;
                    
                    LOOP_LEVELS:
                    for (int l_col = 0; l_col < CONST_NUM_LEVELS; ++l_col) {
                        LOOP_POINTS:
                        for (int p_col = 0; p_col < CONST_NUM_POINTS; ++p_col) {
                            
                            // Get data from local array (only compute once per level)
                            const int spatial_h_ptr = l_col << 1;
                            const int spatial_h = CONST_SPATIAL_SHAPES[spatial_h_ptr];
                            const int spatial_w = CONST_SPATIAL_SHAPES[spatial_h_ptr + 1];
                            const int level_start_id = CONST_LEVEL_START[l_col];

                            // Base address and stride for current level (independent of p_col)
                            const data_t* data_value_ptr = value_data + (data_value_ptr_init_offset + level_start_id * qid_stride);
                            const int w_stride = CONST_NUM_HEADS * CONST_CHANNELS;
                            const int h_stride = spatial_w * w_stride;
                            const int mc_offset = m_col * CONST_CHANNELS + c_col;

                            // Calculate sampling location index (channel-independent, excluding c_col)
                            const int loc_ptr_offset = 
                                b_col * CONST_NUM_QUERIES * CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
                                query_idx * CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
                                m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS * 2 +
                                l_col * CONST_NUM_POINTS * 2 +
                                p_col * 2;
                            
                            // Calculate weight index (channel-independent, excluding c_col)
                            const int weight_ptr_offset =
                                b_col * CONST_NUM_QUERIES * CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS +
                                query_idx * CONST_NUM_HEADS * CONST_NUM_LEVELS * CONST_NUM_POINTS +
                                m_col * CONST_NUM_LEVELS * CONST_NUM_POINTS +
                                l_col * CONST_NUM_POINTS +
                                p_col;
                            
                            // Get sampling coordinates and weights
                            const data_t loc_w = sampling_locations_data[loc_ptr_offset];
                            const data_t loc_h = sampling_locations_data[loc_ptr_offset + 1];
                            const data_t weight = attention_weights_data[weight_ptr_offset];
                            
                            // Coordinate transformation
                            const data_t h_im = loc_h * spatial_h - 0.5;
                            const data_t w_im = loc_w * spatial_w - 0.5;
                            
                            // Calculate 4 corner coordinates of bilinear interpolation
                        const int h_low = (h_im >= 0) ? (int)h_im : (int)(h_im - 1);
                        const int w_low = (w_im >= 0) ? (int)w_im : (int)(w_im - 1);
                        const int h_high = h_low + 1;
                        const int w_high = w_low + 1;

                            // Calculate weights
                            const data_t lh = h_im - h_low;
                            const data_t lw = w_im - w_low;
                            
                            // Pre-calculate base offset (using current channel c_col)
                            const int base_offset = h_low * h_stride + w_low * w_stride + mc_offset;
                            
                            // Boundary checks
                            const int valid1 = (h_low >= 0) & (w_low >= 0) & (h_low < spatial_h) & (w_low < spatial_w);
                            const int valid2 = (h_low >= 0) & (w_high >= 0) & (h_low < spatial_h) & (w_high < spatial_w);
                            const int valid3 = (h_high >= 0) & (w_low >= 0) & (h_high < spatial_h) & (w_low < spatial_w);
                            const int valid4 = (h_high >= 0) & (w_high >= 0) & (h_high < spatial_h) & (w_high < spatial_w);
                            
                            // Calculate addresses of 4 corners (using current channel c_col)
                            const int addr1 = base_offset;
                            const int addr2 = base_offset + (w_high - w_low) * w_stride;
                            const int addr3 = base_offset + (h_high - h_low) * h_stride;
                            const int addr4 = base_offset + (h_high - h_low) * h_stride + (w_high - w_low) * w_stride;
                            
                            // Serial read data of 4 corners (single-port M_AXI)
                            data_t v1, v2, v3, v4;
                            
                            // Read v1
                            if (valid1) {
                                v1 = data_value_ptr[addr1];
                            } else {
                                v1 = 0.0f;
                            }
                            
                            // Read v2
                            if (valid2) {
                                v2 = data_value_ptr[addr2];
                            } else {
                                v2 = 0.0f;
                            }
                            
                            // Read v3
                            if (valid3) {
                                v3 = data_value_ptr[addr3];
                            } else {
                                v3 = 0.0f;
                            }
                            
                            // Read v4
                            if (valid4) {
                                v4 = data_value_ptr[addr4];
                            } else {
                                v4 = 0.0f;
                            }
                            
                            // Bilinear interpolation calculation
                            const data_t sampled_value = bilinear_interpolation(v1, v2, v3, v4, lh, lw);
                            
                            // Accumulate to current channel
                            accumulated_value += sampled_value * weight;
                        } // End of LOOP_POINTS
                    } // End of LOOP_LEVELS
                    
                    // Store result of current channel
                    output_data[output_index++] = accumulated_value;
                } // End of LOOP_CHANNELS
            } // End of LOOP_NUM_HEADS
        } // End of LOOP_NUM_QUERIES
    } // End of LOOP_BATCH
}