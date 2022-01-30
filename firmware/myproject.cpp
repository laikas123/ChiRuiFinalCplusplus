//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

typedef ap_fixed<16,6> model_weightdefault_t;

void myproject(
    hls::stream<input_t> &em_barrel,
    hls::stream<result_t> &layer18_out,
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1,
    model_weightdefault_t  w4 [ 400 ],
    model_weightdefault_t  w8[ 3153920 ],
    model_weightdefault_t  w12 [ 65536 ];
    model_weightdefault_t  w22 [ 36864 ],
    model_weightdefault_t  w27 [ 73728 ],
    model_weightdefault_t  w18 [ 18432 ],
    model_weightdefault_t  w9 [ 4608 ],
    model_weightdefault_t  w13 [ 9216 ],
    model_weightdefault_t  w45 [ 589824 ],
    model_weightdefault_t  w36 [ 294912],
    model_weightdefault_t  w53 [ 256],
    model_weightdefault_t  w31 [ 147456],
    model_weightdefault_t  w49 [ 65536],
    model_weightdefault_t  w40 [ 589824],
    model_weightdefault_t  w16 [ 256]

) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=em_barrel,layer18_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1;
    const_size_out_1 = N_LAYER_16;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 4>(s3, "s3.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(b3, "b3.txt");
        nnet::load_weights_from_txt<model_default_t, 400>(w4, "w4.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 3153920>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 256>(b8, "b8.txt");
        nnet::load_weights_from_txt<model_default_t, 65536>(w12, "w12.txt");
        nnet::load_weights_from_txt<bias12_t, 256>(b12, "b12.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(w16, "w16.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b16, "b16.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=3080
    nnet::resize_nearest<input_t, config2>(em_barrel, layer2_out); // up_sampling2d

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=3080
    nnet::normalize<layer2_t, layer3_t, config3>(layer2_out, layer3_out, s3, b3); // batch_normalization

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=3540
    nnet::zeropad2d_cl<layer3_t, layer19_t, config19>(layer3_out, layer19_out); // zp2d_conv2d

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=3080
    nnet::conv_2d_cl<layer19_t, layer4_t, config4>(layer19_out, layer4_out, w4, b4); // conv2d

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=1
    nnet::dense<layer6_t, layer8_t, config8>(layer6_out, layer8_out, w8, b8); // dense

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=1
    nnet::leaky_relu<layer8_t, layer11_t, LeakyReLU_config11>(layer8_out, 0.30000001192092896, layer11_out); // leaky_re_lu

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=1
    nnet::dense<layer11_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12); // dense_1

    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=1
    nnet::leaky_relu<layer12_t, layer15_t, LeakyReLU_config15>(layer12_out, 0.30000001192092896, layer15_out); // leaky_re_lu_1

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=1
    nnet::dense<layer15_t, layer16_t, config16>(layer15_out, layer16_out, w16, b16); // dense_2

    nnet::relu<layer16_t, result_t, relu_config18>(layer16_out, layer18_out); // activation_1

}
