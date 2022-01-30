#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 56
#define N_INPUT_2_1 11
#define N_INPUT_3_1 4
#define OUT_HEIGHT_2 56
#define OUT_WIDTH_2 55
#define N_CHAN_2 4
#define OUT_HEIGHT_19 60
#define OUT_WIDTH_19 59
#define N_CHAN_19 4
#define OUT_HEIGHT_4 56
#define OUT_WIDTH_4 55
#define N_FILT_4 4
#define N_SIZE_1_7 12320
#define N_LAYER_8 256
#define N_LAYER_12 256
#define N_LAYER_16 1

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef nnet::array<ap_fixed<16,6>, 4*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 4*1> layer2_t;
typedef nnet::array<ap_fixed<16,6>, 4*1> layer3_t;
typedef nnet::array<ap_fixed<16,6>, 4*1> layer19_t;
typedef nnet::array<ap_fixed<16,6>, 4*1> layer4_t;
typedef nnet::array<ap_fixed<16,6>, 256*1> layer8_t;
typedef ap_fixed<16,6> bias8_t;
typedef nnet::array<ap_fixed<16,6>, 256*1> layer11_t;
typedef nnet::array<ap_fixed<16,6>, 256*1> layer12_t;
typedef ap_fixed<16,6> bias12_t;
typedef nnet::array<ap_fixed<16,6>, 256*1> layer15_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer16_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> result_t;

#endif
