#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_image.h"
#include "nnet_utils/nnet_image_stream.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/s3.h"
#include "weights/b3.h"
//#include "weights/w4.h"
#include "weights/b4.h"
//#include "weights/w8.h"
#include "weights/b8.h"
//#include "weights/w12.h"
#include "weights/b12.h"
//#include "weights/w16.h"
#include "weights/b16.h"

//hls-fpga-machine-learning insert layer-config
// up_sampling2d
struct config2 : nnet::resize_config {
    static const unsigned height = 56;
    static const unsigned width = 11;
    static const unsigned n_chan = 4;
    static const unsigned new_height = 56;
    static const unsigned new_width = 55;
};

// batch_normalization
struct config3 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_2*OUT_WIDTH_2*N_CHAN_2;
    static const unsigned n_filt = 4;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 100000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// zp2d_conv2d
struct config19 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_2;
    static const unsigned in_width = OUT_WIDTH_2;
    static const unsigned n_chan = N_CHAN_2;
    static const unsigned out_height = OUT_HEIGHT_19;
    static const unsigned out_width = OUT_WIDTH_19;
    static const unsigned pad_top = 2;
    static const unsigned pad_bottom = 2;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 2;
};

// conv2d
struct config4_mult : nnet::dense_config {
    static const unsigned n_in = 100;
    static const unsigned n_out = 4;
    static const unsigned reuse_factor = 5;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config4 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_19;
    static const unsigned in_width = OUT_WIDTH_19;
    static const unsigned n_chan = N_CHAN_19;
    static const unsigned filt_height = 5;
    static const unsigned filt_width = 5;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_4;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_4;
    static const unsigned out_width = OUT_WIDTH_4;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 9;
    static const unsigned min_width = 9;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config4_mult mult_config;
};
const ap_uint<config4::filt_height * config4::filt_width> config4::pixels[] = {1,3,7,15,31,30,28,24,16,33,99,231,495,1023,990,924,792,528,1057,3171,7399,15855,32767,31710,29596,25368,16912,33825,101475,236775,507375,1048575,1014750,947100,811800,541200,1082401,3247203,7576807,16236015,33554431,32472030,30307228,25977624,17318416,1082400,3247200,7576800,16236000,33554400,32472000,30307200,25977600,17318400,1082368,3247104,7576576,16235520,33553408,32471040,30306304,25976832,17317888,1081344,3244032,7569408,16220160,33521664,32440320,30277632,25952256,17301504,1048576,3145728,7340032,15728640,32505856,31457280,29360128,25165824,16777216};

// dense
struct config8 : nnet::dense_config {
    static const unsigned n_in = N_SIZE_1_7;
    static const unsigned n_out = N_LAYER_8;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 12320;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 3153920;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef bias8_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// leaky_re_lu
struct LeakyReLU_config11 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 16384;
    typedef ap_fixed<18,8> table_t;
};

// dense_1
struct config12 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned n_out = N_LAYER_12;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 16384;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 65536;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef bias12_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// leaky_re_lu_1
struct LeakyReLU_config15 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_12;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 16384;
    typedef ap_fixed<18,8> table_t;
};

// dense_2
struct config16 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_12;
    static const unsigned n_out = N_LAYER_16;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 256;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 256;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// activation_1
struct relu_config18 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 48;
    typedef ap_fixed<18,8> table_t;
};


#endif
