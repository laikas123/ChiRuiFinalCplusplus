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
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_helpers.h"

//hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

typedef ap_fixed<16,6> model_weightdefault_t;

namespace nnet {
    bool trace_enabled = true;
    std::map<std::string, void *> *trace_outputs = NULL;
    size_t trace_type_size = sizeof(double);
}

int main(int argc, char **argv)
{
  //load input data from text file
  std::ifstream fin("tb_data/tb_input_features.dat");
  //load predictions from text file
  std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
  std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
  std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
  std::ofstream fout(RESULTS_LOG);

  std::string iline;
  std::string pline;
  int e = 0;

  if (fin.is_open() && fpr.is_open()) {
    while ( std::getline(fin,iline) && std::getline (fpr,pline) ) {
      if (e % CHECKPOINT == 0) std::cout << "Processing input " << e << std::endl;
      char* cstr=const_cast<char*>(iline.c_str());
      char* current;
      std::vector<float> in;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in.push_back(atof(current));
        current=strtok(NULL," ");
      }
      cstr=const_cast<char*>(pline.c_str());
      std::vector<float> pr;
      current=strtok(cstr," ");
      while(current!=NULL) {
        pr.push_back(atof(current));
        current=strtok(NULL," ");
      }

      //hls-fpga-machine-learning insert data
      hls::stream<input_t> em_barrel("em_barrel");
      nnet::copy_data<float, input_t, 0, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1>(in, em_barrel);
      hls::stream<result_t> layer18_out("layer18_out");

      model_weightdefault_t  w4 [ 400 ];
      model_weightdefault_t  w8[ 3153920 ];
      model_weightdefault_t  w12 [ 65536 ];
      model_weightdefault_t  w22 [ 36864 ];
      model_weightdefault_t  w27 [ 73728 ];
      model_weightdefault_t  w18 [ 18432 ];
      model_weightdefault_t  w9 [ 4608 ];
      model_weightdefault_t  w13 [ 9216 ];
      model_weightdefault_t  w45 [ 589824 ];
      model_weightdefault_t  w36 [ 294912];
      model_weightdefault_t  w53 [ 256];
      model_weightdefault_t  w31 [ 147456];
      model_weightdefault_t  w49 [ 65536];
      model_weightdefault_t  w40 [ 589824];
      model_weightdefault_t  w16 [ 256];


      //hls-fpga-machine-learning insert top-level-function
      unsigned short size_in1,size_out1;
      myproject(em_barrel,layer18_out,size_in1,size_out1m, w4, w8, w12, w22, w27, w18, w9, w13, w45, w36, w53, w31, w49, w40, w16);

      if (e % CHECKPOINT == 0) {
        std::cout << "Predictions" << std::endl;
        //hls-fpga-machine-learning insert predictions
        for(int i = 0; i < N_LAYER_16; i++) {
          std::cout << pr[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Quantized predictions" << std::endl;
        //hls-fpga-machine-learning insert quantized
        nnet::print_result<result_t, N_LAYER_16>(layer18_out, std::cout, true);
      }
      e++;

      //hls-fpga-machine-learning insert tb-output
      nnet::print_result<result_t, N_LAYER_16>(layer18_out, fout);

    }
    fin.close();
    fpr.close();
  } else {
    std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

    //hls-fpga-machine-learning insert zero
    hls::stream<input_t> em_barrel("em_barrel");
    nnet::fill_zero<input_t, N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1>(em_barrel);
    hls::stream<result_t> layer18_out("layer18_out");

    model_weightdefault_t  w4 [ 400 ];
    model_weightdefault_t  w8[ 3153920 ];
    model_weightdefault_t  w12 [ 65536 ];
    model_weightdefault_t  w22 [ 36864 ];
    model_weightdefault_t  w27 [ 73728 ];
    model_weightdefault_t  w18 [ 18432 ];
    model_weightdefault_t  w9 [ 4608 ];
    model_weightdefault_t  w13 [ 9216 ];
    model_weightdefault_t  w45 [ 589824 ];
    model_weightdefault_t  w36 [ 294912];
    model_weightdefault_t  w53 [ 256];
    model_weightdefault_t  w31 [ 147456];
    model_weightdefault_t  w49 [ 65536];
    model_weightdefault_t  w40 [ 589824];
    model_weightdefault_t  w16 [ 256];

    //hls-fpga-machine-learning insert top-level-function
    unsigned short size_in1,size_out1;
    myproject(em_barrel,layer18_out,size_in1,size_out1, w4, w8, w12, w22, w27, w18, w9, w13, w45, w36, w53, w31, w49, w40, w16));

    //hls-fpga-machine-learning insert output
    nnet::print_result<result_t, N_LAYER_16>(layer18_out, std::cout, true);

    //hls-fpga-machine-learning insert tb-output
    nnet::print_result<result_t, N_LAYER_16>(layer18_out, fout);

  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
