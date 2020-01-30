#include <string.h>
#include <cstdio>
#include "winograd.h"


/* ============== buffer ============== */

data_t if_buffer[bfsize_chi][tilesize_ir][tilesize_ic];
data_t W_buffer[bfsize_cho][bfsize_chi][K][K];
data_t of_buffer[tilenum_oc][bfsize_cho][tilesize_or][tilesize_oc];


/* =========== top function =========== */

void cnn(input_t *In, output_t *Out, input_t *W, int *Parameter) {

#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE m_axi depth=40960 port=In offset=slave  //adjust the depth as you need
#pragma HLS INTERFACE m_axi depth=40960 port=Out offset=slave
#pragma HLS INTERFACE m_axi depth=40960 port=W offset=slave
#pragma HLS INTERFACE m_axi depth=256 port=Parameter offset=slave

#pragma HLS data_pack variable=W struct_level
#pragma HLS data_pack variable=Out struct_level
#pragma HLS data_pack variable=In struct_level
/* ============= OPT para ============== */

#pragma HLS ARRAY_PARTITION variable=W_buffer complete dim=4
#pragma HLS ARRAY_PARTITION variable=W_buffer complete dim=3

#pragma HLS ARRAY_PARTITION variable=of_buffer complete dim=3
#pragma HLS ARRAY_PARTITION variable=of_buffer complete dim=4

#pragma HLS ARRAY_PARTITION variable=if_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=if_buffer complete dim=3

  int parameter_buffer[6];
	memcpy((void*)parameter_buffer, (const int*)Parameter, 6*sizeof(int));
#pragma HLS ARRAY_PARTITION variable=parameter_buffer complete dim=1
  int ch_in, ch_out, r_in, c_in, r_out, c_out;
  ch_in = parameter_buffer[0];
  ch_out = parameter_buffer[1];
  r_in = parameter_buffer[2];
  c_in = parameter_buffer[3];
  r_out = (r_in - 3 + STR) / STR;
  c_out = (c_in - 3 + STR) / STR;


  /*
    In: CHIN x RIN x CIN
    W: CHIN x CHOUT x 6 x 6
    Out: CHOUT x ROUT x COUT
  */

  int cho, chi, r, c, tm;
  // load all output channels in Weight
  // c-chi-r, weights are loaded c times(waste: cho x chi x K x K)

  load_w(W_buffer, W, 0, ch_in, ch_out); // all cho
  cnn_label2:for (r = 0; r < r_out; r += tilesize_or) {

    set_of_zero(of_buffer); // the whole line

    // TODO: load all input channels of w & apply data_pack
    cnn_label1:for (chi = 0; chi < ch_in; chi += prfactor_chi) {
      
      // load = calculate
      tm = c = 0;
      // a new input channel
      load_if_next_chi(if_buffer, In, chi, r, c, ch_in, r_in, c_in);
      calculate(of_buffer[tm], if_buffer, W_buffer);

      for (tm = 1, c = tilesize_oc; tm < tilenum_oc; ++tm, c += tilesize_oc) {
        load_if_shift(if_buffer, In, chi, r, c, ch_in, r_in, c_in);
        calculate(of_buffer[tm], if_buffer, W_buffer);
      }
      
    }
    store_of(Out, of_buffer, r, r_out, c_out, ch_out);
  }
  
}
