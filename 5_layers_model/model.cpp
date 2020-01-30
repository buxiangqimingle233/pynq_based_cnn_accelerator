#include "IO.h"
#include "core.h"

void load_and_cal(input_t* In) {
  #pragma HLS DATAFLOW
  channel_t ch_0, ch_1, ch_2, ch_3, ch_4, ch_5, ch_6;
  load_img(In, ch_0);
  conv_0_pool(ch_0, ch_1);
  conv_1(ch_1, ch_2);
  conv_2_pool(ch_2, ch_3);
  conv_3(ch_3, ch_4);
  conv_4_pool(ch_4, ch_5);
  conv_5(ch_5, ch_6);
  FC(ch_6);
}

int batch = 0;

void cnn(output_t* Out, input_t* In, input_w_t* W, input_w_t* B) {
  #pragma HLS INTERFACE s_axilite port=return
  #pragma HLS INTERFACE m_axi depth=40960 port=In offset=slave
  #pragma HLS INTERFACE m_axi depth=40960 port=Out offset=slave
  #pragma HLS INTERFACE m_axi depth=40960 port=W offset=slave
  #pragma HLS INTERFACE m_axi depth=40960 port=B offset=slave

  load_w(W);
  load_b(B);
  for (; batch < TEST_BATCH; ++batch) {
    load_and_cal(In);
  }
  store_of(Out);
}