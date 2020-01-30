#include "core.h"
#include "global_var.h"
#include "IO.h"
#include <fstream>
#include <cstring>

long long int offset = 0;


void inter_max(inter_t &res, inter_t temp[4]) {
  inter_t r0, r1;
  if (temp[0] < temp[1]) {
    r0 = temp[1];
  } else {
    r0 = temp[0];
  }

  if (temp[2] < temp[3]) {
    r1 = temp[3];
  } else {
    r1 = temp[2];
  }

  if (r0 < r1) {
    res = r1;
  } else {
    res = r0;
  }
}


void conv_0_pool(channel_t &input_s, channel_t &output_s) {
  #pragma HLS ARRAY_PARTITION variable=of_0 cyclic factor=pr_0 dim=1
  #pragma HLS ARRAY_PARTITION variable=b_0 cyclic factor=pr_0 dim=1
  #pragma HLS ARRAY_PARTITION variable=w_conv_0 cyclic factor=pr_0 dim=1
  #pragma HLS ARRAY_PARTITION variable=w_conv_0 complete dim=2
  #pragma HLS ARRAY_PARTITION variable=if_0 complete dim=1
  
  // init_w_b();

  #ifndef EXTERN
  batch:
  for (int batch = 0; batch < 500; ++batch) {
  #endif

    // set of_buffer to bias[cho]
    for (int r = 0; r < Out_0; ++r) {
      for (int c = 0; c < Out_0; ++c) {
        for (int cho = 0; cho < CHout_0; ++cho) {
          #pragma HLS PIPELINE
          #pragma HLS UNROLL factor=pr_0
          of_0[cho][r][c] = b_0[cho];
    }}}

    // if_2_t
    for (int chi = 0; chi < CHin_0; ++chi) {
      for (int r = 1; r <= In_0; ++r) {
        for (int c = 1; c <= In_0; ++c) {
        input_s >> if_0[chi][r][c];
    }}}

    // calculation
    
    krr:for (int krr = 0; krr < 3; ++krr) {
      kcc:for (int kcc = 0; kcc < 3; ++kcc) {
        r:for (int r = 0; r < Out_0; ++r) {
          c:for (int c = 0; c < Out_0; ++c) {
            cho:for (int cho = 0; cho < CHout_0; ++cho) {
              #pragma HLS UNROLL factor=pr_0
              #pragma HLS PIPELINE
              inter_t temp = of_0[cho][r][c];
              chi:for (int chi = 0; chi < CHin_0; ++chi) {
                temp += w_conv_0[cho][chi][krr][kcc] * if_0[chi][r+krr][c+kcc];
              }
              of_0[cho][r][c] = temp;
    }}}}}

    // relu & pooling
    for (int cho = 0; cho < CHout_0; ++cho) {
      for (int r = 0; r < Out_0; r += 2) {
        for (int c = 0; c < Out_0; c += 2) {
          inter_t res;
          inter_t temp[4] = {of_0[cho][r][c], of_0[cho][r+1][c],
                             of_0[cho][r][c+1], of_0[cho][r+1][c+1]};
          for (int t = 0; t < 4; t++) {
            #pragma HLS UNROLL
            if (temp[t] < 0)
              temp[t] = 0;
          }
          inter_max(res, temp);
          output_s << res;
    }}}
  #ifndef EXTERN
  }
  #endif
}


void conv_1(channel_t &input_s, channel_t &output_s) {
  #pragma HLS ARRAY_PARTITION variable=of_1 cyclic factor=pr_1 dim=1
  #pragma HLS ARRAY_PARTITION variable=b_1 cyclic factor=pr_1 dim=1
  #pragma HLS ARRAY_PARTITION variable=w_conv_1 cyclic factor=pr_1 dim=1
  #pragma HLS ARRAY_PARTITION variable=w_conv_1 complete dim=2
  #pragma HLS ARRAY_PARTITION variable=if_1 complete dim=1
  
  #ifndef EXTERN
  batch:
  for (int batch = 0; batch < 500; ++batch) {
  #endif
    // set of_buffer to bias[cho]
    for (int r = 0; r < Out_1; ++r) {
      for (int c = 0; c < Out_1; ++c) {
        for (int cho = 0; cho < CHout_1; ++cho) {
          #pragma HLS PIPELINE
          #pragma HLS UNROLL factor=pr_1
          of_1[cho][r][c] = b_1[cho];
    }}}

    // if_2_t
    for (int chi = 0; chi < CHin_1; ++chi) {
      for (int r = 1; r <= In_1; ++r) {
        for (int c = 1; c <= In_1; ++c) {
          #pragma HLS PIPELINE
        input_s >> if_1[chi][r][c];
    }}}

    // calculation
    
    krr:for (int krr = 0; krr < 3; ++krr) {
      kcc:for (int kcc = 0; kcc < 3; ++kcc) {
        r:for (int r = 0; r < Out_1; ++r) {
          c:for (int c = 0; c < Out_1; ++c) {
              cho:for (int cho = 0; cho < CHout_1; cho++) {
				        #pragma HLS UNROLL factor=pr_1
                #pragma HLS PIPELINE
                inter_t temp = of_1[cho][r][c];
                chii:for (int chii = 0; chii < 16; ++chii) {
                  temp += w_conv_1[cho][chii][krr][kcc] * if_1[chii][r+krr][c+kcc];
                }
                of_1[cho][r][c] = temp;
              }
    }}}}

    // relu & pooling
    for (int cho = 0; cho < CHout_1; ++cho) {
      for (int r = 0; r < Out_1; ++r) {
        for (int c = 0; c < Out_1; ++c) {
          #pragma HLS PIPELINE
          inter_t res = of_1[cho][r][c];
          if (res < 0) {
            res = 0;
          }
          output_s << res;
    }}}
  #ifndef EXTERN
  }
  #endif
}


void conv_2_pool(channel_t &input_s, channel_t &output_s) {
  #pragma HLS ARRAY_PARTITION variable=of_2 cyclic factor=pr_2 dim=1
  #pragma HLS ARRAY_PARTITION variable=b_2 cyclic factor=pr_2 dim=1
  #pragma HLS ARRAY_PARTITION variable=w_conv_2 cyclic factor=pr_2 dim=1
  #pragma HLS ARRAY_PARTITION variable=w_conv_2 complete dim=2
  #pragma HLS ARRAY_PARTITION variable=if_2 complete dim=1

  // init_w_b();

  #ifndef EXTERN
  batch:
  for (int batch = 0; batch < 500; ++batch) {
  #endif

    // set of_buffer to bias[cho]
      for (int cho = 0; cho < CHout_2; ++cho) {
        for (int r = 0; r < Out_2; ++r) {
          for (int c = 0; c < Out_2; ++c) {
            #pragma HLS PIPELINE
            of_2[cho][r][c] = b_2[cho];
      }}}

    // if_2_t
      for (int chi = 0; chi < CHin_2; ++chi) {
        for (int r = 1; r <= In_2; ++r) {
          for (int c = 1; c <= In_2; ++c) {
            #pragma HLS PIPELINE
            input_s >> if_2[chi][r][c];
      }}}

    // calculation
    
    krr:for (int krr = 0; krr < 3; ++krr) {
      kcc:for (int kcc = 0; kcc < 3; ++kcc) {
        r:for (int r = 0; r < Out_2; ++r) {
          c:for (int c = 0; c < Out_2; ++c) {
            chii:for (int chii = 0; chii < CHin_2; ++chii) {
              cho:for (int cho = 0; cho < CHout_2; cho++) {
				        #pragma HLS UNROLL factor=pr_2
                #pragma HLS PIPELINE
                inter_t temp = of_2[cho][r][c];
                chii:for (int chii = 0; chii < 16; ++chii) {
                  temp += w_conv_2[cho][chii][krr][kcc] * if_2[chii][r+krr][c+kcc];
                }
    }}}}}}

    // relu & pooling
    for (int cho = 0; cho < CHout_2; ++cho) {
      for (int r = 0; r < Out_2; r += 2) {
        for (int c = 0; c < Out_2; c += 2) {
          #pragma HLS PIPELINE
          inter_t res;
          inter_t temp[4] = {of_2[cho][r][c], of_2[cho][r+1][c],
                             of_2[cho][r][c+1], of_2[cho][r+1][c+1]};
          for (int t = 0; t < 4; t++) {
            #pragma HLS UNROLL
            if (temp[t] < 0)
              temp[t] = 0;
          }
          inter_max(res, temp);
          output_s << res;
    }}}
  #ifndef EXTERN
  }
  #endif
}


void conv_3(channel_t &input_s, channel_t &output_s) {
  #pragma HLS ARRAY_PARTITION variable=of_3 cyclic factor=pr_3 dim=1
  #pragma HLS ARRAY_PARTITION variable=b_3 cyclic factor=pr_3 dim=1
  #pragma HLS ARRAY_PARTITION variable=w_conv_3 cyclic factor=pr_3 dim=1
  #pragma HLS ARRAY_PARTITION variable=w_conv_3 cyclic factor=8 dim=2
  #pragma HLS ARRAY_PARTITION variable=if_3 cyclic factor=8 dim=1
  
  // init_w_b();

  #ifndef EXTERN
  batch:
  for (int batch = 0; batch < 500; ++batch) {
  #endif
    // set of_buffer to bias[cho]
    for (int r = 0; r < Out_3; ++r) {
      for (int c = 0; c < Out_3; ++c) {
        for (int cho = 0; cho < CHout_3; ++cho) {
          #pragma HLS PIPELINE
          #pragma HLS UNROLL factor=pr_3
          of_3[cho][r][c] = b_3[cho];
    }}}

    // if_2_t
    for (int chi = 0; chi < CHin_3; ++chi) {
      for (int r = 1; r <= In_3; ++r) {
        for (int c = 1; c <= In_3; ++c) {
          #pragma HLS PIPELINE
        input_s >> if_3[chi][r][c];
    }}}

    // calculation
    
    krr:for (int krr = 0; krr < 3; ++krr) {
      kcc:for (int kcc = 0; kcc < 3; ++kcc) {
        r:for (int r = 0; r < Out_3; ++r) {
          c:for (int c = 0; c < Out_3; ++c) {
            chi:for (int chi = 0; chi < CHin_3; chi += 8) {
              cho:for (int cho = 0; cho < CHout_3; ++cho) {
                #pragma HLS PIPELINE
                inter_t temp = of_3[cho][r][c];
                chii:for (int chii = 0; chii < 8; ++chii) {
                  temp += w_conv_3[cho][chi+chii][krr][kcc] * if_3[chii+chi][r+krr][c+kcc];
                }
                of_3[cho][r][c] = temp;
              }
    }}}}}

    // relu & pooling
    for (int cho = 0; cho < CHout_3; ++cho) {
      for (int r = 0; r < Out_3; ++r) {
        for (int c = 0; c < Out_3; ++c) {
          #pragma HLS PIPELINE
          inter_t res = of_3[cho][r][c];
          if (res < 0) {
            res = 0;
          }
          output_s << res;
    }}}
  #ifndef EXTERN
  }
  #endif
}


void conv_4_pool(channel_t &input_s, channel_t &output_s) {
  #pragma HLS ARRAY_PARTITION variable=of_4 cyclic factor=pr_4 dim=1
  #pragma HLS ARRAY_PARTITION variable=b_4 cyclic factor=pr_4 dim=1
  #pragma HLS ARRAY_PARTITION variable=w_conv_4 cyclic factor=16 dim=2
  #pragma HLS ARRAY_PARTITION variable=if_4 cyclic factor=16 dim=1

  // init_w_b();

  #ifndef EXTERN
  batch:
  for (int batch = 0; batch < 500; ++batch) {
  #endif
    // set of_buffer to bias[cho]
      for (int cho = 0; cho < CHout_4; ++cho) {
        for (int r = 0; r < Out_4; ++r) {
          for (int c = 0; c < Out_4; ++c) {
            #pragma HLS PIPELINE
            of_4[cho][r][c] = b_4[cho];
      }}}

      // if_4_t
      for (int chi = 0; chi < CHin_4; ++chi) {
        for (int r = 1; r <= In_4; ++r) {
          for (int c = 1; c <= In_4; ++c) {
            #pragma HLS PIPELINE
            input_s >> if_4[chi][r][c];
      }}}

    // calculation
    
    krr:for (int krr = 0; krr < 3; ++krr) {
      kcc:for (int kcc = 0; kcc < 3; ++kcc) {
        r:for (int r = 0; r < Out_4; ++r) {
          c:for (int c = 0; c < Out_4; ++c) {
            chi:for (int chi = 0; chi < CHin_4; chi += 16) {
              cho:for (int cho = 0; cho < CHout_4; ++cho) {
               #pragma HLS PIPELINE
                inter_t temp = of_4[cho][r][c];
                chii:for (int chii = 0; chii < 16; ++chii) {
                  temp += w_conv_4[cho][chi+chii][krr][kcc] * if_4[chii+chi][r+krr][c+kcc];
                }
                of_4[cho][r][c] = temp;
              }
    }}}}}

    // relu & pooling
    for (int cho = 0; cho < CHout_4; ++cho) {
      for (int r = 0; r < Out_4; r += 2) {
        for (int c = 0; c < Out_4; c += 2) {
          #pragma HLS PIPELINE
          inter_t res;
          inter_t temp[4] = {of_4[cho][r][c], of_4[cho][r+1][c],
                             of_4[cho][r][c+1], of_4[cho][r+1][c+1]};
          for (int t = 0; t < 4; t++) {
            #pragma HLS UNROLL
            if (temp[t] < 0)
              temp[t] = 0;
          }
          inter_max(res, temp);
          output_s << res;
    }}}
  #ifndef EXTERN
  }
  #endif
}


void conv_5(channel_t &input_s, channel_t &output_s) {
  #pragma HLS ARRAY_PARTITION variable=of_5 cyclic factor=pr_5 dim=1
  #pragma HLS ARRAY_PARTITION variable=b_5 cyclic factor=pr_5 dim=1
  #pragma HLS ARRAY_PARTITION variable=w_conv_5 cyclic factor=pr_5 dim=1
  #pragma HLS ARRAY_PARTITION variable=w_conv_5 cyclic factor=4 dim=2
  #pragma HLS ARRAY_PARTITION variable=if_5 cyclic factor=4 dim=1
  
  // init_w_b();

  #ifndef EXTERN
  batch:
  for (int batch = 0; batch < 500; ++batch) {
  #endif
    // set of_buffer to bias[cho]
    for (int r = 0; r < Out_5; ++r) {
      for (int c = 0; c < Out_5; ++c) {
        for (int cho = 0; cho < CHout_5; ++cho) {
          #pragma HLS PIPELINE
          #pragma HLS UNROLL factor=pr_5
          of_5[cho][r][c] = b_5[cho];
    }}}

    // if_2_t
    for (int chi = 0; chi < CHin_5; ++chi) {
      for (int r = 1; r <= In_5; ++r) {
        for (int c = 1; c <= In_5; ++c) {
          #pragma HLS PIPELINE
        input_s >> if_5[chi][r][c];
    }}}

    // calculation
    
    krr:for (int krr = 0; krr < 3; ++krr) {
      kcc:for (int kcc = 0; kcc < 3; ++kcc) {
        r:for (int r = 0; r < Out_5; ++r) {
          c:for (int c = 0; c < Out_5; ++c) {
            chi:for (int chi = 0; chi < CHin_5; chi += 4) {
              cho:for (int cho = 0; cho < CHout_5; ++cho) {
                #pragma HLS PIPELINE
                inter_t temp = of_5[cho][r][c];
                chii:for (int chii = 0; chii < 4; ++chii) {
                  temp += w_conv_5[cho][chi+chii][krr][kcc] * if_5[chii+chi][r+krr][c+kcc];
                }
                of_5[cho][r][c] = temp;
              }
    }}}}}

    // relu & pooling
    for (int cho = 0; cho < CHout_5; ++cho) {
      for (int r = 0; r < Out_5; ++r) {
        for (int c = 0; c < Out_5; ++c) {
          #pragma HLS PIPELINE
          inter_t res = of_5[cho][r][c];
          if (res < 0) {
            res = 0;
          }
          output_s << res;
    }}}
  #ifndef EXTERN
  }
  #endif
}


void FC(channel_t &input_s) {
  #ifndef EXTERN
  batch:
  for (int batch = 0; batch < 500; ++batch) {
  #endif
    // initialize of_fc
    for (int i = 0; i < 10; ++i) {
      of_fc[i] = b_fc[i];
    }
    // load if
    for (int i = 0; i < 512; ++i) {
      input_s >> if_fc[i];
    }
    // do calculation
    for (int i = 0; i < 512; ++i) {
      for (int o = 0; o < 10; ++o) {
        of_fc[o] += w_fc[o][i] * if_fc[i];
      }
    }

    // argmax
    int max_p = 0;
    for (int i = 0; i < 10; ++i) {
      if (of_fc[i] > of_fc[max_p]) {
        max_p = i;
      }
    }
    result[batch] = max_p;
  #ifndef EXTERN
  }
  #endif
}
