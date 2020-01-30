#include "IO.h"


const int chin_conv[6] = {3, 16, 32, 32, 32, 32};
const int chout_conv[6] = {16, 32, 32, 32, 32, 32};


static void do_load_conv_w(input_w_t* W, w_buf_t* dst, int layer, long long int* offset_p) {
  // TODO: need optimization?
  // #pragma HLS ARRAY_PARTITION variable=dst factor=8 dim=2
  #pragma HLS DATA_PACK variable=W

  int CHin = chin_conv[layer], CHout = chout_conv[layer];
  long long int offset = *offset_p;
  input_w_t tmp;

  for (int rr = 0; rr < 3; ++rr) {
    for (int cc = 0; cc < 3; ++cc) {
      for (int chi = 0; chi < CHin; chi++) {
        for (int cho = 0; cho < CHout; cho += PRTCL_PR_FC) {
          tmp = *(W + (offset + rr*3*CHin*CHout + cc*CHin*CHout + chi*CHout + cho) / PRTCL_PR_FC);
          for (int f = 0; f < PRTCL_PR_FC; ++f) {
            (*dst)[cho + f][chi][rr][cc] = tmp.fac[f];
          }
  }}}}

  *offset_p = offset + CHin*CHout*9;
}


static void do_load_conv_w_0(input_w_t* W, weight_t (*dst)[16][3][3][3], int layer, long long int* offset_p) {
  // TODO: need optimization?
  // #pragma HLS ARRAY_PARTITION variable=dst factor=8 dim=2
  #pragma HLS DATA_PACK variable=W

  int CHin = 3, CHout = 16;
  long long int offset = *offset_p;
  input_w_t tmp;

  for (int rr = 0; rr < 3; ++rr) {
    for (int cc = 0; cc < 3; ++cc) {
      for (int chi = 0; chi < CHin; chi++) {
        for (int cho = 0; cho < CHout; cho += PRTCL_PR_FC) {
          tmp = *(W + (offset + rr*3*CHin*CHout + cc*CHin*CHout + chi*CHout + cho) / PRTCL_PR_FC);
          for (int f = 0; f < PRTCL_PR_FC; ++f) {
            (*dst)[cho + f][chi][rr][cc] = tmp.fac[f];
          }
  }}}}

  *offset_p = offset + CHin*CHout*9;
}

static void do_load_conv_w_1(input_w_t* W, weight_t (*dst)[32][16][3][3], int layer, long long int* offset_p) {
  // TODO: need optimization?
  // #pragma HLS ARRAY_PARTITION variable=dst factor=8 dim=2
  #pragma HLS DATA_PACK variable=W

  int CHin = 16, CHout = 32;
  long long int offset = *offset_p;
  input_w_t tmp;

  for (int rr = 0; rr < 3; ++rr) {
    for (int cc = 0; cc < 3; ++cc) {
      for (int chi = 0; chi < CHin; chi++) {
        for (int cho = 0; cho < CHout; cho += PRTCL_PR_FC) {
          tmp = *(W + (offset + rr*3*CHin*CHout + cc*CHin*CHout + chi*CHout + cho) / PRTCL_PR_FC);
          for (int f = 0; f < PRTCL_PR_FC; ++f) {
            (*dst)[cho + f][chi][rr][cc] = tmp.fac[f];
          }
  }}}}

  *offset_p = offset + CHin*CHout*9;
}


static void do_load_fc(input_w_t* W, weight_t dst[10][512], long long int* offset_p) {
  
  long long int offset = *offset_p;
  input_w_t tmp;

  for (int o = 0; o < 10; ++o) {
    for (int i = 0; i < 512; i += PRTCL_PR_FC) {
      tmp = *(W + (offset + o * 512 + i) / PRTCL_PR_FC);
      for (int f = 0; f < PRTCL_PR_FC; ++f) {
        dst[o][i + f] = tmp.fac[f];
      }
    }
  }

  *offset_p = offset + 10 * 512;
}


static void do_load_b(input_w_t* B, weight_t* dst, int CHout, long long int* offset_p) {
  #pragma HLS DATA_PACK variable=B
  
  input_w_t tmp;

  long long int offset = *offset_p;
  for (int o = 0; o < CHout; o += PRTCL_PR_FC) {
    tmp = *(B + (offset + o) / PRTCL_PR_FC);
    for (int i = 0; i < PRTCL_PR_FC; ++i) {
      dst[i + o] = tmp.fac[i];
    }
  }

  *offset_p = offset + CHout;
}


void load_w(input_w_t* W) {
  #pragma HLS DATA_PACK variable=W
  // #pragma HLS allocation instances=do_load_conv_w limit=1 function
  #pragma HLS INTERFACE m_axi depth=40960 port=W offset=slave
  // #pragma HLS ARRAY_PARTITION variable=w_conv_4 cyclic factor=16 dim=2
  // #pragma HLS ARRAY_PARTITION variable=w_conv_5 cyclic factor=16 dim=2

  long long int offset = 0;
  
  // do_load_conv_w_0(W, &w_conv_0, 0, &offset);
  // do_load_conv_w_1(W, &w_conv_1, 1, &offset);
  int CHin, CHout;
  input_w_t tmp;

  // to decease the use of BRAM and LUT (owes to some odd features of HLS)
  CHin = 3;
  CHout = 16;
  for (int rr = 0; rr < 3; ++rr) {
    for (int cc = 0; cc < 3; ++cc) {
      for (int chi = 0; chi < CHin; chi++) {
        for (int cho = 0; cho < CHout; cho += PRTCL_PR_FC) {
          tmp = *(W + (offset + rr*3*CHin*CHout + cc*CHin*CHout + chi*CHout + cho) / PRTCL_PR_FC);
          for (int f = 0; f < PRTCL_PR_FC; ++f) {
            w_conv_0[cho + f][chi][rr][cc] = tmp.fac[f];
          }
  }}}}
  offset += CHin * CHout * 9;

  CHin = 16;
  CHout = 32;
  for (int rr = 0; rr < 3; ++rr) {
    for (int cc = 0; cc < 3; ++cc) {
      for (int chi = 0; chi < CHin; chi++) {
        for (int cho = 0; cho < CHout; cho += PRTCL_PR_FC) {
          tmp = *(W + (offset + rr*3*CHin*CHout + cc*CHin*CHout + chi*CHout + cho) / PRTCL_PR_FC);
          for (int f = 0; f < PRTCL_PR_FC; ++f) {
            w_conv_1[cho + f][chi][rr][cc] = tmp.fac[f];
          }
  }}}}

  offset += CHin * CHout * 9;

  do_load_conv_w(W, &w_conv_2, 2, &offset);
  do_load_conv_w(W, &w_conv_3, 3, &offset);
  do_load_conv_w(W, &w_conv_4, 4, &offset);
  do_load_conv_w(W, &w_conv_5, 5, &offset);
  do_load_fc(W, w_fc, &offset);
}


void load_b(input_w_t* B) {
  #pragma HLS DATA_PACK variable=B
  #pragma HLS allocation instances=do_load_b limit=1 function
  long long int offset = 0;

  do_load_b(B, b_0, chout_conv[0], &offset);
  do_load_b(B, b_1, chout_conv[1], &offset);
  do_load_b(B, b_2, chout_conv[2], &offset);
  do_load_b(B, b_3, chout_conv[3], &offset);
  do_load_b(B, b_4, chout_conv[4], &offset);
  do_load_b(B, b_5, chout_conv[5], &offset);
  do_load_b(B, b_fc, 10, &offset);
}

#ifdef EXTERN
long long int layer_offset = 0;
#endif

void load_img(input_t* In, channel_t& ch) {

  input_t tmp;

  #ifndef EXTERN
  long long int layer_offset = 0;
  batch:
  for (int batch = 0; batch < 500; ++batch) {
  #endif

    for (int chi = 0; chi < CHin_0; ++chi) {
      for (int r = 0; r < IMG_SIZE; ++r) {
        for (int c = 0; c < IMG_SIZE; c += PRTCL_PR_FC) {
          tmp = *(In + (layer_offset + chi*IMG_SIZE*IMG_SIZE + r*IMG_SIZE + c) / PRTCL_PR_FC);
          for (int f = 0; f < PRTCL_PR_FC; ++f) {
            ch << 4 * ((inter_t)tmp.fac[f] - 128) / 256;
          }
      }}
    }
    layer_offset += CHin_0 * IMG_SIZE * IMG_SIZE;
  #ifndef EXTERN
  }
  #endif

}


void store_of(output_t* Out) {
  for (int i = 0; i < 500; ++i) {
    Out[i] = result[i];
  }
}
