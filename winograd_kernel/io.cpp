#include "winograd.h"

void load_if_shift(data_t if_buffer[bfsize_chi][tilesize_ir][tilesize_ic], input_t* In, 
  int chi, int r, int c, int ch_in, int r_in, int c_in) 
{
#pragma HLS data_pack variable=In struct_level
  input_t temp;
  // correctness is guaranteed by pherical code
  int chii, rr, cc;
  chii:
  for (chii = 0; chii < bfsize_chi; ++chii) {
    // shift left
    for (cc = 0; cc < tilesize_ic - tilesize_oc; ++cc) {
#pragma HLS UNROLL
      for (rr = 0; rr < tilesize_ir; ++rr) {
#pragma HLS UNROLL
        if_buffer[chii][rr][cc] = if_buffer[chii][rr][cc + tilesize_oc];
      }
    }
  }
    
    // load new features
  for (cc = tilesize_ic - tilesize_oc; cc < tilesize_ic; cc++) {
    for (rr = 0; rr < tilesize_ir; ++rr) {
      for (chii = 0; chii < bfsize_chi; chii += 2) {
#pragma HLS PIPELINE
        if ((chii + chi < ch_in) && (rr + r < r_in) && (cc + c < c_in)) {
          temp = *(In + ((cc + c)*r_in*ch_in + (rr + r)*ch_in + chii + chi) / 2);
          if_buffer[chii][rr][cc] = temp.a[0];
          if_buffer[chii + 1][rr][cc] = temp.a[1];
        } else {
          if_buffer[chii][rr][cc] = 0;
          if_buffer[chii + 1][rr][cc] = 0; 
        }
      }    
    }
  }
}


void load_if_next_chi(data_t if_buffer[bfsize_chi][tilesize_ir][tilesize_ic], input_t* In, 
  int chi, int r, int c, int ch_in, int r_in, int c_in)
{
  int chii, rr, cc;
#pragma HLS data_pack variable=In struct_level
  input_t temp;
  cc:
  for (cc = 0; cc < tilesize_ic; cc ++) {
    rr:
    for (rr = 0; rr < tilesize_ir; ++rr) {
      chii:
      for (chii = 0; chii < bfsize_chi; chii += 2) {
#pragma HLS PIPELINE
        if (chii + chi < ch_in && rr + r < r_in && cc + c < c_in) {
          temp = *(In + ((cc + c)*r_in*ch_in + (rr + r)*ch_in + chii+chi) / 2);
          if_buffer[chii][rr][cc] = temp.a[0];
          if_buffer[chii + 1][rr][cc] = temp.a[1];
        } else {
          if_buffer[chii][rr][cc] = 0;
          if_buffer[chii + 1][rr][cc] = 0;
        }
      }
    }
  }
}



void load_w(data_t W_buffer[bfsize_cho][bfsize_chi][K][K], input_t* W, 
  int chi, int ch_in, int ch_out) 
{
#pragma HLS data_pack variable=W struct_level
	input_t temp;
//	#pragma HLS data_pack variable=temp;

  int choo, chii, krr, kcc;
  for (choo = 0; choo < bfsize_cho; ++choo) {
    for (chii = 0; chii < bfsize_chi; ++chii) {
      for (krr = 0; krr < K; ++krr) {
        for (kcc = 0; kcc < K; kcc += 2) {
#pragma HLS PIPELINE
          if (choo < ch_out && chii + chi < ch_in) {
            temp = *(W + (choo*CHIN*K*K + (chii+chi)*K*K + krr*K + kcc) / 2);
            W_buffer[choo][chii][krr][kcc] = temp.a[0];
            W_buffer[choo][chii][krr][kcc + 1] = temp.a[1];
          } else {
            W_buffer[choo][chii][krr][kcc] = 0;
            W_buffer[choo][chii][krr][kcc + 1] = 0;
          }
        }
      }
    }
  }
}


void set_of_zero(data_t of_buffer[tilenum_oc][bfsize_cho][tilesize_or][tilesize_oc]) {
  int cho, rr, cc, nn;
  for (nn = 0; nn < tilenum_oc; ++nn) {
    for (cho = 0; cho < bfsize_cho; ++cho) {
#pragma HLS PIPELINE
      for (rr = 0; rr < tilesize_or; ++rr) {
        for (cc = 0; cc < tilesize_oc; ++cc) {
          of_buffer[nn][cho][rr][cc] = 0;
        }
      }
    }
  }
}


void store_of(output_t *Out, data_t of_buffer[tilenum_oc][bfsize_cho][tilesize_or][tilesize_oc],
  int r, int r_out, int c_out, int ch_out) 
{
#pragma HLS data_pack variable=Out struct_level
  int tm, choo, cc, rr, c;
  output_t temp;


  choo:
  for (choo = 0; choo < bfsize_cho; ++choo) {
    trr:
    for (rr = 0; rr < tilesize_or; ++rr) {
      tm:
      for (tm = 0; tm < tilenum_oc; ++tm) {
        cc:
        for (cc = 0; cc < tilesize_oc; cc += 2) {
#pragma HLS PIPELINE
          c = tm * tilesize_oc + cc;
          temp.a[0] = of_buffer[tm][choo][rr][cc];
          temp.a[1] = of_buffer[tm][choo][rr][cc + 1];
          // FIXME: need to be adapted
          if (rr + r < r_out && c < c_out && choo < ch_out) {
            *(Out + (choo*r_out*c_out + (rr+r)*c_out + c) / 2) = temp;
          }
        }
      }
    }
  }
}
