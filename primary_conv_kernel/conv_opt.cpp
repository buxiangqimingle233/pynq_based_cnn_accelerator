#include "conv.h"
#include <iostream>

#define MAX_K 5
#define OR_PR_STEP (32)
#define OC_PR_STEP (32)
#define CHI_PR_STEP 8
#define CHO_PR_STEP 4


#define OR_BF_SIZE (OR_PR_STEP<<1)
#define OC_BF_SIZE (OR_PR_STEP<<1)
#define CHO_BF_SIZE CHO_PR_STEP
#define CHI_BF_SIZE CHI_PR_STEP


#define IR_BF_SIZE (OR_BF_SIZE+MAX_K)
#define IC_BF_SIZE (OC_BF_SIZE+MAX_K)
// is irrelevant to stride size


// the out feature map buffer's size is fixed, but it would 
// serve different number of iterations




#define MIN(left, right) ((left) < (right) ? (left) : (right))


void load_w(d_type W_buffer[CHO_BF_SIZE][CHI_BF_SIZE][MAX_K][MAX_K], d_type *W, 
	int cho, int chi, int CHin, int CHout, short K
) {
	// W applies data in a current way, thus it's impossible to unroll the loading phase
	// As a consequence, it's OK to use "MIN" macro to avoid overflow in choo and chii
	// iterations. The sufficient parts are padding with zeros.
	int krr, kcc, choo, chii;
	ch_type KK = K * K;
	ch_type C_K = CHin * KK;
//#pragma HLS RESOURCE variable=KK core=Mul_LUT
//#pragma HLS RESOURCE variable=C_K core=Mul_LUT

	// padding unsed parts with zeros

	// loading W from UltraMemory
	krr:for (krr = 0; krr < MAX_K; ++krr) {
		kcc:for (kcc = 0; kcc < MAX_K; ++kcc) {
			choo:for (choo = 0; choo < CHO_BF_SIZE; ++choo) {
				chii:for (chii = 0; chii < CHI_BF_SIZE; ++chii) {
#pragma HLS PIPELINE
					if (krr < K && kcc < K && choo < CHout-cho && chii < CHIN-chi) {
						W_buffer[choo][chii][krr][kcc] = *(W + (choo+cho)*C_K + (chii+chi)*KK + krr*K + kcc);
					}
					else {
						W_buffer[choo][chii][krr][kcc] = 0;
					}
				}
			}
		}
	}
} 


void load_i_fm(d_type if_buffer[CHI_BF_SIZE][IR_BF_SIZE][IC_BF_SIZE], d_type *In, 
		ch_type chi, ch_type r, ch_type c, ch_type R_in, ch_type C_in, ch_type CHin, ch_type S, ch_type K
) {
	// Just same with loading phase
	ch_type rr, cc, chii;

	// padding sufficient parts with zeros
	ch_type R_C = R_in*C_in;
//#pragma HLS RESOURCE variable=R_C core=Mul_LUT
	rr:for (rr = 0; rr < IR_BF_SIZE; ++rr) {
		cc:for (cc = 0; cc < IC_BF_SIZE; ++cc) {
			chii:for (chii = 0; chii < CHI_BF_SIZE; ++chii) {
#pragma HLS PIPELINE
				if (rr < R_in-r && cc < C_in-c && chii < CHin-chi) {
					ch_type r_S = r*S;
//#pragma HLS RESOURCE variable=r_S core=Mul_LUT
					if_buffer[chii][rr][cc] = *(In + (chii+chi)*R_C + (rr+r_S)*C_in + (cc+c*S));
				}
				else {
					if_buffer[chii][rr][cc] = 0;
				}
			}
		}
	}
}


void calculate(d_type of_buffer[CHO_BF_SIZE][OR_BF_SIZE][OC_BF_SIZE], d_type if_buffer[CHI_BF_SIZE][IR_BF_SIZE][IC_BF_SIZE], 
	d_type W_buffer[CHO_BF_SIZE][CHI_BF_SIZE][MAX_K][MAX_K], int K, int S) 
{
	int kr, kc, rr, cc, choo, chii;
	Kc:for (kc = 0; kc < K; kc++) {
		Kr:for (kr = 0; kr < K; kr++) {
			rr:for (rr = 0; rr < OR_PR_STEP; rr++) {
				cc:for (cc = 0; cc < OC_PR_STEP; cc++) {
#pragma HLS PIPELINE
					choo:for (choo = 0; choo < CHO_PR_STEP; choo++) {
						d_type temp = 0;
						chii:for (chii = 0; chii < CHI_PR_STEP; chii++) {

								of_buffer[choo][rr][cc] += W_buffer[choo][chii][kr][kc] * if_buffer[chii][S*rr + kr][S*cc + kc];

						}
					}
				}
			}  
		}
	}
}

void store_o_fm(d_type *Out, d_type of_buffer[CHO_BF_SIZE][OR_BF_SIZE][OC_BF_SIZE], 
	d_type of_buffer_[CHO_BF_SIZE][OR_BF_SIZE][OC_BF_SIZE], 
	int cho, int r, int c, int R_out, int C_out, int CHout, int S
) {
	// store appropriate locations
	ch_type choo, rr, cc;
	ch_type R_C = R_out * C_out;

//#pragma HLS RESOURCE variable=R_C core=Mul_LUT

	rr:for (rr = 0; rr < OR_PR_STEP; ++rr) {
		cc:for (cc = 0; cc < OC_PR_STEP; ++cc) {
			choo:for (choo = 0; choo < CHO_PR_STEP; ++choo) {
#pragma HLS PIPELINE
				// FIXME: assume that Out is in shape [CHOUT, R, C]
				if (rr < R_out-r && cc < C_out-c && choo < CHout-cho) {
					ch_type C_R_C = (choo+cho)*R_C;
//#pragma HLS RESOURCE variable=C_R_C core=Mul_LUT
					ch_type r_C_o = (rr+r)*C_out;
//#pragma HLS RESOURCE variable=r_C_o core=Mul_LUT
					*(Out + C_R_C + r_C_o + (cc+c)) = of_buffer[choo][rr][cc] + of_buffer_[choo][rr][cc];
				}				
			}
		}
	}
}



void set_of_zero(d_type of_buffer[CHO_BF_SIZE][OR_BF_SIZE][OC_BF_SIZE]) {
	for (int j = 0; j < OR_BF_SIZE; j++) {
		for (int k = 0; k < OC_BF_SIZE; k++) {
			for (int i = 0; i < CHO_BF_SIZE; i++) {
#pragma HLS UNROLL
				of_buffer[i][j][k] = 0;
			}
		}
	}
}

void cnn(d_type *In, d_type *Out, d_type *W, int *Parameter) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE m_axi depth=40960 port=In offset=slave           //adjust the depth as you need
#pragma HLS INTERFACE m_axi depth=40960 port=Out offset=slave
#pragma HLS INTERFACE m_axi depth=40960 port=W offset=slave
#pragma HLS INTERFACE m_axi depth=256 port=Parameter offset=slave

  // preparation

	int parameter_buffer[NParameter];
#pragma HLS ARRAY_PARTITION variable=parameter_buffer complete dim=1
	memcpy((void*)parameter_buffer, (const int*)Parameter, NParameter *sizeof(int));


	int CHin, CHout, R_in, C_in, K, S, R_out, C_out;
	CHin = parameter_buffer[0];
	CHout = parameter_buffer[1];
	R_in = parameter_buffer[2];
	C_in = parameter_buffer[3];
	K = parameter_buffer[4];
	S = parameter_buffer[5];
	R_out = (R_in - K + S) / S;
	C_out = (C_in - K + S) / S;

	int r, c, cho, chi;
	int rr, cc, choo, chii;
	int kr, kc;
	d_type w_ofs, if_ofs, of_ofs;

	d_type of_buffer_0[CHO_BF_SIZE][OR_BF_SIZE][OC_BF_SIZE];
	d_type of_buffer_1[CHO_BF_SIZE][OR_BF_SIZE][OC_BF_SIZE];
#pragma HLS ARRAY_PARTITION variable=of_buffer_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=of_buffer_1 complete dim=1
	d_type if_buffer_0[CHI_BF_SIZE][IR_BF_SIZE][IC_BF_SIZE];
	d_type if_buffer_1[CHI_BF_SIZE][IR_BF_SIZE][IC_BF_SIZE];
#pragma HLS ARRAY_PARTITION variable=if_buffer_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=if_buffer_1 complete dim=1
	d_type W_buffer_0[CHO_BF_SIZE][CHI_BF_SIZE][MAX_K][MAX_K];
	d_type W_buffer_1[CHO_BF_SIZE][CHI_BF_SIZE][MAX_K][MAX_K];
#pragma HLS ARRAY_PARTITION variable=W_buffer_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=W_buffer_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=W_buffer_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=W_buffer_1 complete dim=1


	cho:for (cho = 0; cho < CHout; cho += CHO_PR_STEP) {
		r:for (r = 0; r < R_out; r += OR_PR_STEP) {
			c:for (c = 0; c < C_out; c += OC_PR_STEP) {

				set_of_zero(of_buffer_0);
				set_of_zero(of_buffer_1);

				bool flag = true;
				chi:for (chi = 0; chi < CHin; chi += CHI_PR_STEP) {
					if (flag) {
						load_i_fm(if_buffer_0, In, chi, r, c, R_in, C_in, CHin, S, K);
						load_w(W_buffer_0, W, cho, chi, CHin, CHout, K);
						if (chi != 0) {
							calculate(of_buffer_1, if_buffer_1, W_buffer_1, K, S);
						} 
					}
					else {
						load_i_fm(if_buffer_1, In, chi, r, c, R_in, C_in, CHin, S, K);
						load_w(W_buffer_1, W, cho, chi, CHin, CHout, K);
						calculate(of_buffer_0, if_buffer_0, W_buffer_0, K, S);
					}
					flag = !flag;
				}
				if (flag) {
					calculate(of_buffer_1, if_buffer_1, W_buffer_1, K, S);
				} else {
					calculate(of_buffer_0, if_buffer_0, W_buffer_0, K, S);
				}
				// store output feature maps
				store_o_fm(Out, of_buffer_0, of_buffer_1, cho, r, c, R_out, C_out, CHout, S);
			}
		}
	}
}
