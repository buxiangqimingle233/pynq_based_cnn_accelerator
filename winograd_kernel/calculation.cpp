#include "winograd.h"
#include <iostream>

static float max_bt_z = 0, max_at_T = 0, max_u = 0, max_T = 0;

static const data_t At[4][6] = {
  {1, 1, 1, 1, 1, 0},
  {0, 1, -1, 2, -2, 0},
  {0, 1, 1, 4, 4, 0},
  {0, 1, -1, 8, -8, 1}
};
static const data_t Bt[6][6] = {
  {4, 0, -5, 0, 1, 0},
  {0, -4, -4, 1, 1, 0},
  {0, 4, -4, -1, 1, 0},
  {0, -2, -1, 2, 1, 0},
  {0, 2, -1, -2, 1, 0},
  {0, 4, 0, -5, 0, 1}
};

void check_max(data_t* src, data_t* cnt, int size) {
  for (int i = 0; i < size; ++i) {
    *cnt = *cnt > *(src + i) ? *cnt : *(src + i);
  }
}


void calculate(data_t Y[bfsize_cho][tilesize_or][tilesize_oc],
  data_t Z[bfsize_chi][tilesize_ir][tilesize_ic],
  data_t V[bfsize_cho][bfsize_chi][K][K])
{

#pragma HLS ARRAY_PARTITION variable=V complete dim=4
#pragma HLS ARRAY_PARTITION variable=V complete dim=3

#pragma HLS ARRAY_PARTITION variable=Y complete dim=2
#pragma HLS ARRAY_PARTITION variable=Y complete dim=3

#pragma HLS ARRAY_PARTITION variable=Z complete dim=2
#pragma HLS ARRAY_PARTITION variable=Z complete dim=3

  int i, j, cho, chi;
  inter_t T[6][6], At_T[6][6];
  inter_t Bt_Z[6][6];
  inter_t U[6][6];
  inter_t temp_Y[4][4];
#pragma HLS ARRAY_PARTITION variable=T complete dim=0
#pragma HLS ARRAY_PARTITION variable=At_T complete dim=0
#pragma HLS ARRAY_PARTITION variable=Bt_Z complete dim=2
#pragma HLS ARRAY_PARTITION variable=U complete dim=0
#pragma HLS ARRAY_PARTITION variable=temp_Y complete dim=0


  /* ============= Y = At x T(U * V) x A ============= */
  // we merge all the process of calculating Y into this function to 
  // save the resouces, leaving the optimization to HLS

  for (chi = 0; chi < bfsize_chi; ++chi) {
    for (cho = 0; cho < prfactor_cho; ++cho) {
      #pragma HLS PIPELINE

      // read from output
      for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) {
          temp_Y[i][j] = Y[cho][i][j];
        }
      }

      // cal U
      for (j = 0; j < 6; ++j) {
        Bt_Z[0][j] = 4*Z[chi][0][j] - 5*Z[chi][2][j] + Z[chi][4][j];
        Bt_Z[1][j] = -4*Z[chi][1][j] - 4*Z[chi][2][j] + Z[chi][3][j] + Z[chi][4][j];
        Bt_Z[2][j] = 4*Z[chi][1][j] - 4*Z[chi][2][j] - Z[chi][3][j] + Z[chi][4][j];
        Bt_Z[3][j] = -2*Z[chi][1][j] - Z[chi][2][j] + 2*Z[chi][3][j] + Z[chi][4][j];
        Bt_Z[4][j] = 2*Z[chi][1][j] - Z[chi][2][j] -2*Z[chi][3][j] + Z[chi][4][j];
        Bt_Z[5][j] = 4*Z[chi][1][j] - 5*Z[chi][3][j] + Z[chi][5][j];
      }

      for (i = 0; i < 6; ++i) {
        U[i][0] = 4*Bt_Z[i][0] - 5*Bt_Z[i][2] + Bt_Z[i][4];
        U[i][1] = -4*Bt_Z[i][1] - 4*Bt_Z[i][2] + Bt_Z[i][3] + Bt_Z[i][4];
        U[i][2] = 4*Bt_Z[i][1] - 4*Bt_Z[i][2] - Bt_Z[i][3] + Bt_Z[i][4];
        U[i][3] = -2*Bt_Z[i][1] - Bt_Z[i][2] + 2*Bt_Z[i][3] + Bt_Z[i][4];
        U[i][4] = 2*Bt_Z[i][1] - Bt_Z[i][2] - 2*Bt_Z[i][3] + Bt_Z[i][4];
        U[i][5] = 4*Bt_Z[i][1] - 5*Bt_Z[i][3] + Bt_Z[i][5];
      }
    
      // cal t
      for (i = 0; i < 6; ++i) {
        for (j = 0; j < 6; ++j) {
          inter_t temp = V[cho][chi][i][j];
          T[i][j] = U[i][j] * temp;
        }
      }

      // cal Y
      for (j = 0; j < 6; ++j) {
        At_T[0][j] = T[0][j] + T[1][j] + T[2][j] + T[3][j] + T[4][j];
        At_T[1][j] = T[1][j] - T[2][j] + 2*T[3][j] - 2*T[4][j];
        At_T[2][j] = T[1][j] + T[2][j] + 4*T[3][j] + 4*T[4][j];
        At_T[3][j] = T[1][j] - T[2][j] + 8*T[3][j] - 8*T[4][j] + T[5][j];              
      }
      for (i = 0; i < 4; ++i) {
        temp_Y[i][0] += At_T[i][0] + At_T[i][1] + At_T[i][2] + At_T[i][3] + At_T[i][4];
        temp_Y[i][1] += At_T[i][1] - At_T[i][2] + 2*At_T[i][3] - 2*At_T[i][4];
        temp_Y[i][2] += At_T[i][1] + At_T[i][2] + 4*At_T[i][3] + 4*At_T[i][4];
        temp_Y[i][3] += At_T[i][1] - At_T[i][2] + 8*At_T[i][3] - 8*At_T[i][4] + At_T[i][5];
      }

      // write back
      for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) {
          Y[cho][i][j] = temp_Y[i][j];
        }
      }

    }
  }
}
