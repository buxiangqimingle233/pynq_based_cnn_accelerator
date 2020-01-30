#ifndef __GLOBAL_VAR__
#define __GLOBAL_VAR__

#include "base.h"

extern weight_t w_conv_0[16][3][3][3];
extern weight_t w_conv_1[32][16][3][3];
extern weight_t w_conv_2[32][32][3][3];
extern weight_t w_conv_3[32][32][3][3];
extern weight_t w_conv_4[32][32][3][3];
extern weight_t w_conv_5[32][32][3][3];
extern weight_t w_fc[10][512];

// intermediate buffer for core.cpp
extern inter_t if_0[3][34][34];
extern inter_t of_0[16][Out_0][Out_0];

extern inter_t if_1[16][18][18];
extern inter_t of_1[32][Out_1][Out_1];
// extern inter_t w_1[32][16][3][3];

extern inter_t if_2[32][18][18];
extern inter_t of_2[32][Out_2][Out_2];
// extern inter_t w_2[32][32][3][3];

extern inter_t if_3[32][10][10];
extern inter_t of_3[32][Out_3][Out_3];
// extern inter_t w_3[32][32][3][3];

extern inter_t if_4[32][10][10];
extern inter_t of_4[32][Out_4][Out_4];
// extern inter_t w_4[32][32][3][3];

extern inter_t if_5[32][6][6];
extern inter_t of_5[32][Out_5][Out_5];
// extern inter_t w_5[32][32][3][3];

extern inter_t if_fc[512];
extern inter_t of_fc[10];


extern weight_t b_fc[10];
extern weight_t b_0[CHout_0], b_1[CHout_1], b_2[CHout_2], b_3[CHout_3], b_4[CHout_4], b_5[CHout_5];

extern int result[500];

#endif /* __GLOBAL_VAR__ */