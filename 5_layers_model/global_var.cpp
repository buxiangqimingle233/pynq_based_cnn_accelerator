#include "global_var.h"

weight_t b_0[CHout_0];
weight_t b_1[CHout_1];
weight_t b_2[CHout_2];
weight_t b_3[CHout_3];
weight_t b_4[CHout_4];
weight_t b_5[CHout_5];
weight_t b_fc[10];

inter_t if_0[3][34][34];
inter_t of_0[16][Out_0][Out_0];

inter_t if_1[16][18][18];
inter_t of_1[32][Out_1][Out_1];

inter_t if_2[32][18][18];
inter_t of_2[32][Out_2][Out_2];

inter_t if_3[32][10][10];
inter_t of_3[32][Out_3][Out_3];

inter_t if_4[32][10][10];
inter_t of_4[32][Out_4][Out_4];

inter_t if_5[32][6][6];
inter_t of_5[32][Out_5][Out_5];

inter_t if_fc[512];
inter_t of_fc[10];

int result[500];

weight_t w_conv_0[16][3][3][3];
weight_t w_conv_1[32][16][3][3];
weight_t w_conv_2[32][32][3][3];
weight_t w_conv_3[32][32][3][3];
weight_t w_conv_4[32][32][3][3];
weight_t w_conv_5[32][32][3][3];
weight_t w_fc[10][512];