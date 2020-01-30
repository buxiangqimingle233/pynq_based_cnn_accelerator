#ifndef __IO__
#define __IO__
#include "base.h"
#include "global_var.h"

void load_w(input_w_t* W);
void load_b(input_w_t* B);
void load_img(input_t* In, channel_t& ch);
void store_of(output_t* Out);



#endif /* __IO__ */