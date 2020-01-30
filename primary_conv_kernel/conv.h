
#define __SAMPLE_1__
#define __RELEASE__
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <memory.h>
#include "ap_int.h"


#ifdef __DEBUG__
  #define CHIN 2
  #define CHOUT 2
  #define R 5
  #define C 5
  #define KER 2
  #define STR 2
  #define OF_PATH ""
#elif defined __SAMPLE_1__
  #define CHIN 3
  #define CHOUT 16
  #define R 32
  #define C 32
  #define KER 3
  #define STR 1
  #define ROUT 30
  #define COUT 30
#elif defined __SAMPLE_2__
  #define CHIN 64
  #define CHOUT 64
  #define R 128
  #define C 128 
  #define KER 5
  #define STR 2

  #define ROUT 62
  #define COUT 62
#else
  #define CHIN 64
  #define CHOUT 64
  #define R 128
  #define C 128
  #define KER 5
  #define STR 2
#endif

#ifdef __TEST__
	const int CHin = CHIN;
  const int CHout = CHOUT;
  const int R_in = R;
  const int C_in = C;
  const int K = KER;
  const int S = STR;
  const int R_out = ((R_in-KER+STR)/STR);
  const int C_out = ((R_in-KER+STR)/STR);
#endif


#define R_OUT ((R-KER+STR) / STR)
#define C_OUT ((C-KER+STR) / STR)

typedef int ch_type;
typedef float d_type;
typedef const unsigned int s_type;
const int NParameter = 6;

void cnn(d_type* In, d_type* Out, d_type* W, int *Parameter);
void cnn_org(d_type* In, d_type* Out, d_type* W, int *Parameter);

