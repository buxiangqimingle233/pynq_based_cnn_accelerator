#ifndef __BASE__
#define __BASE__

#include <ap_int.h>
#include <hls_stream.h>
// #define DEBUG
// #define EXTERN

/* ========= SIZE ============ */
#define IMG_SIZE 32

#define CHin_0 3
#define CHout_0 16
#define In_0 32
#define Out_0 32

#define CHin_1 16
#define CHout_1 32
#define In_1 16
#define Out_1 16

#define CHin_2 32
#define CHout_2 32
#define In_2 16
#define Out_2 16

#define CHin_3 32
#define CHout_3 32
#define In_3 8
#define Out_3 8

#define CHin_4 32
#define CHout_4 32
#define In_4 8
#define Out_4 8

#define CHin_5 32
#define CHout_5 32
#define In_5 4
#define Out_5 4

#define FC_in 512
#define FC_out 10

/* ======== type defination ======== */

#ifdef DEBUG
  typedef float inter_t;
  typedef int i_fac_t;
  typedef int o_fac_t;
  typedef float weight_t;
#else
  typedef ap_fixed<24, 5> inter_t;
  typedef ap_fixed<16, 1> weight_t;
  typedef ap_int<8> i_fac_t;
  typedef ap_int<8> o_fac_t;
#endif

typedef hls::stream<inter_t> channel_t;

#define PRTCL_PR_FC 8
typedef struct Input {
  i_fac_t fac[8];
} input_t;
typedef struct Weight{
  weight_t fac[8];
} input_w_t;
typedef o_fac_t output_t;

typedef weight_t w_buf_t[32][32][3][3];

/* =========== para for hls =============*/

const int pr_0 = 8;
const int pr_1 = 2;
const int pr_2 = 2;
const int pr_3 = 1;
const int pr_4 = 1;
const int pr_5 = 1;

/* ========== for debug ========== */
#ifdef EXTERN
extern int batch;
#endif
#define TEST_BATCH 50


#endif /* __BASE__ */
