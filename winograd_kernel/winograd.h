#ifndef __WINOGRAD__
#define __WINOGRAD__

/* =============== path =================== */
#define PREFIX "F:\\PKU\\HLS_prj\\Wino_prj\\src\\dat\\"
#define IF_PATH PREFIX "new_sample_input.dat"
#define OF_PATH PREFIX "sample_out.dat"
#define W_PATH PREFIX "sample_weight.dat"
#define LOG_PATH "F:\\PKU\\HLS_prj\\Wino_prj\\src\\log.txt"

/* =============== size =================== */

// FIXME: 
#define CHIN 32
#define CHOUT 32
#define RIN 128
#define CIN 128
#define ROUT 126
#define COUT 126  
#define K 6
#define STR 1

/* ============== typedef =============== */
// #define HLS

#ifdef HLS
  #include <ap_int.h>
  #include <ap_fixed.h> 
  typedef ap_fixed<16, 3, AP_RND_INF, AP_SAT> data_t;
  typedef ap_fixed<24, 8, AP_RND_INF, AP_SAT> inter_t;
  typedef struct {
    float a[2];
//  #pragma HLS DATA_PACK
  } input_t;
  // typedef float inter_t;
  typedef struct {
    float a[2];
  } output_t;
#else
  typedef struct {
    float a[2];
//  #pragma HLS DATA_PACK
  } input_t;
  // typedef float inter_t;
  typedef struct {
    float a[2];
  } output_t;
  typedef float data_t;
  typedef float inter_t;
#endif
// typedef float input_t;
// typedef float output_t;s


const int tilesize_oc = 4;
const int tilesize_or = 4;
const int tilesize_ic = 6;
const int tilesize_ir = 6;

/* ============== design space =============== */
// const int tilenum_ir = 22;
// const int tilenum_ic = 1
const int tilenum_oc = 32;

const int bfsize_chi = CHIN;
const int bfsize_cho = CHOUT;

const int prfactor_chi = bfsize_chi;
const int prfactor_cho = CHOUT;
const int prfactor_or = 1;
const int prfactor_oc = 1;


/* ============== functions ============= */
void cnn(input_t *In, output_t *Out, input_t *W, int *Parameter);

void calculate(data_t of_buffer[bfsize_cho][tilesize_or][tilesize_oc],
  data_t if_buffer[bfsize_chi][tilesize_ir][tilesize_ic],
  data_t W_buffer[bfsize_cho][bfsize_chi][K][K]
);

void load_if_shift(data_t if_buffer[bfsize_chi][tilesize_ir][tilesize_ic], input_t* In, 
  int chi, int r, int c, int ch_in, int r_in, int c_in);

void load_if_next_chi(data_t if_buffer[bfsize_chi][tilesize_ir][tilesize_ic], input_t* In, 
  int chi, int r, int c, int ch_in, int r_in, int c_in);

void load_w(data_t W_buffer[bfsize_cho][bfsize_chi][K][K], input_t* W, 
  int chi, int ch_in, int ch_out);

void set_of_zero(data_t of_buffer[tilenum_oc][bfsize_cho][tilesize_or][tilesize_oc]);

void store_of(output_t *Out, data_t of_buffer[tilenum_oc][bfsize_cho][tilesize_or][tilesize_oc],
  int r, int r_out, int c_out, int ch_out);

#endif /* __WINOGRAD__ */
