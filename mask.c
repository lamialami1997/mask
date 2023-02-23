//
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <immintrin.h>
#include <omp.h>

#include "types.h"

//Defining error codes
#define ERR_FNAME_NULL   0
#define ERR_MALLOC_NULL  1
#define ERR_STAT         3
#define ERR_OPEN_FILE    4
#define ERR_READ_BYTES   5
#define ERR_NULL_POINTER 6
#define ERR_CREAT_FILE   7

//
#define ALIGN 64

typedef double f64;

//Sequence definitions
typedef struct {

  //Sequence elements/bytes
  u8 *bases;

  //Sequence length
  u64 len;

} seq_t;

//Global error variable
u64 err_id = 0;

//Error messages
const char *err_msg[] = {

  "file name pointer NULL",
  "memory allocation fail, 'malloc' returned NULL",
  "cannot 'stat' file",
  "cannot open file, 'fopen' returned NULL",
  "mismatch between read bytes and file length",
  "cannot create file",
  
  NULL
};

//
void error()
{
  //
  printf("Error (%llu): %s\n", err_id, err_msg[err_id]);
  
  //
  exit(-1);
}

//
seq_t *restrict load_seq(const char *fname)
{
  //
  if (!fname)
    {
      err_id = ERR_FNAME_NULL;
      return NULL;
    }

  //
  struct stat sb;

  if (stat(fname, &sb) < 0)
    {
      err_id = ERR_STAT;
      return NULL;
    }
  
  //Allocate sequence 
  seq_t *restrict s = aligned_alloc(ALIGN , sizeof(seq_t));
  
  if (!s)
    {
      err_id = ERR_MALLOC_NULL;
      return NULL;
    }
  
  //Length of sequence is file size in bytes
  s->len = sb.st_size;

  //Allocating memory for sequence bases
  s->bases = aligned_alloc(ALIGN , sizeof(u8) * sb.st_size);
  
  if (!s->bases)
    {
      err_id = ERR_MALLOC_NULL;
      return NULL;
    }

  //Opening the file
  FILE *fp = fopen(fname, "rb");

  if (!fp)
    {
      err_id = ERR_OPEN_FILE;
      return NULL;
    }

  //Reading bytes from file
  size_t read_bytes = fread(s->bases, sizeof(u8), s->len, fp);

  //Closing file
  fclose(fp);

  //Check if bytes were fully read
  if (read_bytes != s->len)
    {
      err_id = ERR_READ_BYTES;
      return NULL;
    }
  
  //
  return s;
}

//
void release_seq(seq_t *s)
{
  //
  if (s)
    {
      //
      if (s->bases)
  free(s->bases);
      else
  err_id = ERR_NULL_POINTER;
    
      //
      s->len = 0;
    }
  else
    err_id = ERR_NULL_POINTER;
}

//version de base "Naive"

void mask(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  //
  for (u64 i = 0; i < n; i++)
    c[i] = a[i] ^ b[i];
}
// version para "Naive_para"
void mask_para(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  #pragma omp parallel for schedule(static)
  for (u64 i = 0; i < n; i++)
    c[i] = a[i] ^ b[i];
}

//version Unroll 
#if __AVX512F__
void mask_unroll(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  //
  for (u64 i = 0; i < n -(n%64); i+=64) 
  {
    c[i+0] = a[i+0]  ^ b[i+0] ;
    c[i+1] = a[i+1]  ^ b[i+1] ;
    c[i+2] = a[i+2]  ^ b[i+2] ;
    c[i+3] = a[i+3]  ^ b[i+3] ;
    c[i+4] = a[i+4]  ^ b[i+4] ;
    c[i+5] = a[i+5]  ^ b[i+5] ;
    c[i+6] = a[i+6]  ^ b[i+6] ;
    c[i+7] = a[i+7]  ^ b[i+7] ;
    c[i+8] = a[i+8]  ^ b[i+8] ;
    c[i+9] = a[i+0] ^ b[i+0];
    c[i+10] = a[i+10] ^ b[i+10];
    c[i+11] = a[i+11] ^ b[i+11];
    c[i+12] = a[i+12] ^ b[i+12];
    c[i+13] = a[i+13] ^ b[i+13];
    c[i+14] = a[i+14] ^ b[i+14];
    c[i+15] = a[i+15] ^ b[i+15];
    c[i+16] = a[i+16] ^ b[i+16];
    c[i+17] = a[i+17] ^ b[i+17];
    c[i+18] = a[i+18] ^ b[i+18];
    c[i+19] = a[i+19] ^ b[i+19];
    c[i+20] = a[i+20] ^ b[i+20];
    c[i+21] = a[i+21] ^ b[i+21];
    c[i+22] = a[i+22] ^ b[i+22];
    c[i+23] = a[i+23] ^ b[i+23];
    c[i+24] = a[i+24] ^ b[i+24];
    c[i+25] = a[i+25] ^ b[i+25];
    c[i+26] = a[i+26] ^ b[i+26];
    c[i+27] = a[i+27] ^ b[i+27];
    c[i+28] = a[i+28] ^ b[i+28];
    c[i+29] = a[i+29] ^ b[i+29];
    c[i+30] = a[i+30] ^ b[i+30];
    c[i+31] = a[i+31] ^ b[i+31];
    c[i+32] = a[i+32] ^ b[i+32];
    c[i+33] = a[i+33] ^ b[i+33];
    c[i+34] = a[i+34] ^ b[i+34];
    c[i+35] = a[i+35] ^ b[i+35];
    c[i+36] = a[i+36] ^ b[i+36];
    c[i+37] = a[i+37] ^ b[i+37];
    c[i+38] = a[i+38] ^ b[i+38];
    c[i+39] = a[i+39] ^ b[i+39];
    c[i+41] = a[i+41] ^ b[i+41];
    c[i+42] = a[i+42] ^ b[i+42];
    c[i+43] = a[i+43] ^ b[i+43];
    c[i+44] = a[i+44] ^ b[i+44];
    c[i+45] = a[i+45] ^ b[i+45];
    c[i+46] = a[i+46] ^ b[i+46];
    c[i+47] = a[i+47] ^ b[i+47];
    c[i+48] = a[i+48] ^ b[i+48];
    c[i+49] = a[i+49] ^ b[i+49];
    c[i+50] = a[i+50] ^ b[i+50];
    c[i+51] = a[i+51] ^ b[i+51];
    c[i+52] = a[i+52] ^ b[i+52];
    c[i+53] = a[i+53] ^ b[i+53];
    c[i+54] = a[i+54] ^ b[i+54];
    c[i+55] = a[i+55] ^ b[i+55];
    c[i+56] = a[i+56] ^ b[i+56];
    c[i+57] = a[i+57] ^ b[i+57];
    c[i+58] = a[i+58] ^ b[i+58];
    c[i+59] = a[i+59] ^ b[i+59];
    c[i+60] = a[i+60] ^ b[i+60];
    c[i+61] = a[i+61] ^ b[i+61];
    c[i+62] = a[i+62] ^ b[i+62];
    c[i+63] = a[i+63] ^ b[i+63];
  }
  for (u64 i = n - (n%64); i < n; i++)
    c[i] = a[i] ^ b[i]; 
    
}

#else
void mask_unroll(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  //
  for (u64 i = 0; i < n - (n%32); i+=32)  
    {
      c[i+0] = a[i+0]  ^ b[i+0] ;
      c[i+1] = a[i+1]  ^ b[i+1] ;
      c[i+2] = a[i+2]  ^ b[i+2] ;
      c[i+3] = a[i+3]  ^ b[i+3] ;
      c[i+4] = a[i+4]  ^ b[i+4] ;
      c[i+5] = a[i+5]  ^ b[i+5] ;
      c[i+6] = a[i+6]  ^ b[i+6] ;
      c[i+7] = a[i+7]  ^ b[i+7] ;
      c[i+8] = a[i+8]  ^ b[i+8] ;
      c[i+9] = a[i+0] ^ b[i+0];
      c[i+10] = a[i+10] ^ b[i+10];
      c[i+11] = a[i+11] ^ b[i+11];
      c[i+12] = a[i+12] ^ b[i+12];
      c[i+13] = a[i+13] ^ b[i+13];
      c[i+14] = a[i+14] ^ b[i+14];
      c[i+15] = a[i+15] ^ b[i+15];
      c[i+16] = a[i+16] ^ b[i+16];
      c[i+17] = a[i+17] ^ b[i+17];
      c[i+18] = a[i+18] ^ b[i+18];
      c[i+19] = a[i+19] ^ b[i+19];
      c[i+20] = a[i+20] ^ b[i+20];
      c[i+21] = a[i+21] ^ b[i+21];
      c[i+22] = a[i+22] ^ b[i+22];
      c[i+23] = a[i+23] ^ b[i+23];
      c[i+24] = a[i+24] ^ b[i+24];
      c[i+25] = a[i+25] ^ b[i+25];
      c[i+26] = a[i+26] ^ b[i+26];
      c[i+27] = a[i+27] ^ b[i+27];
      c[i+28] = a[i+28] ^ b[i+28];
      c[i+29] = a[i+29] ^ b[i+29];
      c[i+30] = a[i+30] ^ b[i+30];
      c[i+31] = a[i+31] ^ b[i+31];
    }
    for (u64 i = n - (n%32); i < n; i++)
    c[i] = a[i] ^ b[i];
    
} 
#endif

//version Unroll parallel 
#if __AVX512F__
void mask_unroll_para(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  #pragma omp parallel for schedule(static)
  for (u64 i = 0; i < n -(n%64); i+=64) 
  {
    c[i+0] = a[i+0]  ^ b[i+0] ;
    c[i+1] = a[i+1]  ^ b[i+1] ;
    c[i+2] = a[i+2]  ^ b[i+2] ;
    c[i+3] = a[i+3]  ^ b[i+3] ;
    c[i+4] = a[i+4]  ^ b[i+4] ;
    c[i+5] = a[i+5]  ^ b[i+5] ;
    c[i+6] = a[i+6]  ^ b[i+6] ;
    c[i+7] = a[i+7]  ^ b[i+7] ;
    c[i+8] = a[i+8]  ^ b[i+8] ;
    c[i+9] = a[i+0] ^ b[i+0];
    c[i+10] = a[i+10] ^ b[i+10];
    c[i+11] = a[i+11] ^ b[i+11];
    c[i+12] = a[i+12] ^ b[i+12];
    c[i+13] = a[i+13] ^ b[i+13];
    c[i+14] = a[i+14] ^ b[i+14];
    c[i+15] = a[i+15] ^ b[i+15];
    c[i+16] = a[i+16] ^ b[i+16];
    c[i+17] = a[i+17] ^ b[i+17];
    c[i+18] = a[i+18] ^ b[i+18];
    c[i+19] = a[i+19] ^ b[i+19];
    c[i+20] = a[i+20] ^ b[i+20];
    c[i+21] = a[i+21] ^ b[i+21];
    c[i+22] = a[i+22] ^ b[i+22];
    c[i+23] = a[i+23] ^ b[i+23];
    c[i+24] = a[i+24] ^ b[i+24];
    c[i+25] = a[i+25] ^ b[i+25];
    c[i+26] = a[i+26] ^ b[i+26];
    c[i+27] = a[i+27] ^ b[i+27];
    c[i+28] = a[i+28] ^ b[i+28];
    c[i+29] = a[i+29] ^ b[i+29];
    c[i+30] = a[i+30] ^ b[i+30];
    c[i+31] = a[i+31] ^ b[i+31];
    c[i+32] = a[i+32] ^ b[i+32];
    c[i+33] = a[i+33] ^ b[i+33];
    c[i+34] = a[i+34] ^ b[i+34];
    c[i+35] = a[i+35] ^ b[i+35];
    c[i+36] = a[i+36] ^ b[i+36];
    c[i+37] = a[i+37] ^ b[i+37];
    c[i+38] = a[i+38] ^ b[i+38];
    c[i+39] = a[i+39] ^ b[i+39];
    c[i+41] = a[i+41] ^ b[i+41];
    c[i+42] = a[i+42] ^ b[i+42];
    c[i+43] = a[i+43] ^ b[i+43];
    c[i+44] = a[i+44] ^ b[i+44];
    c[i+45] = a[i+45] ^ b[i+45];
    c[i+46] = a[i+46] ^ b[i+46];
    c[i+47] = a[i+47] ^ b[i+47];
    c[i+48] = a[i+48] ^ b[i+48];
    c[i+49] = a[i+49] ^ b[i+49];
    c[i+50] = a[i+50] ^ b[i+50];
    c[i+51] = a[i+51] ^ b[i+51];
    c[i+52] = a[i+52] ^ b[i+52];
    c[i+53] = a[i+53] ^ b[i+53];
    c[i+54] = a[i+54] ^ b[i+54];
    c[i+55] = a[i+55] ^ b[i+55];
    c[i+56] = a[i+56] ^ b[i+56];
    c[i+57] = a[i+57] ^ b[i+57];
    c[i+58] = a[i+58] ^ b[i+58];
    c[i+59] = a[i+59] ^ b[i+59];
    c[i+60] = a[i+60] ^ b[i+60];
    c[i+61] = a[i+61] ^ b[i+61];
    c[i+62] = a[i+62] ^ b[i+62];
    c[i+63] = a[i+63] ^ b[i+63];
  }
  for (u64 i = n - (n%64); i < n; i++)
    c[i] = a[i] ^ b[i]; 
    
}

#else
void mask_unroll_para(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  #pragma omp parallel for schedule(static)
  for (u64 i = 0; i < n - (n%32); i+=32)  
    {
      c[i+0] = a[i+0]  ^ b[i+0] ;
      c[i+1] = a[i+1]  ^ b[i+1] ;
      c[i+2] = a[i+2]  ^ b[i+2] ;
      c[i+3] = a[i+3]  ^ b[i+3] ;
      c[i+4] = a[i+4]  ^ b[i+4] ;
      c[i+5] = a[i+5]  ^ b[i+5] ;
      c[i+6] = a[i+6]  ^ b[i+6] ;
      c[i+7] = a[i+7]  ^ b[i+7] ;
      c[i+8] = a[i+8]  ^ b[i+8] ;
      c[i+9] = a[i+0] ^ b[i+0];
      c[i+10] = a[i+10] ^ b[i+10];
      c[i+11] = a[i+11] ^ b[i+11];
      c[i+12] = a[i+12] ^ b[i+12];
      c[i+13] = a[i+13] ^ b[i+13];
      c[i+14] = a[i+14] ^ b[i+14];
      c[i+15] = a[i+15] ^ b[i+15];
      c[i+16] = a[i+16] ^ b[i+16];
      c[i+17] = a[i+17] ^ b[i+17];
      c[i+18] = a[i+18] ^ b[i+18];
      c[i+19] = a[i+19] ^ b[i+19];
      c[i+20] = a[i+20] ^ b[i+20];
      c[i+21] = a[i+21] ^ b[i+21];
      c[i+22] = a[i+22] ^ b[i+22];
      c[i+23] = a[i+23] ^ b[i+23];
      c[i+24] = a[i+24] ^ b[i+24];
      c[i+25] = a[i+25] ^ b[i+25];
      c[i+26] = a[i+26] ^ b[i+26];
      c[i+27] = a[i+27] ^ b[i+27];
      c[i+28] = a[i+28] ^ b[i+28];
      c[i+29] = a[i+29] ^ b[i+29];
      c[i+30] = a[i+30] ^ b[i+30];
      c[i+31] = a[i+31] ^ b[i+31];
    }
    for (u64 i = n - (n%32); i < n; i++)
    c[i] = a[i] ^ b[i];
    
} 
#endif


//version Intrasic "intrasic"
#if __AVX512F__
void mask_intrasic(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  // a = load(a)
  // b = load(b)
  // x = xor(a,b)
  // c = store(x)
  for (u64 i = 0; i < n; i+=64) 
  {

   __m512i va = _mm512_load_si512 (&a[i]);
   __m512i vb = _mm512_load_si512 (&b[i]);
   __m512i vx = _mm512_xor_si512 (va , vb);
    _mm512_store_si512 ( &c[i],vx);
  
  } 
    
}

#else
void mask_intrasic(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  //
  for (u64 i = 0; i < n; i+=32)  
    {
      __m256i va = _mm256_load_si256 (&a[i]);
      __m256i vb = _mm256_load_si256 (&b[i]);
      __m256i vx = _mm256_xor_si256 (va,vb);
      _mm256_store_si256 (&c[i],vx);
    }
    
} 
#endif

//version Intrasic parallel  "intrasic_parallel"
#if __AVX512F__
void mask_intrasic_para(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  // a = load(a)
  // b = load(b)
  // x = xor(a,b)
  // c = store(x)
  #pragma omp parallel for schedule(static)
  for (u64 i = 0; i < n; i+=64) 
  {

   __m512i va = _mm512_load_si512 (&a[i]);
   __m512i vb = _mm512_load_si512 (&b[i]);
   __m512i vx = _mm512_xor_si512 (va , vb);
    _mm512_store_si512 ( &c[i],vx);
  
  } 
    
}

#else
void mask_intrasic_para(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  #pragma omp parallel for schedule(static)
  for (u64 i = 0; i < n; i+=32)  
    {
      __m256i va = _mm256_load_si256 (&a[i]);
      __m256i vb = _mm256_load_si256 (&b[i]);
      __m256i vx = _mm256_xor_si256 (va,vb);
      _mm256_store_si256 (&c[i],vx);
    }
    
} 
#endif


//Version vectorisation avec les directive OpenMP "mask_simd"
#if __AVX512F__
void mask_simd(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  #pragma omp simd aligned(a,b,c)
  //#pragma unroll(32)
  for (u64 i = 0; i < n; i++)
    c[i] = a[i] ^ b[i];
}
#else
void mask_simd(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  #pragma omp simd aligned(a,b,c)
  //#pragma unroll(64)
  for (u64 i = 0; i < n; i++)
    c[i] = a[i] ^ b[i];
}
#endif

//Version vectorisation avec les directive OpenMP "mask_simd"
#if __AVX512F__
void mask_simd_para(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  #pragma omp parallel for simd aligned(a,b,c)  
    for (u64 i = 0; i < n; i++)
      c[i] = a[i] ^ b[i];
}
#else
void mask_simd_para(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  #pragma omp parallel for simd aligned(a,b,c) 
    for (u64 i = 0; i < n; i++)
      c[i] = a[i] ^ b[i];
}
#endif

//version assembleur "ASM"
/* void mask_asm(const u8 *restrict a, const u8 *restrict b, u8 *restrict c, u64 n)
{  
  
  //for (u64 i = 0; i < n; i++)
    //c[i] = a[i] ^ b[i];
  __asm__ volatile(
        "xor %%rcx, %%rcx;\n"
        
        "loop:;\n"
        
        "vmovdqu (%[_a], %%rcx), ymm0;\n"
        "vmovdqu (%[_b], %%rcx), ymm1;\n"
        
        "vpxor (%[_c], %%rcx),ymm0,ymm1;\n"

        "add $32, %%rcx;\n"
        "cmp %[_n], %%rcx;\n"
        "jl loop;\n"

        : //outputs
          
        : //inputs
          [_a] "r" (a),
          [_b] "r" (b),
          [_c] "r" (c),
          [_n] "r" (n)
          
        : //clobber
          "cc", "memory", "rcx",
          "ymm0", "ymm1", "ymm2"
        );
}*/

void measure_mask(const char *title,
      void kernel(const u8 *, const u8 *, u8 *, u64),
      u8 *s1,
      u8 *s2,
      u64 n)
{
  u64 r = 3;
  f64 elapsed = 0.0;
  struct timespec t1, t2;

  u8 *restrict cmp_mask = aligned_alloc(ALIGN ,sizeof(u8) * n);

  FILE *fp = fopen("mask.dat", "wb");

  if (!fp)
    {
      err_id = ERR_CREAT_FILE;
      error();
    }
  
  do
    {
      //Warmup
      kernel(s1, s2, cmp_mask, n);

      //
      clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
      
      for (u64 i = 0; i < r; i++) 
      kernel(s1, s2, cmp_mask, n);
      
      clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
      
      elapsed = (f64)(t2.tv_nsec - t1.tv_nsec) / (f64)r;
    }
  while (elapsed <= 0.0);
  
  /*
    GB/s can represent gigabytes per second or gigabases per second
    given that bases are encoded using 1 byte per base.

    Here, we compare two sequences and store into another. 
    Hence the length multiplication.
  */
  f64 bw = (n * 3) / elapsed;
  
  printf("%25s; 0x%08x; %15.3lf; %15.3lf\n", title, cmp_mask[0], elapsed, bw);

  fwrite(cmp_mask, sizeof(u8), n, fp);
  
  //free(cmp_mask);
  fclose(fp);
}

//
int main(int argc, char **argv)
{
  //
  if (argc < 3)
    return printf("Usage: %s [seq1] [seq2]\n", argv[0]), 1;
  
  //
  seq_t *s1 = load_seq(argv[1]);

  if (!s1)
    error();

  //
  seq_t *s2 = load_seq(argv[2]);

  if (!s2)
    error();
  
  //
  if (s1->len != s2->len)
    return printf("Error: sequences must match in length"), 2;

  //
  printf("%25s; %10s; %15s; %15s\n", "title", "mask", "elapsed(ns)", "GB/s");
  
  //
  measure_mask("Naive", mask, s1->bases, s2->bases, s1->len);
  measure_mask("Naive_para", mask_para, s1->bases, s2->bases, s1->len);
  measure_mask("Unroll", mask_unroll, s1->bases, s2->bases, s1->len);
  measure_mask("Unroll_para", mask_unroll_para, s1->bases, s2->bases, s1->len);
  measure_mask("Intasic", mask_intrasic, s1->bases, s2->bases, s1->len);
  measure_mask("Intasic_para", mask_intrasic_para, s1->bases, s2->bases, s1->len);
  measure_mask("OMP_Simd", mask_simd, s1->bases, s2->bases, s1->len);
  measure_mask("OMP_Simd_para", mask_simd_para, s1->bases, s2->bases, s1->len);
  //measure_mask("Assembly", mask_asm, s1->bases, s2->bases, s1->len);
  
  //
  release_seq(s1); free(s1);
  release_seq(s2); free(s2);
  
  //
  return 0;

}


// A -> 00
// C -> 01
// G -> 10