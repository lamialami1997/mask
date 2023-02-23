//
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

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
seq_t *load_seq(const char *fname)
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
  seq_t *s = malloc(sizeof(seq_t));
  
  if (!s)
    {
      err_id = ERR_MALLOC_NULL;
      return NULL;
    }
  
  //Length of sequence is file size in bytes
  s->len = sb.st_size;

  //Allocating memory for sequence bases
  s->bases = malloc(sizeof(u8) * sb.st_size);
  
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

//
void mask(const u8 *a, const u8 *b, u8 *c, u64 n)
{  
  //
  for (u64 i = 0; i < n; i++)
    c[i] = a[i] ^ b[i];
}

//
void measure_mask(const char *title,
		  void kernel(const u8 *, const u8 *, u8 *, u64),
		  u8 *s1,
		  u8 *s2,
		  u64 n)
{
  u64 r = 3;
  f64 elapsed = 0.0;
  struct timespec t1, t2;

  u8 *cmp_mask = malloc(sizeof(u8) * n);

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
  
  free(cmp_mask);
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
  
  //
  release_seq(s1); free(s1);
  release_seq(s2); free(s2);
  
  //
  return 0;

}
