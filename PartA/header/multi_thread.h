#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

struct arg_struct {
    int N;
    int *matA;
    int *matB;
    int *output;
    int i;
    int *colsadded;
    int j;
};

void* rowaddition(void* arguments)
{
    struct arg_struct *args = (struct arg_struct *)arguments;
    
   for (int ii = 0; ii < args->N; ii+=8){
    
        __m256i firstrow = _mm256_loadu_si256((__m256i*)&args->matA[(args->i) * (args->N) + ii]);
        __m256i secondrow = _mm256_loadu_si256((__m256i*)&args->matA[((args->i) + 1) * (args->N) + ii]);

       __m256i addedrows = _mm256_add_epi32(firstrow,secondrow);
        
        _mm256_storeu_si256((__m256i*) &args->output[(args->i)/2 * (args->N) + ii], addedrows);

    	}
   }

void* coladdition(void* arguments)
{
    struct arg_struct *args = (struct arg_struct *)arguments;

    for (int j = 0; j < args->N; j+=8)
    {
        __m256i firstcol = _mm256_setr_epi32(args->matB[j*args->N + args->i], args->matB[(j+1)*args->N + args->i], args->matB[(j+2)*args->N + args->i], args->matB[(j+3)*args->N + args->i], args->matB[(j+4)*args->N + args->i], args->matB[(j+5)*args->N + args->i], args->matB[(j+6)*args->N + args->i], args->matB[(j+7)*args->N + args->i]);
          __m256i secondcol = _mm256_setr_epi32(args->matB[j*args->N + args->i+1], args->matB[(j+1)*args->N + args->i+1], args->matB[(j+2)*args->N + args->i+1], args->matB[(j+3)*args->N + args->i+1], args->matB[(j+4)*args->N + args->i+1], args->matB[(j+5)*args->N + args->i+1], args->matB[(j+6)*args->N + args->i+1], args->matB[(j+7)*args->N + args->i+1]);
          
          __m256i addedcols = _mm256_add_epi32(firstcol,secondcol);
          _mm256_storeu_si256((__m256i*) &args->output[(args->i/2)*args->N + j], addedcols);
          
    }
    
    
}

void* multiplication(void* arguments)
{
    struct arg_struct *args = (struct arg_struct *)arguments;
    for (int k = 0; k < args->N/2; k=k+1)
    {
        for(int jj=0; jj<args->N; jj+=8){
      __m256i firstmultiplier = _mm256_loadu_si256((__m256i*)&args->matA[args->i*args->N+jj]);
      __m256i secondmultiplier = _mm256_loadu_si256((__m256i*)&args->colsadded[k*args->N+jj]);
      
      __m256i results = _mm256_mullo_epi32(firstmultiplier, secondmultiplier);
      
        int finalres = _mm256_extract_epi32(results, 0) + _mm256_extract_epi32(results, 1) + _mm256_extract_epi32(results, 2) + _mm256_extract_epi32(results, 3) + _mm256_extract_epi32(results, 4) + _mm256_extract_epi32(results, 5) + _mm256_extract_epi32(results, 6) + _mm256_extract_epi32(results, 7);
     args->output[args->i*args->N/2 + k]  += finalres;  
      }
    }
    
}

void multiThread(int N, int *matA, int *matB, int *output)
{
	int *coladdedarr = new int[N*(N>>1)];
	int *rowaddedarr = new int[N*(N>>1)];
    pthread_t rowadderthreads[N/2];
    struct arg_struct args_ptrs[N/2];
    for (int iii = 0; iii < N; iii+=2) {
        struct arg_struct args;
        args.matA = matA;
        args.matB = matB;
        args.N = N;
        args.output = rowaddedarr;
        args.i = iii;
        args_ptrs[iii/2] = args;
    }
     for (int iii = 0; iii < N; iii+=2) {
    
   	 pthread_create(&rowadderthreads[iii/2], NULL, rowaddition, (void*) &args_ptrs[iii/2]);
    
    }
    
    for (int i = 0; i < N/2; i++)
        pthread_join(rowadderthreads[i], NULL);
    
    pthread_t coladderthreads[N>>1];

    for (int ii = 0; ii < N; ii+=2)
    {
        struct arg_struct args;
        args.matA = matA;
        args.matB = matB;
        args.N = N;
        args.output = coladdedarr;
        args.i = ii;
        args_ptrs[ii/2] = args;
    }
   
    
    for (int ii = 0; ii < N; ii+=2)
    {
    	pthread_create(&coladderthreads[ii/2], NULL, coladdition, (void*) &args_ptrs[ii/2]);
	}
    for (int i = 0; i < N/2; i++)
        pthread_join(coladderthreads[i], NULL);      

    pthread_t multiplier[(N>>1)];
    for (int ii = 0; ii < (N>>1); ii++)
    {
 	
        struct arg_struct args;
        args.matA = rowaddedarr;
        args.matB = matB;
        args.N = N;
        args.output = output;
        args.colsadded = coladdedarr;
        args.i = ii;
       	args_ptrs[ii] = args;
        
    }
    for (int ii = 0; ii < N>>1; ii++)
    {
    	
     pthread_create(&multiplier[ii], NULL, multiplication, (void*) &args_ptrs[ii]);
     
     }
    for (int i = 0; i < (N>>1); i++)
        pthread_join(multiplier[i], NULL);   
}
