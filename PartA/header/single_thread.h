
#include <smmintrin.h>
#include <immintrin.h>

void singleThread(int N, int *matA, int *matB, int *output)
{
  assert( N>=4 and N == ( N &~ (N-1)));
  int n = N>>1;
  
  int *addedcolsarr = new int[(N*N)>>1];
  int *addedrowsarr = new int[(N*N)>>1];
  int *temparr1 = new int[8];
  int *temparr2 = new int[8];
  int count = 0;
  for(int i = 0; i<N; i+=2){
  	for(int j=0; j<N; j+=8){
  		__m256i firstrow = _mm256_loadu_si256((__m256i*)&matA[i*N + j]);
  		__m256i secondrow = _mm256_loadu_si256((__m256i*)&matA[(i+1)*N+j]);
  		
  		__m256i addedrows = _mm256_add_epi32(firstrow,secondrow);
  		
  		_mm256_storeu_si256((__m256i*) &addedrowsarr[count], addedrows);
  		
  		count+=8;
  	}
  }

  count = 0;
  for(int i = 0; i<N; i+=2){
  	for(int j=0; j<N; j+=8){
  		__m256i firstcol = _mm256_setr_epi32(matB[j*N + i], matB[(j+1)*N + i], matB[(j+2)*N + i], matB[(j+3)*N + i], matB[(j+4)*N + i], matB[(j+5)*N + i], matB[(j+6)*N + i], matB[(j+7)*N + i]);
  		__m256i secondcol = _mm256_setr_epi32(matB[j*N + i+1], matB[(j+1)*N + i+1], matB[(j+2)*N + i+1], matB[(j+3)*N + i+1], matB[(j+4)*N + i+1], matB[(j+5)*N + i+1], matB[(j+6)*N + i+1], matB[(j+7)*N + i+1]);
  		
  		__m256i addedcols = _mm256_add_epi32(firstcol,secondcol);
  		_mm256_storeu_si256((__m256i*) &addedcolsarr[count], addedcols);
  		count+=8;
  	}
  }
  int c = 0;
  
  for(int i=0; i<N>>1; i+=1){
  for(int k=0; k<N>>1; k+=1){
  
  	for(int j=0; j<N; j+=8){
  	__m256i firstmultiplier = _mm256_loadu_si256((__m256i*)&addedrowsarr[i*N+j]);
  	__m256i secondmultiplier = _mm256_loadu_si256((__m256i*)&addedcolsarr[k*N+j]);
  	
  	__m256i results = _mm256_mullo_epi32(firstmultiplier, secondmultiplier);
  	
  	  int finalres = _mm256_extract_epi32(results, 0) + _mm256_extract_epi32(results, 1) + _mm256_extract_epi32(results, 2) + _mm256_extract_epi32(results, 3) + _mm256_extract_epi32(results, 4) + _mm256_extract_epi32(results, 5) + _mm256_extract_epi32(results, 6) + _mm256_extract_epi32(results, 7);
  	output[c] += finalres;
  	}
  	c++;
  	}
  }
 }
