// originally from djm34 (https://github.com/djm34/ccminer-sp-neoscrypt/)

#include <stdio.h>
#include <memory.h>
#include "cuda_helper.h"
#include "cuda_vector.h" 

#define vectype uintx64bis
#define vectypeS uint28 

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif

#ifdef _MSC_VER
#define THREAD __declspec(thread)
#else
#define THREAD __thread
#endif

static THREAD cudaStream_t stream[2];

__device__  __align__(16) vectypeS *  W;
__device__  __align__(16) vectypeS * W2;
__device__  __align__(16) vectypeS* Tr;
__device__  __align__(16) vectypeS* Tr2;
__device__  __align__(16) vectypeS* Input;
__device__  __align__(16) vectypeS* B2;

static uint32_t *d_NNonce[MAX_GPUS];

__constant__  uint32_t pTarget[8];
__constant__  uint32_t key_init[16];
__constant__  uint32_t input_init[16];
__constant__  uint32_t c_data[64];

#define SALSA_SMALL_UNROLL 1
#define CHACHA_SMALL_UNROLL 1
#define BLAKE2S_BLOCK_SIZE    64U 
#define BLAKE2S_OUT_SIZE      32U
#define BLAKE2S_KEY_SIZE      32U
#define BLOCK_SIZE            64U
#define FASTKDF_BUFFER_SIZE  256U
#define PASSWORD_LEN          80U
/// constants ///

static const __constant__  uint8 BLAKE2S_IV_Vec =
{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};


static const  uint8 BLAKE2S_IV_Vechost =
{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const uint32_t BLAKE2S_SIGMA_host[10][16] =
{
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
	{7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
	{9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
	{2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
	{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
	{13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
	{6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
	{10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
};

__constant__ uint32_t BLAKE2S_SIGMA[10][16] =
{
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
	{7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
	{9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
	{2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
	{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
	{13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
	{6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
	{10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
};

#define SALSA(a,b,c,d) { \
    b^=rotate(a+d,  7);    \
    c^=rotate(b+a,  9);    \
    d^=rotate(c+b, 13);    \
    a^=rotate(d+c, 18);     \
}

#define SALSA_CORE(state) { \
\
SALSA(state.s0,state.s4,state.s8,state.sc); \
SALSA(state.s5,state.s9,state.sd,state.s1); \
SALSA(state.sa,state.se,state.s2,state.s6); \
SALSA(state.sf,state.s3,state.s7,state.sb); \
SALSA(state.s0,state.s1,state.s2,state.s3); \
SALSA(state.s5,state.s6,state.s7,state.s4); \
SALSA(state.sa,state.sb,state.s8,state.s9); \
SALSA(state.sf,state.sc,state.sd,state.se); \
		} 

static __forceinline__ __device__ void shift256R4(uint32_t * ret, const uint8 &vec4, uint32_t shift2)
{
	uint32_t shift = 32 - shift2;
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[0]) : "r"(0), "r"(vec4.s0), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[1]) : "r"(vec4.s0), "r"(vec4.s1), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[2]) : "r"(vec4.s1), "r"(vec4.s2), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[3]) : "r"(vec4.s2), "r"(vec4.s3), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[4]) : "r"(vec4.s3), "r"(vec4.s4), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[5]) : "r"(vec4.s4), "r"(vec4.s5), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[6]) : "r"(vec4.s5), "r"(vec4.s6), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[7]) : "r"(vec4.s6), "r"(vec4.s7), "r"(shift));
	asm("shr.b32         %0, %1, %2;"     : "=r"(ret[8]) : "r"(vec4.s7), "r"(shift));
}

/*static __device__ __inline__ void chacha_step(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d)
{
asm("{\n\t"
"add.u32 %0,%0,%1; \n\t"
"xor.b32 %3,%3,%0; \n\t"
"prmt.b32 %3, %3, 0, 0x1032; \n\t"
"add.u32 %2,%2,%3; \n\t"
"xor.b32 %1,%1,%2; \n\t"
"shf.l.wrap.b32 %1, %1, %1, 12; \n\t"
"add.u32 %0,%0,%1; \n\t"
"xor.b32 %3,%3,%0; \n\t"
"prmt.b32 %3, %3, 0, 0x2103; \n\t"
"add.u32 %2,%2,%3; \n\t"
"xor.b32 %1,%1,%2; \n\t"
"shf.l.wrap.b32 %1, %1, %1, 7; \n\t}"
: "+r"(a), "+r"(b), "+r"(c), "+r"(d));
}
*/
#if __CUDA_ARCH__ >=500  

#define CHACHA_STEP(a,b,c,d) { \
a += b; d = __byte_perm(d^a,0,0x1032); \
c += d; b = rotate(b^c, 12); \
a += b; d = __byte_perm(d^a,0,0x2103); \
c += d; b = rotate(b^c, 7); \
	}

//#define CHACHA_STEP(a,b,c,d) chacha_step(a,b,c,d)
#else 
#define CHACHA_STEP(a,b,c,d) { \
a += b; d = rotate(d^a,16); \
c += d; b = rotate(b^c, 12); \
a += b; d = rotate(d^a,8); \
c += d; b = rotate(b^c, 7); \
	}
#endif

#define CHACHA_CORE_PARALLEL(state)	 { \
 \
	CHACHA_STEP(state.lo.s0, state.lo.s4, state.hi.s0, state.hi.s4); \
	CHACHA_STEP(state.lo.s1, state.lo.s5, state.hi.s1, state.hi.s5); \
	CHACHA_STEP(state.lo.s2, state.lo.s6, state.hi.s2, state.hi.s6); \
	CHACHA_STEP(state.lo.s3, state.lo.s7, state.hi.s3, state.hi.s7); \
	CHACHA_STEP(state.lo.s0, state.lo.s5, state.hi.s2, state.hi.s7); \
	CHACHA_STEP(state.lo.s1, state.lo.s6, state.hi.s3, state.hi.s4); \
	CHACHA_STEP(state.lo.s2, state.lo.s7, state.hi.s0, state.hi.s5); \
	CHACHA_STEP(state.lo.s3, state.lo.s4, state.hi.s1, state.hi.s6); \
\
	}

#define CHACHA_CORE_PARALLEL2(i0,state)	 { \
 \
  CHACHA_STEP(state[2*i0].x.x, state[2*i0].z.x, state[2*i0+1].x.x, state[2*i0+1].z.x); \
  CHACHA_STEP(state[2*i0].x.y, state[2*i0].z.y, state[2*i0+1].x.y, state[2*i0+1].z.y); \
  CHACHA_STEP(state[2*i0].y.x, state[2*i0].w.x, state[2*i0+1].y.x, state[2*i0+1].w.x); \
	CHACHA_STEP(state[2*i0].y.y, state[2*i0].w.y, state[2*i0+1].y.y, state[2*i0+1].w.y); \
	CHACHA_STEP(state[2*i0].x.x, state[2*i0].z.y, state[2*i0+1].y.x, state[2*i0+1].w.y); \
  CHACHA_STEP(state[2*i0].x.y, state[2*i0].w.x, state[2*i0+1].y.y, state[2*i0+1].z.x); \
  CHACHA_STEP(state[2*i0].y.x, state[2*i0].w.y, state[2*i0+1].x.x, state[2*i0+1].z.y); \
	CHACHA_STEP(state[2*i0].y.y, state[2*i0].z.x, state[2*i0+1].x.y, state[2*i0+1].w.x); \
\
	}

#define BLAKE2S_BLOCK_SIZE    64U
#define BLAKE2S_OUT_SIZE      32U
#define BLAKE2S_KEY_SIZE      32U

#if __CUDA_ARCH__ >= 500
#define BLAKE_G(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA[idx0][idx1]; a += key[idx]; \
	a += b; d = __byte_perm(d^a,0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	idx = BLAKE2S_SIGMA[idx0][idx1+1]; a += key[idx]; \
	a += b; d = __byte_perm(d^a,0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
} 
#else 
#define BLAKE_G(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA[idx0][idx1]; a += key[idx]; \
	a += b; d = rotate(d ^ a,16); \
	c += d; b = rotateR(b ^ c, 12); \
	idx = BLAKE2S_SIGMA[idx0][idx1 + 1]; a += key[idx]; \
  a += b; d = rotateR(d ^ a,8); \
	c += d; b = rotateR(b ^ c, 7); \
} 
#endif

#if __CUDA_ARCH__ >= 500

#define BLAKE(a, b, c, d, key1,key2) { \
	a += b + key1; \
	d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += b + key2; \
	d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
} 

#define BLAKE_G_PRE(idx0, idx1, a, b, c, d, key) { \
	a += b + key[idx0]; \
	d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += b + key[idx1]; \
	d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
} 

#define BLAKE_G_PRE0(idx0, idx1, a, b, c, d, key) { \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
} 

#define BLAKE_G_PRE1(idx0, idx1, a, b, c, d, key) { \
	a += b + key[idx0]; \
	d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
} 

#define BLAKE_G_PRE2(idx0, idx1, a, b, c, d, key) { \
	a += b; d = __byte_perm(d^a,0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += b + key[idx1]; \
	d = __byte_perm(d^a,0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
} 

#else 
#define BLAKE(a, b, c, d, key1,key2) { \
  \
  a += key1; \
  a += b; d = rotate(d^a,16); \
	c += d; b = rotateR(b^c, 12); \
  a += key2; \
  a += b; d = rotateR(d^a,8); \
	c += d; b = rotateR(b^c, 7); \
	} 

#define BLAKE_G_PRE(idx0, idx1, a, b, c, d, key) { \
  a += key[idx0]; \
  a += b; d = rotate(d^a,16); \
	c += d; b = rotateR(b^c, 12); \
  a += key[idx1]; \
  a += b; d = rotateR(d^a,8); \
	c += d; b = rotateR(b^c, 7); \
	} 

#define BLAKE_G_PRE0(idx0, idx1, a, b, c, d, key) { \
     \
  a += b; d = rotate(d^a,16); \
	c += d; b = rotateR(b^c, 12); \
    \
  a += b; d = rotateR(d^a,8); \
	c += d; b = rotateR(b^c, 7); \
	} 

#define BLAKE_G_PRE1(idx0, idx1, a, b, c, d, key) { \
  a += key[idx0]; \
  a += b; d = rotate(d^a,16); \
	c += d; b = rotateR(b^c, 12); \
  a += b; d = rotateR(d^a,8); \
	c += d; b = rotateR(b^c, 7); \
	} 

#define BLAKE_G_PRE2(idx0, idx1, a, b, c, d, key) { \
     \
  a += b; d = rotate(d^a,16); \
	c += d; b = rotateR(b^c, 12); \
  a += key[idx1]; \
  a += b; d = rotateR(d^a,8); \
	c += d; b = rotateR(b^c, 7); \
	} 
#endif

#define BLAKE_Ghost(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA_host[idx0][idx1]; \
	a += b + key[idx]; \
	d = ROTR32(d ^ a, 16); \
	c += d; b = ROTR32(b ^ c, 12); \
	idx = BLAKE2S_SIGMA_host[idx0][idx1 + 1]; \
	a += b + key[idx]; \
	d = ROTR32(d ^ a, 8); \
	c += d; b = ROTR32(b ^ c, 7); \
} 
#if __CUDA_ARCH__ < 500
static __forceinline__ __device__ void Blake2S(uint32_t * __restrict__ out, const uint32_t* __restrict__  inout, const  uint32_t * __restrict__ TheKey)
{
	uint16 V;
	uint32_t idx;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vec;
	V.lo = BLAKE2S_IV_Vec;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;


	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE0(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE0(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE1(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE1(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);



	//		{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	BLAKE_G_PRE2(9, 0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(5, 7, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(10, 15, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(14, 1, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(11, 12, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 8, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(3, 13, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	BLAKE_G_PRE1(2, 12, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 10, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE1(0, 11, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(8, 3, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(4, 13, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(7, 5, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(15, 14, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(1, 9, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	BLAKE_G_PRE2(12, 5, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(1, 15, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 13, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(4, 10, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 7, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(6, 3, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(9, 2, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(8, 11, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	BLAKE_G_PRE0(13, 11, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(7, 14, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(12, 1, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(3, 9, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(5, 0, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(15, 4, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 6, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 10, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	BLAKE_G_PRE1(6, 15, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE0(14, 9, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(11, 3, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(0, 8, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(12, 2, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(13, 7, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(1, 4, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(10, 5, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	BLAKE_G_PRE2(10, 2, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 4, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(7, 6, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(1, 5, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(15, 11, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(9, 14, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(3, 12, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(13, 0, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	V.lo ^= V.hi ^ tmpblock;

	V.hi = BLAKE2S_IV_Vec;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);


	//		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	//		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);


	//		{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	for(int x = 4; x < 10; ++x)
	{
		BLAKE_G(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
		BLAKE_G(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
		BLAKE_G(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
		BLAKE_G(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
		BLAKE_G(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
		BLAKE_G(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
		BLAKE_G(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
		BLAKE_G(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	}

	V.lo ^= V.hi ^ tmpblock;

	((uint8*)out)[0] = V.lo;
}
#else
static __forceinline__ __device__ void Blake2S_v2(uint32_t * __restrict__ out, const uint32_t* __restrict__  inout, const  uint32_t * __restrict__ TheKey)
{
	uint16 V;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vec;
	V.lo = BLAKE2S_IV_Vec;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE0(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE0(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE1(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE1(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	BLAKE_G_PRE2(9, 0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(5, 7, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(10, 15, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(14, 1, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(11, 12, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 8, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(3, 13, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	BLAKE_G_PRE1(2, 12, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 10, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE1(0, 11, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(8, 3, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(4, 13, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(7, 5, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(15, 14, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(1, 9, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	BLAKE_G_PRE2(12, 5, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(1, 15, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 13, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(4, 10, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 7, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(6, 3, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(9, 2, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(8, 11, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	BLAKE_G_PRE0(13, 11, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(7, 14, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(12, 1, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(3, 9, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(5, 0, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(15, 4, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 6, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 10, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	BLAKE_G_PRE1(6, 15, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE0(14, 9, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(11, 3, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(0, 8, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(12, 2, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(13, 7, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(1, 4, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(10, 5, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	BLAKE_G_PRE2(10, 2, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 4, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(7, 6, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(1, 5, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(15, 11, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(9, 14, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(3, 12, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(13, 0, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	V.hi = BLAKE2S_IV_Vec;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);


	//		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	//		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);


	//		{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	//#pragma unroll

	//		13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
	//		6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,
	//		10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[9], inout[0]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[5], inout[7]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[2], inout[4]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[10], inout[15]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[14], inout[1]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[11], inout[12]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[6], inout[8]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[3], inout[13]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[2], inout[12]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[6], inout[10]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[0], inout[11]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[8], inout[3]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[4], inout[13]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[7], inout[5]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[15], inout[14]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[1], inout[9]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[12], inout[5]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[1], inout[15]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[14], inout[13]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[4], inout[10]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[0], inout[7]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[6], inout[3]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[9], inout[2]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[8], inout[11]);

	//		13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
	//		6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[13], inout[11]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[7], inout[14]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[12], inout[1]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[3], inout[9]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[5], inout[0]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[15], inout[4]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[8], inout[6]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[2], inout[10]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[6], inout[15]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[14], inout[9]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[11], inout[3]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[0], inout[8]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[12], inout[2]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[13], inout[7]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[1], inout[4]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[10], inout[5]);
	//		10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,
	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[10], inout[2]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[8], inout[4]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[7], inout[6]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[1], inout[5]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[15], inout[11]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[9], inout[14]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[3], inout[12]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[13], inout[0]);

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	((uint8*)out)[0] = V.lo;
}
#endif

static __forceinline__ __device__ uint16 salsa_small_scalar_rnd(const uint16 &X)
{
	uint16 state = X;

#pragma unroll 1
	for(int i = 0; i < 10; ++i)
	{
		SALSA_CORE(state);
	}

	return(X + state);
}

static __device__ __forceinline__ uint16 chacha_small_parallel_rnd(const uint16 &X)
{
	uint16 st = X;
#pragma nounroll 
	for(int i = 0; i < 10; ++i)
	{
		CHACHA_CORE_PARALLEL(st);
	}
	return(X + st);
}

static __device__ __forceinline__ void neoscrypt_chacha(uint16 *XV)
{
	uint16 temp;

	XV[0] = chacha_small_parallel_rnd(XV[0] ^ XV[3]);
	temp = chacha_small_parallel_rnd(XV[1] ^ XV[0]);
	XV[1] = chacha_small_parallel_rnd(XV[2] ^ temp);
	XV[3] = chacha_small_parallel_rnd(XV[3] ^ XV[1]);
	XV[2] = temp;
}

static __device__ __forceinline__ void neoscrypt_salsa(uint16 *XV)
{
	uint16 temp;

	XV[0] = salsa_small_scalar_rnd(XV[0] ^ XV[3]);
	temp = salsa_small_scalar_rnd(XV[1] ^ XV[0]);
	XV[1] = salsa_small_scalar_rnd(XV[2] ^ temp);
	XV[3] = salsa_small_scalar_rnd(XV[3] ^ XV[1]);
	XV[2] = temp;
}

static __forceinline__ __host__ void Blake2Shost(uint32_t * inout, const uint32_t * inkey)
{
	uint16 V;
	uint32_t idx;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vechost;
	V.lo = BLAKE2S_IV_Vechost;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;

	for(int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inkey);
		BLAKE_Ghost(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inkey);
	}

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	V.hi = BLAKE2S_IV_Vechost;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	for(int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
		BLAKE_Ghost(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	}

	V.lo ^= V.hi ^ tmpblock;

	((uint8*)inout)[0] = V.lo;
}

#if __CUDA_ARCH__ < 500
static __forceinline__ __device__ void fastkdf256_v1(uint32_t thread, const uint32_t nonce, const uint32_t * __restrict__  s_data) //, vectypeS * output)
{
	vectypeS __align__(16) output[8];
	uint8_t bufidx;
	uchar4 bufhelper;
	uint32_t B[64];
	uint32_t qbuf, rbuf, bitbuf;
	uint32_t input[BLAKE2S_BLOCK_SIZE / 4];
	uint32_t key[BLAKE2S_BLOCK_SIZE / 4] = {0};

	const uint32_t data18 = s_data[18];
	const uint32_t data20 = s_data[0];

	((uintx64*)(B))[0] = ((uintx64*)s_data)[0];
	((uint32_t*)B)[19] = nonce;
	((uint32_t*)B)[39] = nonce;
	((uint32_t*)B)[59] = nonce;
	__syncthreads();

	((uint816*)input)[0] = ((uint816*)input_init)[0];
	((uint48*)key)[0] = ((uint48*)key_init)[0];

#pragma unroll  1
	for(int i = 0; i < 31; ++i)
	{
		bufhelper = ((uchar4*)input)[0];
		for(int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x)
		{
			bufhelper += ((uchar4*)input)[x];
		}
		bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;

		qbuf = bufidx / 4;
		rbuf = bufidx & 3;
		bitbuf = rbuf << 3;

		uint32_t shifted[9];

		shift256R4(shifted, ((uint8*)input)[0], bitbuf);

		//#pragma unroll
		uint32_t temp[9];

		for(int k = 0; k < 9; ++k)
		{
			uint32_t indice = (k + qbuf) & 0x0000003f;
			temp[k] = B[indice] ^ shifted[k];
			B[indice] = temp[k];
		}
		__syncthreads();

		uint32_t a = s_data[qbuf & 0x0000003f], b;
		//#pragma unroll
		for(int k = 0; k<16; k += 2)
		{
			b = s_data[(qbuf + k + 1) & 0x0000003f];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[k]) : "r"(a), "r"(b), "r"(bitbuf));
			a = s_data[(qbuf + k + 2) & 0x0000003f];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[k + 1]) : "r"(b), "r"(a), "r"(bitbuf));
		}

		const uint32_t noncepos = 19 - qbuf % 20;
		if(noncepos <= 16 && qbuf<60)
		{
			if(noncepos != 0)
				asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos - 1]) : "r"(data18), "r"(nonce), "r"(bitbuf));
			if(noncepos != 16)
				asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos]) : "r"(nonce), "r"(data20), "r"(bitbuf));
		}

		for(int k = 0; k<8; k++)
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[k]) : "r"(temp[k]), "r"(temp[k + 1]), "r"(bitbuf));

		Blake2S(input, input, key); //yeah right...
	}
	bufhelper = ((uchar4*)input)[0];
	for(int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x)
	{
		bufhelper += ((uchar4*)input)[x];
	}
	bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;

	qbuf = bufidx / 4;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;

	for(int i = 0; i<64; i++)
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(((uint32_t*)output)[i]) : "r"(B[(qbuf + i) & 0x0000003f]), "r"(B[(qbuf + i + 1) & 0x0000003f4]), "r"(bitbuf));

	((ulonglong4*)output)[0] ^= ((ulonglong4*)input)[0];

	((uintx64*)output)[0] ^= ((uintx64*)s_data)[0];
	((uint32_t*)output)[19] ^= nonce;
	((uint32_t*)output)[39] ^= nonce;
	((uint32_t*)output)[59] ^= nonce;

	for(int i = 0; i<8; i++)
		(Input + 8 * thread)[i] = output[i];
}

static __forceinline__ __device__ void fastkdf32_v1(uint32_t thread, const  uint32_t  nonce, const uint32_t * __restrict__ salt, const uint32_t *__restrict__  s_data, uint32_t &output)
{
	uint8_t bufidx;
	uchar4 bufhelper;
	uint32_t temp[9];

#define Bshift 16*thread

	uint32_t* const B0 = (uint32_t*)&B2[Bshift];
	const uint32_t cdata7 = s_data[7];
	const uint32_t data18 = s_data[18];
	const uint32_t data20 = s_data[0];

	((uintx64*)B0)[0] = ((uintx64*)salt)[0];
	uint32_t input[BLAKE2S_BLOCK_SIZE / 4]; uint32_t key[BLAKE2S_BLOCK_SIZE / 4] = {0};
	((uint816*)input)[0] = ((uint816*)s_data)[0];
	((uint48*)key)[0] = ((uint48*)salt)[0];
	uint32_t qbuf, rbuf, bitbuf;
	__syncthreads();

#pragma nounroll  
	for(int i = 0; i < 31; i++)
	{
		Blake2S(input, input, key);

		bufidx = 0;
		bufhelper = ((uchar4*)input)[0];
		for(int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x)
		{
			bufhelper += ((uchar4*)input)[x];
		}
		bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;
		qbuf = bufidx / 4;
		rbuf = bufidx & 3;
		bitbuf = rbuf << 3;
		uint32_t shifted[9];

		shift256R4(shifted, ((uint8*)input)[0], bitbuf);

		for(int k = 0; k < 9; k++)
		{
			temp[k] = B0[(k + qbuf) & 0x0000003f];
		}

		((uint28*)temp)[0] ^= ((uint28*)shifted)[0];
		temp[8] ^= shifted[8];

		uint32_t a = s_data[qbuf & 0x0000003f], b;
		//#pragma unroll
		for(int k = 0; k<16; k += 2)
		{
			b = s_data[(qbuf + k + 1) & 0x0000003f];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[k]) : "r"(a), "r"(b), "r"(bitbuf));
			a = s_data[(qbuf + k + 2) & 0x0000003f];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[k + 1]) : "r"(b), "r"(a), "r"(bitbuf));
		}

		const uint32_t noncepos = 19 - qbuf % 20;
		if(noncepos <= 16 && qbuf<60)
		{
			if(noncepos != 0)	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos - 1]) : "r"(data18), "r"(nonce), "r"(bitbuf));
			if(noncepos != 16)	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos]) : "r"(nonce), "r"(data20), "r"(bitbuf));
		}
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[0]) : "r"(temp[0]), "r"(temp[1]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[1]) : "r"(temp[1]), "r"(temp[2]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[2]) : "r"(temp[2]), "r"(temp[3]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[3]) : "r"(temp[3]), "r"(temp[4]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[4]) : "r"(temp[4]), "r"(temp[5]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[5]) : "r"(temp[5]), "r"(temp[6]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[6]) : "r"(temp[6]), "r"(temp[7]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[7]) : "r"(temp[7]), "r"(temp[8]), "r"(bitbuf));

		for(int k = 0; k < 9; k++)
		{
			B0[(k + qbuf) & 0x0000003f] = temp[k];
		}
		__syncthreads();
	}

	Blake2S(input, input, key);

	bufidx = 0;
	bufhelper = ((uchar4*)input)[0];
	for(int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x)
	{
		bufhelper += ((uchar4*)input)[x];
	}
	bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;
	qbuf = bufidx / 4;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;

	for(int k = 7; k < 9; k++)
	{
		temp[k] = B0[(k + qbuf) & 0x0000003f];
	}
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(output) : "r"(temp[7]), "r"(temp[8]), "r"(bitbuf));
	output ^= input[7] ^ cdata7;
}

#else
static __forceinline__ __device__ void fastkdf256_v2(uint32_t thread, const uint32_t nonce, const uint32_t* __restrict__ s_data) //, vectypeS * output)
{
	vectypeS __align__(16) output[8];
	uint8_t bufidx;
	uchar4 bufhelper;
	const uint32_t data18 = s_data[18];
	const uint32_t data20 = s_data[0];
	uint32_t input[16];
	uint32_t key[16] = {0};
	uint32_t qbuf, rbuf, bitbuf;

#define Bshift 16*thread

	uint32_t *const B = (uint32_t*)&B2[Bshift];
	((uintx64*)(B))[0] = ((uintx64*)s_data)[0];

	B[19] = nonce;
	B[39] = nonce;
	B[59] = nonce;
	__syncthreads();

	((ulonglong4*)input)[0] = ((ulonglong4*)input_init)[0];
	((uint28*)key)[0] = ((uint28*)key_init)[0];


#pragma unroll  1
	for(int i = 0; i < 31; ++i)
	{
		bufhelper = ((uchar4*)input)[0];
		for(int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x)
		{
			bufhelper += ((uchar4*)input)[x];
		}
		bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;

		qbuf = bufidx >> 2;
		rbuf = bufidx & 3;
		bitbuf = rbuf << 3;
		uint32_t shifted[9];

		shift256R4(shifted, ((uint8*)input)[0], bitbuf);

		uint32_t temp[9];

		for(int k = 0; k < 9; ++k)
			temp[k] = __ldg(&B[(k + qbuf) & 0x0000003f]) ^ shifted[k];

		uint32_t a = s_data[qbuf & 0x0000003f], b;
		//#pragma unroll

		for(int k = 0; k<16; k += 2)
		{
			b = s_data[(qbuf + k + 1) & 0x0000003f];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[k]) : "r"(a), "r"(b), "r"(bitbuf));
			a = s_data[(qbuf + k + 2) & 0x0000003f];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[k + 1]) : "r"(b), "r"(a), "r"(bitbuf));
		}

		const uint32_t noncepos = 19 - qbuf % 20;
		if(noncepos <= 16 && qbuf<60)
		{
			if(noncepos != 0)
				asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos - 1]) : "r"(data18), "r"(nonce), "r"(bitbuf));
			if(noncepos != 16)
				asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos]) : "r"(nonce), "r"(data20), "r"(bitbuf));
		}

		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[0]) : "r"(temp[0]), "r"(temp[1]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[1]) : "r"(temp[1]), "r"(temp[2]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[2]) : "r"(temp[2]), "r"(temp[3]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[3]) : "r"(temp[3]), "r"(temp[4]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[4]) : "r"(temp[4]), "r"(temp[5]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[5]) : "r"(temp[5]), "r"(temp[6]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[6]) : "r"(temp[6]), "r"(temp[7]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[7]) : "r"(temp[7]), "r"(temp[8]), "r"(bitbuf));

		Blake2S_v2(input, input, key);

		for(int k = 0; k < 9; k++)
			B[(k + qbuf) & 0x0000003f] = temp[k];
		__syncthreads();
	}

	bufhelper = ((uchar4*)input)[0];
	for(int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x)
	{
		bufhelper += ((uchar4*)input)[x];
	}
	bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;

	qbuf = bufidx / 4;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;

	for(int i = 0; i<64; i++)
	{
		const uint32_t a = (qbuf + i) & 0x0000003f, b = (qbuf + i + 1) & 0x0000003f;
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(((uint32_t*)output)[i]) : "r"(__ldg(&B[a])), "r"(__ldg(&B[b])), "r"(bitbuf));
	}

	output[0] ^= ((uint28*)input)[0];
	for(int i = 0; i<8; i++)
		output[i] ^= ((uint28*)s_data)[i];
	//	((ulonglong16 *)output)[0] ^= ((ulonglong16*)s_data)[0];
	((uint32_t*)output)[19] ^= nonce;
	((uint32_t*)output)[39] ^= nonce;
	((uint32_t*)output)[59] ^= nonce;;
	((ulonglong16 *)(Input + 8 * thread))[0] = ((ulonglong16*)output)[0];
}

static __forceinline__ __device__ void fastkdf32_v3(uint32_t thread, const  uint32_t  nonce, const uint32_t * __restrict__ salt, const uint32_t * __restrict__  s_data, uint32_t &output)
{
	uint32_t temp[9];
	uint8_t bufidx;
	uchar4 bufhelper;

#define Bshift 16*thread

	uint32_t*const B0 = (uint32_t*)&B2[Bshift];
	const uint32_t cdata7 = s_data[7];
	const uint32_t data18 = s_data[18];
	const uint32_t data20 = s_data[0];

	((uintx64*)B0)[0] = ((uintx64*)salt)[0];
	uint32_t input[BLAKE2S_BLOCK_SIZE / 4]; uint32_t key[BLAKE2S_BLOCK_SIZE / 4] = {0};
	((uint816*)input)[0] = ((uint816*)s_data)[0];
	((uint48*)key)[0] = ((uint48*)salt)[0];
	uint32_t qbuf, rbuf, bitbuf;
	__syncthreads();

#pragma nounroll  
	for(int i = 0; i < 31; i++)
	{
		Blake2S_v2(input, input, key);

		bufidx = 0;
		bufhelper = ((uchar4*)input)[0];
		for(int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x)
		{
			bufhelper += ((uchar4*)input)[x];
		}
		bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;
		qbuf = bufidx / 4;
		rbuf = bufidx & 3;
		bitbuf = rbuf << 3;
		uint32_t shifted[9];

		shift256R4(shifted, ((uint8*)input)[0], bitbuf);

		for(int k = 0; k < 9; k++)
		{
			temp[k] = __ldg(&B0[(k + qbuf) & 0x0000003f]);
		}

		((uint28*)temp)[0] ^= ((uint28*)shifted)[0];
		temp[8] ^= shifted[8];

		uint32_t a = s_data[qbuf & 0x0000003f], b;
		//#pragma unroll
		for(int k = 0; k<16; k += 2)
		{
			b = s_data[(qbuf + k + 1) & 0x0000003f];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[k]) : "r"(a), "r"(b), "r"(bitbuf));
			a = s_data[(qbuf + k + 2) & 0x0000003f];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[k + 1]) : "r"(b), "r"(a), "r"(bitbuf));
		}

		const uint32_t noncepos = 19 - qbuf % 20;
		if(noncepos <= 16 && qbuf<60)
		{
			if(noncepos != 0)
				asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos - 1]) : "r"(data18), "r"(nonce), "r"(bitbuf));
			if(noncepos != 16)
				asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos]) : "r"(nonce), "r"(data20), "r"(bitbuf));
		}

		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[0]) : "r"(temp[0]), "r"(temp[1]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[1]) : "r"(temp[1]), "r"(temp[2]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[2]) : "r"(temp[2]), "r"(temp[3]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[3]) : "r"(temp[3]), "r"(temp[4]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[4]) : "r"(temp[4]), "r"(temp[5]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[5]) : "r"(temp[5]), "r"(temp[6]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[6]) : "r"(temp[6]), "r"(temp[7]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[7]) : "r"(temp[7]), "r"(temp[8]), "r"(bitbuf));

		for(int k = 0; k < 9; k++)
		{
			B0[(k + qbuf) & 0x0000003f] = temp[k];
		}
		__syncthreads();
	}

	Blake2S_v2(input, input, key);

	bufidx = 0;
	bufhelper = ((uchar4*)input)[0];
	for(int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x)
	{
		bufhelper += ((uchar4*)input)[x];
	}
	bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;
	qbuf = bufidx / 4;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;

	temp[7] = __ldg(&B0[(qbuf + 7) & 0x0000003f]);
	temp[8] = __ldg(&B0[(qbuf + 8) & 0x0000003f]);
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(output) : "r"(temp[7]), "r"(temp[8]), "r"(bitbuf));
	output ^= input[7] ^ cdata7;
}

#endif


#define SHIFT 128
#define TPB 128
#define TPB2 64

__global__ __launch_bounds__(TPB2, 1) void neoscrypt_gpu_hash_start(int stratum, uint32_t threads, uint32_t startNonce)
{
	__shared__ uint32_t s_data[64];

#if TPB2<64
#error TPB2 too low
#else
#if TPB2>64
	if(threadIdx.x<64)
#endif
#endif
		s_data[threadIdx.x] = c_data[threadIdx.x];
	__syncthreads();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t nonce = startNonce + thread;
	if(thread >= threads)
		return;

	const uint32_t ZNonce = (stratum) ? cuda_swab32(nonce) : nonce; //freaking morons !!!

#if __CUDA_ARCH__ < 500
	fastkdf256_v1(thread, ZNonce, s_data);
#else 
	fastkdf256_v2(thread, ZNonce, s_data);
#endif

}

__global__ __launch_bounds__(TPB, 1) void neoscrypt_gpu_hash_chacha1_stream1(uint32_t threads, uint32_t startNonce)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const int shift = SHIFT * 8 * thread;
	const unsigned int shiftTr = 8 * thread;
	if(thread >= threads)
		return;

	vectypeS __align__(16) X[8];
	for(int i = 0; i<8; i++)
		X[i] = __ldg4(&(Input + shiftTr)[i]);

#pragma nounroll  
	for(int i = 0; i < 128; ++i)
	{
		uint32_t offset = shift + i * 8;
		for(int j = 0; j<8; j++)
			(W + offset)[j] = X[j];
		neoscrypt_chacha((uint16*)X);

	}
	for(int i = 0; i<8; i++)
		(Tr + shiftTr)[i] = X[i];
}

__global__ __launch_bounds__(TPB, 1) void neoscrypt_gpu_hash_chacha2_stream1(uint32_t threads, uint32_t startNonce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const int shift = SHIFT * 8 * thread;
	const int shiftTr = 8 * thread;
	if(thread >= threads)
		return;

	vectypeS __align__(16) X[8];
#pragma unroll
	for(int i = 0; i<8; i++)
		X[i] = __ldg4(&(Tr + shiftTr)[i]);

#pragma nounroll
	for(int t = 0; t < 128; t++)
	{
		int idx = (X[6].x.x & 0x7F) << 3;

		for(int j = 0; j<8; j++)
			X[j] ^= __ldg4(&(W + shift + idx)[j]);
		neoscrypt_chacha((uint16*)X);
	}
#pragma unroll
	for(int i = 0; i<8; i++)
		(Tr + shiftTr)[i] = X[i];  // best checked
}

__global__ __launch_bounds__(TPB, 1) void neoscrypt_gpu_hash_salsa1_stream1(uint32_t threads, uint32_t startNonce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const int shift = SHIFT * 8 * thread;
	const int shiftTr = 8 * thread;
	if(thread >= threads)
		return;

	vectypeS __align__(16) Z[8];
#pragma unroll
	for(int i = 0; i<8; i++)
		Z[i] = __ldg4(&(Input + shiftTr)[i]);

#pragma nounroll
	for(int i = 0; i < 128; ++i)
	{
		for(int j = 0; j<8; j++)
			(W2 + shift + i * 8)[j] = Z[j];
		neoscrypt_salsa((uint16*)Z);
	}
#pragma unroll
	for(int i = 0; i<8; i++)
		(Tr2 + shiftTr)[i] = Z[i];
}

__global__ __launch_bounds__(TPB, 1) void neoscrypt_gpu_hash_salsa2_stream1(uint32_t threads, uint32_t startNonce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const int shift = SHIFT * 8 * thread;
	const int shiftTr = 8 * thread;
	if(thread >= threads)
		return;

	vectypeS __align__(16) X[8];
#pragma unroll
	for(int i = 0; i<8; i++)
		X[i] = __ldg4(&(Tr2 + shiftTr)[i]);

#pragma nounroll 
	for(int t = 0; t < 128; t++)
	{
		int idx = (X[6].x.x & 0x7F) << 3;

		for(int j = 0; j<8; j++)
			X[j] ^= __ldg4(&(W2 + shift + idx)[j]);
		neoscrypt_salsa((uint16*)X);
	}
#pragma unroll
	for(int i = 0; i<8; i++)
		(Tr2 + shiftTr)[i] = X[i];  // best checked
}

__global__ __launch_bounds__(TPB2, 8) void neoscrypt_gpu_hash_ending(int stratum, uint32_t threads, uint32_t startNonce, uint32_t *nonceVector)
{
	__shared__ uint32_t s_data[64];

#if TPB2<64
#error TPB2 too low
#else
#if TPB2>64
	if(threadIdx.x<64)
#endif
#endif
		s_data[threadIdx.x] = c_data[threadIdx.x];
	__syncthreads();
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t nonce = startNonce + thread;
	if(thread >= threads)
		return;

	const int shiftTr = 8 * thread;
	vectypeS __align__(16) Z[8];
	uint32_t outbuf;

	const uint32_t ZNonce = (stratum) ? cuda_swab32(nonce) : nonce;

#pragma unroll
	for(int i = 0; i<8; i++)
		Z[i] = (Tr2 + shiftTr)[i] ^ (Tr + shiftTr)[i];

#if __CUDA_ARCH__ < 500		 
	fastkdf32_v1(thread, ZNonce, (uint32_t*)Z, s_data, outbuf);
#else
	fastkdf32_v3(thread, ZNonce, (uint32_t*)Z, s_data, outbuf);
#endif
	if(outbuf <= pTarget[7])
	{
		uint32_t tmp = atomicExch(nonceVector, nonce);
		if(tmp != 0xffffffff)
			nonceVector[1] = tmp;
	}
}

void neoscrypt_cpu_init_2stream(int thr_id, uint32_t threads)
{
	uint32_t *hash1;
	uint32_t *hash2; // 2 streams
	uint32_t *Trans1;
	uint32_t *Trans2; // 2 streams
	uint32_t *Trans3; // 2 streams
	uint32_t *Bhash;

	CUDA_SAFE_CALL(cudaStreamCreate(&stream[0]));
	CUDA_SAFE_CALL(cudaStreamCreate(&stream[1]));

	CUDA_SAFE_CALL(cudaMalloc(&d_NNonce[thr_id], 2 * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc(&hash1, 32ULL * 128 * sizeof(uint64_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&hash2, 32ULL * 128 * sizeof(uint64_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&Trans1, 32ULL * sizeof(uint64_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&Trans2, 32ULL * sizeof(uint64_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&Trans3, 32ULL * sizeof(uint64_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&Bhash, 128ULL * sizeof(uint32_t) * threads));

	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(B2, &Bhash, sizeof(uint28*), 0, cudaMemcpyHostToDevice, stream[0]));
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(W, &hash1, sizeof(uint28*), 0, cudaMemcpyHostToDevice, stream[0]));
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(W2, &hash2, sizeof(uint28*), 0, cudaMemcpyHostToDevice, stream[0]));
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(Tr, &Trans1, sizeof(uint28*), 0, cudaMemcpyHostToDevice, stream[0]));
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(Tr2, &Trans2, sizeof(uint28*), 0, cudaMemcpyHostToDevice, stream[0]));
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(Input, &Trans3, sizeof(uint28*), 0, cudaMemcpyHostToDevice, stream[0]));
	if(opt_debug)
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

__host__ void neoscrypt_cpu_hash_k4_2stream(bool stratum, int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *result)
{
	const uint32_t threadsperblock = TPB;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	const uint32_t threadsperblock2 = TPB2;
	dim3 grid2((threads + threadsperblock2 - 1) / threadsperblock2);
	dim3 block2(threadsperblock2);

	neoscrypt_gpu_hash_start << <grid2, block2, 0, stream[0] >> >(stratum, threads, startNounce); //fastkdf

	CUDA_SAFE_CALL(cudaStreamSynchronize(stream[0]));

	neoscrypt_gpu_hash_salsa1_stream1 << <grid, block, 0, stream[0] >> >(threads, startNounce); //chacha
	if(opt_debug)
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	neoscrypt_gpu_hash_chacha1_stream1 << <grid, block, 0, stream[1] >> >(threads, startNounce); //salsa
	if(opt_debug)
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

	neoscrypt_gpu_hash_salsa2_stream1 << <grid, block, 0, stream[0] >> >(threads, startNounce); //chacha
	if(opt_debug)
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	neoscrypt_gpu_hash_chacha2_stream1 << <grid, block, 0, stream[1] >> >(threads, startNounce); //salsa

	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	neoscrypt_gpu_hash_ending << <grid2, block2 >> >(stratum, threads, startNounce, d_NNonce[thr_id]); //fastkdf+end

	CUDA_SAFE_CALL(cudaMemcpy(result, d_NNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

__host__ void neoscrypt_setBlockTarget(int thr_id, uint32_t* pdata, const void *target)
{
	uint32_t PaddedMessage[64];
	uint32_t input[16], key[16] = {0};

	for(int i = 0; i < 19; i++)
	{
		PaddedMessage[i] = pdata[i];
		PaddedMessage[i + 20] = pdata[i];
		PaddedMessage[i + 40] = pdata[i];
	}
	for(int i = 0; i<4; i++)
		PaddedMessage[i + 60] = pdata[i];

	PaddedMessage[19] = 0;
	PaddedMessage[39] = 0;
	PaddedMessage[59] = 0;

	for(int i = 0; i < 16; i++)
		input[i] = pdata[i];
	for(int i = 0; i < 8; i++)
		key[i] = pdata[i];

	Blake2Shost(input, key);

	cudaMemcpyToSymbolAsync(pTarget, target, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream[1]);
	cudaMemcpyToSymbolAsync(input_init, input, 16 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyToSymbolAsync(key_init, key, 16 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream[1]);
	cudaMemcpyToSymbolAsync(c_data, PaddedMessage, 64 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream[0]);

	CUDA_SAFE_CALL(cudaMemsetAsync(d_NNonce[thr_id], 0xff, 2 * sizeof(uint32_t), stream[1]));
	if(opt_debug)
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

__global__ void get_cuda_arch_neo_gpu(int *d_version)
{
#ifdef __CUDA_ARCH__
	*d_version = __CUDA_ARCH__;
#endif
}

__host__ void get_cuda_arch_neo(int *version)
{
	int *d_version;
	cudaMalloc(&d_version, sizeof(int));
	get_cuda_arch_neo_gpu << < 1, 1 >> > (d_version);
	cudaMemcpy(version, d_version, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_version);
}
