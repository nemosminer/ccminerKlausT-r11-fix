#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"
#include "cuda_vector.h"

#define ROTR(x,n) ROTR64(x,n)

#define USE_SHUFFLE 0

static uint2* c_PaddedMessage80[MAX_GPUS]; // padded message (80 bytes + padding)
__constant__ uint2 __align__(16) c_PaddedM[10];
__constant__ uint28 Hostprecalc[4];
__constant__ uint2 __align__(16) pre[224];


__constant__ uint2 c_u512[16] =
{
	{0x85a308d3UL, 0x243f6a88}, {0x03707344UL, 0x13198a2e},
	{0x299f31d0UL, 0xa4093822}, {0xec4e6c89UL, 0x082efa98},
	{0x38d01377UL, 0x452821e6}, {0x34e90c6cUL, 0xbe5466cf},
	{0xc97c50ddUL, 0xc0ac29b7}, {0xb5470917UL, 0x3f84d5b5},
	{0x8979fb1bUL, 0x9216d5d9}, {0x98dfb5acUL, 0xd1310ba6},
	{0xd01adfb7UL, 0x2ffd72db}, {0x6a267e96UL, 0xb8e1afed},
	{0xf12c7f99UL, 0xba7c9045}, {0xb3916cf7UL, 0x24a19947},
	{0x858efc16UL, 0x0801f2e2}, {0x71574e69UL, 0x636920d8}
};

// ---------------------------- BEGIN CUDA quark_blake512 functions ------------------------------------

#define GSPREC_SP(a,b,c,d) { \
	v[a] += (pre[i++]) + v[b]; \
	v[d] = eorswap32( v[d] , v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 25); \
	v[a] += (pre[i++]) + v[b]; \
	v[d] = ROR16(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 11); \
	}

#define GSPREC_SP_HI(a,b,c,d,idx1,idx2) { \
	v[a] += (block[idx2] ^ u512[idx1]) + v[b]; \
	v[d] = eorswap32( v[d] , v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 25); \
	v[a] += (pre[i++]) + v[b]; \
	v[d] = ROR16(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 11); \
	}

#define GSPREC_SP_LO(a,b,c,d,idx1,idx2) { \
	v[a] += (pre[i++]) + v[b]; \
	v[d] = eorswap32( v[d] , v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 25); \
	v[a] += (block[idx1] ^ u512[idx2]) + v[b]; \
	v[d] = ROR16(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 11); \
		}

#define Gprecalc(a,b,c,d,idx1,idx2) { \
	v[a] += (block[idx2] ^ u512[idx1]) + v[b]; \
	v[d] = eorswap32( v[d] , v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 25); \
	v[a] += (block[idx1] ^ u512[idx2]) + v[b]; \
	v[d] = ROR16(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 11); \
	}

#define RSPRECHOST(idx1,idx2) { \
	prehost[i++] = (block[idx2] ^ u512[idx1]); \
	prehost[i++] = (block[idx1] ^ u512[idx2]); \
	}

#define RSPRECHOSTLO(idx1,idx2) { \
	prehost[i++] = (block[idx2] ^ u512[idx1]); \
	}
#define RSPRECHOSTHI(idx1,idx2) { \
	prehost[i++] = (block[idx1] ^ u512[idx2]); \
	}


#define GprecalcHost(a,b,c,d,idx1,idx2) { \
	v[a] += (block[idx2] ^ u512[idx1]) + v[b]; \
	v[d] = ROTR64( v[d] ^ v[a],32); \
	v[c] += v[d]; \
	v[b] = ROTR64(v[b] ^ v[c], 25); \
	v[a] += (block[idx1] ^ u512[idx2]) + v[b]; \
	v[d] = ROTR64(v[d] ^ v[a],16); \
	v[c] += v[d]; \
	v[b] = ROTR64(v[b] ^ v[c], 11); \
		}

__constant__ uint8_t c_sigma[16][16] = {
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
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
	{7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
	{9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
	{2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9}};


#define G(a,b,c,d,x) { \
	uint32_t idx1 = c_sigma[i][x]; \
	uint32_t idx2 = c_sigma[i][x+1]; \
	v[a] += (block[idx1] ^ c_u512[idx2]) + v[b]; \
	v[d] = eorswap32(v[d] , v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2( v[b] ^ v[c], 25); \
	v[a] += (block[idx2] ^ c_u512[idx1]) + v[b]; \
	v[d] = ROR16( v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2( v[b] ^ v[c], 11); \
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(256, 1)
#else
__launch_bounds__(256, 2)
#endif
void quark_blake512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *const __restrict__ g_nonceVector, uint2 *const __restrict__ g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

#if USE_SHUFFLE
	//	const int warpID = threadIdx.x & 0x02F; // 16 warps
	const int warpBlockID = (thread + 15) >> 5; // aufrunden auf volle Warp-Bl�cke
	//	const int maxHashPosition = thread<<3;
#endif

#if USE_SHUFFLE
	if(warpBlockID < ((threads + 15) >> 5))
#else
	if(thread < threads)
#endif
	{
		const uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		const int hashPosition = nounce - startNounce;

		uint2 block[16];
		uint2 msg[16];

		uint28 *phash = (uint28*)&g_hash[hashPosition * 8];
		uint28 *outpt = (uint28*)msg;
		outpt[0] = phash[0];
		outpt[1] = phash[1];
		block[0].x = cuda_swab32(msg[0].y);
		block[0].y = cuda_swab32(msg[0].x);
		block[1].x = cuda_swab32(msg[1].y);
		block[1].y = cuda_swab32(msg[1].x);
		block[2].x = cuda_swab32(msg[2].y);
		block[2].y = cuda_swab32(msg[2].x);
		block[3].x = cuda_swab32(msg[3].y);
		block[3].y = cuda_swab32(msg[3].x);
		block[4].x = cuda_swab32(msg[4].y);
		block[4].y = cuda_swab32(msg[4].x);
		block[5].x = cuda_swab32(msg[5].y);
		block[5].y = cuda_swab32(msg[5].x);
		block[6].x = cuda_swab32(msg[6].y);
		block[6].y = cuda_swab32(msg[6].x);
		block[7].x = cuda_swab32(msg[7].y);
		block[7].y = cuda_swab32(msg[7].x);


		block[8] = vectorizehigh(0x80000000);
		block[9] = vectorizelow(0x0);
		block[10] = vectorizelow(0x0);
		block[11] = vectorizelow(0x0);
		block[12] = vectorizelow(0x0);
		block[13] = vectorizelow(0x1);
		block[14] = vectorizelow(0x0);
		block[15] = vectorizelow(0x200);

		const uint2 h[8] =
		{
			{0xf3bcc908UL, 0x6a09e667UL},
			{0x84caa73bUL, 0xbb67ae85UL},
			{0xfe94f82bUL, 0x3c6ef372UL},
			{0x5f1d36f1UL, 0xa54ff53aUL},
			{0xade682d1UL, 0x510e527fUL},
			{0x2b3e6c1fUL, 0x9b05688cUL},
			{0xfb41bd6bUL, 0x1f83d9abUL},
			{0x137e2179UL, 0x5be0cd19UL}
		};
		const uint2 u512[16] =
		{
			{0x85a308d3UL, 0x243f6a88}, {0x03707344UL, 0x13198a2e},
			{0x299f31d0UL, 0xa4093822}, {0xec4e6c89UL, 0x082efa98},
			{0x38d01377UL, 0x452821e6}, {0x34e90c6cUL, 0xbe5466cf},
			{0xc97c50ddUL, 0xc0ac29b7}, {0xb5470917UL, 0x3f84d5b5},
			{0x8979fb1bUL, 0x9216d5d9}, {0x98dfb5acUL, 0xd1310ba6},
			{0xd01adfb7UL, 0x2ffd72db}, {0x6a267e96UL, 0xb8e1afed},
			{0xf12c7f99UL, 0xba7c9045}, {0xb3916cf7UL, 0x24a19947},
			{0x858efc16UL, 0x0801f2e2}, {0x71574e69UL, 0x636920d8}
		};

		uint2 v[16] =
		{
			h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
			u512[0], u512[1], u512[2], u512[3], u512[4] ^ 512, u512[5] ^ 512, u512[6], u512[7]
		};

		Gprecalc(0, 4, 8, 12, 0x1, 0x0)
			Gprecalc(1, 5, 9, 13, 0x3, 0x2)
			Gprecalc(2, 6, 10, 14, 0x5, 0x4)
			Gprecalc(3, 7, 11, 15, 0x7, 0x6)
			Gprecalc(0, 5, 10, 15, 0x9, 0x8)
			Gprecalc(1, 6, 11, 12, 0xb, 0xa)
			Gprecalc(2, 7, 8, 13, 0xd, 0xc)
			Gprecalc(3, 4, 9, 14, 0xf, 0xe)

			Gprecalc(0, 4, 8, 12, 0xa, 0xe)
			Gprecalc(1, 5, 9, 13, 0x8, 0x4)
			Gprecalc(2, 6, 10, 14, 0xf, 0x9)
			Gprecalc(3, 7, 11, 15, 0x6, 0xd)
			Gprecalc(0, 5, 10, 15, 0xc, 0x1)
			Gprecalc(1, 6, 11, 12, 0x2, 0x0)
			Gprecalc(2, 7, 8, 13, 0x7, 0xb)
			Gprecalc(3, 4, 9, 14, 0x3, 0x5)

			Gprecalc(0, 4, 8, 12, 0x8, 0xb)
			Gprecalc(1, 5, 9, 13, 0x0, 0xc)
			Gprecalc(2, 6, 10, 14, 0x2, 0x5)
			Gprecalc(3, 7, 11, 15, 0xd, 0xf)
			Gprecalc(0, 5, 10, 15, 0xe, 0xa)
			Gprecalc(1, 6, 11, 12, 0x6, 0x3)
			Gprecalc(2, 7, 8, 13, 0x1, 0x7)
			Gprecalc(3, 4, 9, 14, 0x4, 0x9)

			Gprecalc(0, 4, 8, 12, 0x9, 0x7)
			Gprecalc(1, 5, 9, 13, 0x1, 0x3)
			Gprecalc(2, 6, 10, 14, 0xc, 0xd)
			Gprecalc(3, 7, 11, 15, 0xe, 0xb)
			Gprecalc(0, 5, 10, 15, 0x6, 0x2)
			Gprecalc(1, 6, 11, 12, 0xa, 0x5)
			Gprecalc(2, 7, 8, 13, 0x0, 0x4)
			Gprecalc(3, 4, 9, 14, 0x8, 0xf)

			Gprecalc(0, 4, 8, 12, 0x0, 0x9)
			Gprecalc(1, 5, 9, 13, 0x7, 0x5)
			Gprecalc(2, 6, 10, 14, 0x4, 0x2)
			Gprecalc(3, 7, 11, 15, 0xf, 0xa)
			Gprecalc(0, 5, 10, 15, 0x1, 0xe)
			Gprecalc(1, 6, 11, 12, 0xc, 0xb)
			Gprecalc(2, 7, 8, 13, 0x8, 0x6)
			Gprecalc(3, 4, 9, 14, 0xd, 0x3)

			Gprecalc(0, 4, 8, 12, 0xc, 0x2)
			Gprecalc(1, 5, 9, 13, 0xa, 0x6)
			Gprecalc(2, 6, 10, 14, 0xb, 0x0)
			Gprecalc(3, 7, 11, 15, 0x3, 0x8)
			Gprecalc(0, 5, 10, 15, 0xd, 0x4)
			Gprecalc(1, 6, 11, 12, 0x5, 0x7)
			Gprecalc(2, 7, 8, 13, 0xe, 0xf)
			Gprecalc(3, 4, 9, 14, 0x9, 0x1)

			Gprecalc(0, 4, 8, 12, 0x5, 0xc)
			Gprecalc(1, 5, 9, 13, 0xf, 0x1)
			Gprecalc(2, 6, 10, 14, 0xd, 0xe)
			Gprecalc(3, 7, 11, 15, 0xa, 0x4)
			Gprecalc(0, 5, 10, 15, 0x7, 0x0)
			Gprecalc(1, 6, 11, 12, 0x3, 0x6)
			Gprecalc(2, 7, 8, 13, 0x2, 0x9)
			Gprecalc(3, 4, 9, 14, 0xb, 0x8)

			Gprecalc(0, 4, 8, 12, 0xb, 0xd)
			Gprecalc(1, 5, 9, 13, 0xe, 0x7)
			Gprecalc(2, 6, 10, 14, 0x1, 0xc)
			Gprecalc(3, 7, 11, 15, 0x9, 0x3)
			Gprecalc(0, 5, 10, 15, 0x0, 0x5)
			Gprecalc(1, 6, 11, 12, 0x4, 0xf)
			Gprecalc(2, 7, 8, 13, 0x6, 0x8)
			Gprecalc(3, 4, 9, 14, 0xa, 0x2)

			Gprecalc(0, 4, 8, 12, 0xf, 0x6)
			Gprecalc(1, 5, 9, 13, 0x9, 0xe)
			Gprecalc(2, 6, 10, 14, 0x3, 0xb)
			Gprecalc(3, 7, 11, 15, 0x8, 0x0)
			Gprecalc(0, 5, 10, 15, 0x2, 0xc)
			Gprecalc(1, 6, 11, 12, 0x7, 0xd)
			Gprecalc(2, 7, 8, 13, 0x4, 0x1)
			Gprecalc(3, 4, 9, 14, 0x5, 0xa)

			Gprecalc(0, 4, 8, 12, 0x2, 0xa)
			Gprecalc(1, 5, 9, 13, 0x4, 0x8)
			Gprecalc(2, 6, 10, 14, 0x6, 0x7)
			Gprecalc(3, 7, 11, 15, 0x5, 0x1)
			Gprecalc(0, 5, 10, 15, 0xb, 0xf)
			Gprecalc(1, 6, 11, 12, 0xe, 0x9)
			Gprecalc(2, 7, 8, 13, 0xc, 0x3)
			Gprecalc(3, 4, 9, 14, 0x0, 0xd)

#if __CUDA_ARCH__ == 500

			Gprecalc(0, 4, 8, 12, 0x1, 0x0)
			Gprecalc(1, 5, 9, 13, 0x3, 0x2)
			Gprecalc(2, 6, 10, 14, 0x5, 0x4)
			Gprecalc(3, 7, 11, 15, 0x7, 0x6)
			Gprecalc(0, 5, 10, 15, 0x9, 0x8)
			Gprecalc(1, 6, 11, 12, 0xb, 0xa)
			Gprecalc(2, 7, 8, 13, 0xd, 0xc)
			Gprecalc(3, 4, 9, 14, 0xf, 0xe)

			Gprecalc(0, 4, 8, 12, 0xa, 0xe)
			Gprecalc(1, 5, 9, 13, 0x8, 0x4)
			Gprecalc(2, 6, 10, 14, 0xf, 0x9)
			Gprecalc(3, 7, 11, 15, 0x6, 0xd)
			Gprecalc(0, 5, 10, 15, 0xc, 0x1)
			Gprecalc(1, 6, 11, 12, 0x2, 0x0)
			Gprecalc(2, 7, 8, 13, 0x7, 0xb)
			Gprecalc(3, 4, 9, 14, 0x3, 0x5)

			Gprecalc(0, 4, 8, 12, 0x8, 0xb)
			Gprecalc(1, 5, 9, 13, 0x0, 0xc)
			Gprecalc(2, 6, 10, 14, 0x2, 0x5)
			Gprecalc(3, 7, 11, 15, 0xd, 0xf)
			Gprecalc(0, 5, 10, 15, 0xe, 0xa)
			Gprecalc(1, 6, 11, 12, 0x6, 0x3)
			Gprecalc(2, 7, 8, 13, 0x1, 0x7)
			Gprecalc(3, 4, 9, 14, 0x4, 0x9)

			Gprecalc(0, 4, 8, 12, 0x9, 0x7)
			Gprecalc(1, 5, 9, 13, 0x1, 0x3)
			Gprecalc(2, 6, 10, 14, 0xc, 0xd)
			Gprecalc(3, 7, 11, 15, 0xe, 0xb)
			Gprecalc(0, 5, 10, 15, 0x6, 0x2)
			Gprecalc(1, 6, 11, 12, 0xa, 0x5)
			Gprecalc(2, 7, 8, 13, 0x0, 0x4)
			Gprecalc(3, 4, 9, 14, 0x8, 0xf)

			Gprecalc(0, 4, 8, 12, 0x0, 0x9)
			Gprecalc(1, 5, 9, 13, 0x7, 0x5)
			Gprecalc(2, 6, 10, 14, 0x4, 0x2)
			Gprecalc(3, 7, 11, 15, 0xf, 0xa)
			Gprecalc(0, 5, 10, 15, 0x1, 0xe)
			Gprecalc(1, 6, 11, 12, 0xc, 0xb)
			Gprecalc(2, 7, 8, 13, 0x8, 0x6)
			Gprecalc(3, 4, 9, 14, 0xd, 0x3)

			Gprecalc(0, 4, 8, 12, 0xc, 0x2)
			Gprecalc(1, 5, 9, 13, 0xa, 0x6)
			Gprecalc(2, 6, 10, 14, 0xb, 0x0)
			Gprecalc(3, 7, 11, 15, 0x3, 0x8)
			Gprecalc(0, 5, 10, 15, 0xd, 0x4)
			Gprecalc(1, 6, 11, 12, 0x5, 0x7)
			Gprecalc(2, 7, 8, 13, 0xe, 0xf)
			Gprecalc(3, 4, 9, 14, 0x9, 0x1)

#else

			for(int i = 10; i < 16; i++)
			{
				/* column step */
				G(0, 4, 8, 12, 0);
				G(1, 5, 9, 13, 2);
				G(2, 6, 10, 14, 4);
				G(3, 7, 11, 15, 6);
				/* diagonal step */
				G(0, 5, 10, 15, 8);
				G(1, 6, 11, 12, 10);
				G(2, 7, 8, 13, 12);
				G(3, 4, 9, 14, 14);
			}
#endif

		v[0] = cuda_swap(h[0] ^ v[0] ^ v[8]);
		v[1] = cuda_swap(h[1] ^ v[1] ^ v[9]);
		v[2] = cuda_swap(h[2] ^ v[2] ^ v[10]);
		v[3] = cuda_swap(h[3] ^ v[3] ^ v[11]);
		v[4] = cuda_swap(h[4] ^ v[4] ^ v[12]);
		v[5] = cuda_swap(h[5] ^ v[5] ^ v[13]);
		v[6] = cuda_swap(h[6] ^ v[6] ^ v[14]);
		v[7] = cuda_swap(h[7] ^ v[7] ^ v[15]);

		phash = (uint28*)v;
		outpt = (uint28*)&g_hash[hashPosition * 8];
		outpt[0] = phash[0];
		outpt[1] = phash[1];
	}
}

__global__
void quark_blake512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if(thread < threads)
	{
		const uint32_t nounce = startNounce + thread;
		uint2 block[16];

		block[0] = c_PaddedM[0];
		block[1] = c_PaddedM[1];
		block[2] = c_PaddedM[2];
		block[3] = c_PaddedM[3];
		block[4] = c_PaddedM[4];
		block[5] = c_PaddedM[5];
		block[6] = c_PaddedM[6];
		block[7] = c_PaddedM[7];
		block[8] = c_PaddedM[8];
		block[9].y = c_PaddedM[9].y;
		block[10] = vectorizehigh(0x80000000);
		block[11] = vectorizelow(0);
		block[12] = vectorizelow(0);
		block[13] = vectorizelow(0x1);
		block[14] = vectorizelow(0);
		block[15] = vectorizelow(0x280);
		block[9].x = nounce;

		const uint2 u512[16] =
		{
			{0x85a308d3UL, 0x243f6a88}, {0x03707344UL, 0x13198a2e},
			{0x299f31d0UL, 0xa4093822}, {0xec4e6c89UL, 0x082efa98},
			{0x38d01377UL, 0x452821e6}, {0x34e90c6cUL, 0xbe5466cf},
			{0xc97c50ddUL, 0xc0ac29b7}, {0xb5470917UL, 0x3f84d5b5},
			{0x8979fb1bUL, 0x9216d5d9}, {0x98dfb5acUL, 0xd1310ba6},
			{0xd01adfb7UL, 0x2ffd72db}, {0x6a267e96UL, 0xb8e1afed},
			{0xf12c7f99UL, 0xba7c9045}, {0xb3916cf7UL, 0x24a19947},
			{0x858efc16UL, 0x0801f2e2}, {0x71574e69UL, 0x636920d8}
		};

		/*		const uint2 u512[16] =
		{
		{ 0x85a308d3UL, 0x243f6a88 }, { 0x03707344UL, 0x13198a2e },
		{ 0x299f31d0UL, 0xa4093822 }, { 0xec4e6c89UL, 0x082efa98 },
		{ 0x38d01377UL, 0x452821e6 }, { 0x34e90c6cUL, 0xbe5466cf },
		{ 0xc97c50ddUL, 0xc0ac29b7 }, { 0xb5470917UL, 0x3f84d5b5 },
		{ 0x8979fb1bUL, 0x9216d5d9 }, { 0x98dfb5acUL, 0xd1310ba6 },
		{ 0xd01adfb7UL, 0x2ffd72db }, { 0x6a267e96UL, 0xb8e1afed },
		{ 0xf12c7f99UL, 0xba7c9045 }, { 0xb3916cf7UL, 0x24a19947 },
		{ 0x858efc16UL, 0x0801f2e2 }, { 0x71574e69UL, 0x636920d8 }
		};
		*/
		const uint2 h[8] = {
			{0xf3bcc908UL, 0x6a09e667UL},
			{0x84caa73bUL, 0xbb67ae85UL},
			{0xfe94f82bUL, 0x3c6ef372UL},
			{0x5f1d36f1UL, 0xa54ff53aUL},
			{0xade682d1UL, 0x510e527fUL},
			{0x2b3e6c1fUL, 0x9b05688cUL},
			{0xfb41bd6bUL, 0x1f83d9abUL},
			{0x137e2179UL, 0x5be0cd19UL}
		};
		uint2 v[16];
		uint28 *outpt = (uint28*)v;
		outpt[0] = Hostprecalc[0];
		outpt[1] = Hostprecalc[1];
		outpt[2] = Hostprecalc[2];
		outpt[3] = Hostprecalc[3];

		int i = 0;

		v[0] += (block[9] ^ c_u512[8]);
		v[15] = ROR16(v[15] ^ v[0]);
		v[10] += v[15];
		v[5] = ROR2(v[5] ^ v[10], 11);

		GSPREC_SP(0, 4, 8, 12)

			//		Gprecalc(1, 5, 9, 13, 0x8, 0x4)
			v[1] += v[5];
		v[13] = eorswap32(v[13], v[1]);
		v[9] += v[13];

		v[5] = ROR2(v[5] ^ v[9], 25);
		v[1] += (pre[i++]) + v[5];
		v[13] = ROR16(v[13] ^ v[1]);
		v[9] += v[13];
		v[5] = ROR2(v[5] ^ v[9], 11);

		//		Gprecalc(2, 6, 10, 14, 0xf, 0x9)
		v[2] += (block[9] ^ c_u512[0xf]);
		v[14] = eorswap32(v[14], v[2]);
		v[10] += v[14];
		v[6] = ROR2(v[6] ^ v[10], 25);
		v[2] += pre[i++] + v[6];
		v[14] = ROR16(v[14] ^ v[2]);
		v[10] += v[14];
		v[6] = ROR2(v[6] ^ v[10], 11);

		//		Gprecalc(3, 7, 11, 15, 0x6, 0xd)
		v[15] = eorswap32(v[15], v[3]);
		v[11] += v[15];
		v[7] = ROR2(v[7] ^ v[11], 25);
		v[3] += pre[i++] + v[7];
		v[15] = ROR16(v[15] ^ v[3]);
		v[11] += v[15];
		v[7] = ROR2(v[7] ^ v[11], 11);

		GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP(3, 4, 9, 14)

			GSPREC_SP(0, 4, 8, 12)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP_HI(3, 4, 9, 14, 0x4, 0x9)

			GSPREC_SP_LO(0, 4, 8, 12, 0x9, 0x7)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP(3, 4, 9, 14)


			GSPREC_SP_HI(0, 4, 8, 12, 0x0, 0x9)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP(3, 4, 9, 14)

			GSPREC_SP(0, 4, 8, 12)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP_LO(3, 4, 9, 14, 0x9, 0x1)

			GSPREC_SP(0, 4, 8, 12)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP_HI(2, 7, 8, 13, 0x2, 0x9)
			GSPREC_SP(3, 4, 9, 14)


			GSPREC_SP(0, 4, 8, 12)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP_LO(3, 7, 11, 15, 0x9, 0x3)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP(3, 4, 9, 14)

			GSPREC_SP(0, 4, 8, 12)
			GSPREC_SP_LO(1, 5, 9, 13, 0x9, 0xe)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP(3, 4, 9, 14)

			GSPREC_SP(0, 4, 8, 12)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP_HI(1, 6, 11, 12, 0xe, 0x9)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP(3, 4, 9, 14)

			GSPREC_SP(0, 4, 8, 12)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP_LO(0, 5, 10, 15, 0x9, 0x8)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP(3, 4, 9, 14)

			GSPREC_SP(0, 4, 8, 12)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP_HI(2, 6, 10, 14, 0xf, 0x9)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP(3, 4, 9, 14)

			GSPREC_SP(0, 4, 8, 12)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP_HI(3, 4, 9, 14, 0x4, 0x9)

			GSPREC_SP_LO(0, 4, 8, 12, 0x9, 0x7)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP(3, 4, 9, 14)

			GSPREC_SP_HI(0, 4, 8, 12, 0x0, 0x9)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP(3, 4, 9, 14)

			GSPREC_SP(0, 4, 8, 12)
			GSPREC_SP(1, 5, 9, 13)
			GSPREC_SP(2, 6, 10, 14)
			GSPREC_SP(3, 7, 11, 15)
			GSPREC_SP(0, 5, 10, 15)
			GSPREC_SP(1, 6, 11, 12)
			GSPREC_SP(2, 7, 8, 13)
			GSPREC_SP_LO(3, 4, 9, 14, 0x9, 0x1)


			/*			Gprecalc(0, 5, 10, 15, 0xc, 0x1)
			Gprecalc(1, 6, 11, 12, 0x2, 0x0)
			Gprecalc(2, 7, 8, 13, 0x7, 0xb)
			Gprecalc(3, 4, 9, 14, 0x3, 0x5)
			Gprecalc(0, 4, 8, 12, 0x8, 0xb)
			Gprecalc(1, 5, 9, 13, 0x0, 0xc)
			Gprecalc(2, 6, 10, 14, 0x2, 0x5)
			Gprecalc(3, 7, 11, 15, 0xd, 0xf)
			Gprecalc(0, 5, 10, 15, 0xe, 0xa)
			Gprecalc(1, 6, 11, 12, 0x6, 0x3)
			Gprecalc(2, 7, 8, 13, 0x1, 0x7)
			Gprecalc(3, 4, 9, 14, 0x4, 0x9)
			Gprecalc(0, 4, 8, 12, 0x9, 0x7)
			Gprecalc(1, 5, 9, 13, 0x1, 0x3)
			Gprecalc(2, 6, 10, 14, 0xc, 0xd)
			Gprecalc(3, 7, 11, 15, 0xe, 0xb)
			Gprecalc(0, 5, 10, 15, 0x6, 0x2)
			Gprecalc(1, 6, 11, 12, 0xa, 0x5)
			Gprecalc(2, 7, 8, 13, 0x0, 0x4)
			Gprecalc(3, 4, 9, 14, 0x8, 0xf)
			Gprecalc(0, 4, 8, 12, 0x0, 0x9)
			Gprecalc(1, 5, 9, 13, 0x7, 0x5)
			Gprecalc(2, 6, 10, 14, 0x4, 0x2)
			Gprecalc(3, 7, 11, 15, 0xf, 0xa)
			Gprecalc(0, 5, 10, 15, 0x1, 0xe)
			Gprecalc(1, 6, 11, 12, 0xc, 0xb)
			Gprecalc(2, 7, 8, 13, 0x8, 0x6)
			Gprecalc(3, 4, 9, 14, 0xd, 0x3)
			Gprecalc(0, 4, 8, 12, 0xc, 0x2)
			Gprecalc(1, 5, 9, 13, 0xa, 0x6)
			Gprecalc(2, 6, 10, 14, 0xb, 0x0)
			Gprecalc(3, 7, 11, 15, 0x3, 0x8)
			Gprecalc(0, 5, 10, 15, 0xd, 0x4)
			Gprecalc(1, 6, 11, 12, 0x5, 0x7)
			Gprecalc(2, 7, 8, 13, 0xe, 0xf)
			Gprecalc(3, 4, 9, 14, 0x9, 0x1)
			Gprecalc(0, 4, 8, 12, 0x5, 0xc)
			Gprecalc(1, 5, 9, 13, 0xf, 0x1)
			Gprecalc(2, 6, 10, 14, 0xd, 0xe)
			Gprecalc(3, 7, 11, 15, 0xa, 0x4)
			Gprecalc(0, 5, 10, 15, 0x7, 0x0)
			Gprecalc(1, 6, 11, 12, 0x3, 0x6)
			Gprecalc(2, 7, 8, 13, 0x2, 0x9)
			Gprecalc(3, 4, 9, 14, 0xb, 0x8)
			Gprecalc(0, 4, 8, 12, 0xb, 0xd)
			Gprecalc(1, 5, 9, 13, 0xe, 0x7)
			Gprecalc(2, 6, 10, 14, 0x1, 0xc)
			Gprecalc(3, 7, 11, 15, 0x9, 0x3)
			Gprecalc(0, 5, 10, 15, 0x0, 0x5)
			Gprecalc(1, 6, 11, 12, 0x4, 0xf)
			Gprecalc(2, 7, 8, 13, 0x6, 0x8)
			Gprecalc(3, 4, 9, 14, 0xa, 0x2)
			Gprecalc(0, 4, 8, 12, 0xf, 0x6)
			Gprecalc(1, 5, 9, 13, 0x9, 0xe)
			Gprecalc(2, 6, 10, 14, 0x3, 0xb)
			Gprecalc(3, 7, 11, 15, 0x8, 0x0)
			Gprecalc(0, 5, 10, 15, 0x2, 0xc)
			Gprecalc(1, 6, 11, 12, 0x7, 0xd)
			Gprecalc(2, 7, 8, 13, 0x4, 0x1)
			Gprecalc(3, 4, 9, 14, 0x5, 0xa)
			Gprecalc(0, 4, 8, 12, 0x2, 0xa)
			Gprecalc(1, 5, 9, 13, 0x4, 0x8)
			Gprecalc(2, 6, 10, 14, 0x6, 0x7)
			Gprecalc(3, 7, 11, 15, 0x5, 0x1)
			Gprecalc(0, 5, 10, 15, 0xb, 0xf)
			Gprecalc(1, 6, 11, 12, 0xe, 0x9)
			Gprecalc(2, 7, 8, 13, 0xc, 0x3)
			Gprecalc(3, 4, 9, 14, 0x0, 0xd)
			Gprecalc(0, 4, 8, 12, 0x1, 0x0)
			Gprecalc(1, 5, 9, 13, 0x3, 0x2)
			Gprecalc(2, 6, 10, 14, 0x5, 0x4)
			Gprecalc(3, 7, 11, 15, 0x7, 0x6)
			Gprecalc(0, 5, 10, 15, 0x9, 0x8)
			Gprecalc(1, 6, 11, 12, 0xb, 0xa)
			Gprecalc(2, 7, 8, 13, 0xd, 0xc)
			Gprecalc(3, 4, 9, 14, 0xf, 0xe)
			Gprecalc(0, 4, 8, 12, 0xa, 0xe)
			Gprecalc(1, 5, 9, 13, 0x8, 0x4)
			Gprecalc(2, 6, 10, 14, 0xf, 0x9)
			Gprecalc(3, 7, 11, 15, 0x6, 0xd)
			Gprecalc(0, 5, 10, 15, 0xc, 0x1)
			Gprecalc(1, 6, 11, 12, 0x2, 0x0)
			Gprecalc(2, 7, 8, 13, 0x7, 0xb)
			Gprecalc(3, 4, 9, 14, 0x3, 0x5)
			Gprecalc(0, 4, 8, 12, 0x8, 0xb)
			Gprecalc(1, 5, 9, 13, 0x0, 0xc)
			Gprecalc(2, 6, 10, 14, 0x2, 0x5)
			Gprecalc(3, 7, 11, 15, 0xd, 0xf)
			Gprecalc(0, 5, 10, 15, 0xe, 0xa)
			Gprecalc(1, 6, 11, 12, 0x6, 0x3)
			Gprecalc(2, 7, 8, 13, 0x1, 0x7)
			Gprecalc(3, 4, 9, 14, 0x4, 0x9)
			Gprecalc(0, 4, 8, 12, 0x9, 0x7)
			Gprecalc(1, 5, 9, 13, 0x1, 0x3)
			Gprecalc(2, 6, 10, 14, 0xc, 0xd)
			Gprecalc(3, 7, 11, 15, 0xe, 0xb)
			Gprecalc(0, 5, 10, 15, 0x6, 0x2)
			Gprecalc(1, 6, 11, 12, 0xa, 0x5)
			Gprecalc(2, 7, 8, 13, 0x0, 0x4)
			Gprecalc(3, 4, 9, 14, 0x8, 0xf)
			Gprecalc(0, 4, 8, 12, 0x0, 0x9)
			Gprecalc(1, 5, 9, 13, 0x7, 0x5)
			Gprecalc(2, 6, 10, 14, 0x4, 0x2)
			Gprecalc(3, 7, 11, 15, 0xf, 0xa)
			Gprecalc(0, 5, 10, 15, 0x1, 0xe)
			Gprecalc(1, 6, 11, 12, 0xc, 0xb)
			Gprecalc(2, 7, 8, 13, 0x8, 0x6)
			Gprecalc(3, 4, 9, 14, 0xd, 0x3)
			Gprecalc(0, 4, 8, 12, 0xc, 0x2)
			Gprecalc(1, 5, 9, 13, 0xa, 0x6)
			Gprecalc(2, 6, 10, 14, 0xb, 0x0)
			Gprecalc(3, 7, 11, 15, 0x3, 0x8)
			Gprecalc(0, 5, 10, 15, 0xd, 0x4)
			Gprecalc(1, 6, 11, 12, 0x5, 0x7)
			Gprecalc(2, 7, 8, 13, 0xe, 0xf)
			Gprecalc(3, 4, 9, 14, 0x9, 0x1)
			*/
		v[0] = cuda_swap(h[0] ^ v[0] ^ v[8]);
		v[1] = cuda_swap(h[1] ^ v[1] ^ v[9]);
		v[2] = cuda_swap(h[2] ^ v[2] ^ v[10]);
		v[3] = cuda_swap(h[3] ^ v[3] ^ v[11]);
		v[4] = cuda_swap(h[4] ^ v[4] ^ v[12]);
		v[5] = cuda_swap(h[5] ^ v[5] ^ v[13]);
		v[6] = cuda_swap(h[6] ^ v[6] ^ v[14]);
		v[7] = cuda_swap(h[7] ^ v[7] ^ v[15]);

		uint28 *phash = (uint28*)v;
		outpt = (uint28*)&outputHash[8 * thread];
		outpt[0] = phash[0];
		outpt[1] = phash[1];
	}
}



// ---------------------------- END CUDA quark_blake512 functions ------------------------------------

__host__ void quark_blake512_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&c_PaddedMessage80[thr_id], 10 * sizeof(uint2)));
}

__host__ void quark_blake512_cpu_setBlock_80_multi(uint32_t thr_id, uint64_t *pdata)
{
	uint64_t PaddedMessage[10];
	for(int i = 0; i < 10; i++)
		PaddedMessage[i] = cuda_swab64(pdata[i]);
	CUDA_SAFE_CALL(cudaMemcpy(c_PaddedMessage80[thr_id], PaddedMessage, 10 * sizeof(uint64_t), cudaMemcpyHostToDevice));
}

__host__ void quark_blake512_cpu_setBlock_80(int thr_id, uint64_t *pdata)
{
	const uint64_t u512[16] =
	{
		0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,
		0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
		0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL,
		0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
		0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL,
		0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
		0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL,
		0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
	};

	uint64_t h[8] = {
		0x6a09e667f3bcc908ULL,
		0xbb67ae8584caa73bULL,
		0x3c6ef372fe94f82bULL,
		0xa54ff53a5f1d36f1ULL,
		0x510e527fade682d1ULL,
		0x9b05688c2b3e6c1fULL,
		0x1f83d9abfb41bd6bULL,
		0x5be0cd19137e2179ULL
	};

	uint64_t v[16] =
	{
		h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
		u512[0], u512[1], u512[2], u512[3], u512[4] ^ 640, u512[5] ^ 640, u512[6], u512[7]
	};

	uint64_t PaddedMessage[10];
	uint64_t block[16];
	uint64_t prehost[224];

	for(int i = 0; i < 10; i++)
	{
		PaddedMessage[i] = cuda_swab64(pdata[i]);
		block[i] = PaddedMessage[i];
	}
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(c_PaddedM, PaddedMessage, 10 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice, gpustream[thr_id]));
	
	block[10] = 0x8000000000000000;
	block[11] = 0;
	block[12] = 0;
	block[13] = 1;
	block[14] = 0;
	block[15] = 0x280;

	GprecalcHost(0, 4, 8, 12, 0x1, 0x0)
	GprecalcHost(1, 5, 9, 13, 0x3, 0x2)
	GprecalcHost(2, 6, 10, 14, 0x5, 0x4)
	GprecalcHost(3, 7, 11, 15, 0x7, 0x6)

	GprecalcHost(1, 6, 11, 12, 0xb, 0xa)
	GprecalcHost(2, 7, 8, 13, 0xd, 0xc)

	v[0] += (block[8] ^ u512[9]) + v[5];
	v[15] = ROTR64(v[15] ^ v[0], 32);
	v[10] += v[15];
	v[5] = ROTR64(v[5] ^ v[10], 25);
	v[0] += v[5];

	GprecalcHost(3, 4, 9, 14, 0xf, 0xe);

	v[1] += (block[0x4] ^ u512[0x8]);
	v[2] += v[6];
	v[3] += (block[0xd] ^ u512[6]) + v[7];
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(Hostprecalc, v, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice, gpustream[thr_id]));

	int i = 0;
	RSPRECHOST(0xa, 0xe);
	prehost[i++] = block[8] ^ u512[4];
	prehost[i++] = block[0xf] ^ u512[9];
	prehost[i++] = block[6] ^ u512[0xd];


	RSPRECHOST(0xc, 0x1)
	RSPRECHOST(0x2, 0x0)
	RSPRECHOST(0x7, 0xb)
	RSPRECHOST(0x3, 0x5)

	RSPRECHOST(0x8, 0xb)
	RSPRECHOST(0x0, 0xc)
	RSPRECHOST(0x2, 0x5)
	RSPRECHOST(0xd, 0xf)
	RSPRECHOST(0xe, 0xa)
	RSPRECHOST(0x6, 0x3)
	RSPRECHOST(0x1, 0x7)
	RSPRECHOSTHI(0x4, 0x9)

	RSPRECHOSTLO(0x9, 0x7)
	RSPRECHOST(0x1, 0x3)
	RSPRECHOST(0xc, 0xd)
	RSPRECHOST(0xe, 0xb)
	RSPRECHOST(0x6, 0x2)
	RSPRECHOST(0xa, 0x5)
	RSPRECHOST(0x0, 0x4)
	RSPRECHOST(0x8, 0xf)

	RSPRECHOSTHI(0, 0x9)
	RSPRECHOST(0x7, 0x5)
	RSPRECHOST(0x4, 0x2)
	RSPRECHOST(0xf, 0xa)
	RSPRECHOST(0x1, 0xe)
	RSPRECHOST(0xc, 0xb)
	RSPRECHOST(0x8, 0x6)
	RSPRECHOST(0xd, 0x3)

	RSPRECHOST(0xc, 0x2)
	RSPRECHOST(0xa, 0x6)
	RSPRECHOST(0xb, 0x0)
	RSPRECHOST(0x3, 0x8)
	RSPRECHOST(0xd, 0x4)
	RSPRECHOST(0x5, 0x7)
	RSPRECHOST(0xe, 0xf)
	RSPRECHOSTLO(0x9, 0x1)

	RSPRECHOST(0x5, 0xc)
	RSPRECHOST(0xf, 0x1)
	RSPRECHOST(0xd, 0xe)
	RSPRECHOST(0xa, 0x4)
	RSPRECHOST(0x7, 0x0)
	RSPRECHOST(0x3, 0x6)
	RSPRECHOSTHI(0x2, 0x9)
	RSPRECHOST(0xb, 0x8)

	RSPRECHOST(0xb, 0xd)
	RSPRECHOST(0xe, 0x7)
	RSPRECHOST(0x1, 0xc)
	RSPRECHOSTLO(0x9, 0x3)
	RSPRECHOST(0x0, 0x5)
	RSPRECHOST(0x4, 0xf)
	RSPRECHOST(0x6, 0x8)
	RSPRECHOST(0xa, 0x2)

	RSPRECHOST(0xf, 0x6)
	RSPRECHOSTLO(0x9, 0xe)
	RSPRECHOST(0x3, 0xb)
	RSPRECHOST(0x8, 0x0)
	RSPRECHOST(0x2, 0xc)
	RSPRECHOST(0x7, 0xd)
	RSPRECHOST(0x4, 0x1)
	RSPRECHOST(0x5, 0xa)

	RSPRECHOST(0x2, 0xa)
	RSPRECHOST(0x4, 0x8)
	RSPRECHOST(0x6, 0x7)
	RSPRECHOST(0x5, 0x1)
	RSPRECHOST(0xb, 0xf)
	RSPRECHOSTHI(0xe, 0x9)
	RSPRECHOST(0xc, 0x3)
	RSPRECHOST(0x0, 0xd)
	
	RSPRECHOST(0x1, 0x0)
	RSPRECHOST(0x3, 0x2)
	RSPRECHOST(0x5, 0x4)
	RSPRECHOST(0x7, 0x6)
	RSPRECHOSTLO(0x9, 0x8)
	RSPRECHOST(0xb, 0xa)
	RSPRECHOST(0xd, 0xc)
	RSPRECHOST(0xf, 0xe)
	
	RSPRECHOST(0xa, 0xe)
	RSPRECHOST(0x8, 0x4)
	RSPRECHOSTHI(0xf, 0x9)
	RSPRECHOST(0x6, 0xd)
	RSPRECHOST(0xc, 0x1)
	RSPRECHOST(0x2, 0x0)
	RSPRECHOST(0x7, 0xb)
	RSPRECHOST(0x3, 0x5)

	RSPRECHOST(0x8, 0xb)
	RSPRECHOST(0x0, 0xc)
	RSPRECHOST(0x2, 0x5)
	RSPRECHOST(0xd, 0xf)
	RSPRECHOST(0xe, 0xa)
	RSPRECHOST(0x6, 0x3)
	RSPRECHOST(0x1, 0x7)
	RSPRECHOSTHI(0x4, 0x9)

	RSPRECHOSTLO(0x9, 0x7)
	RSPRECHOST(0x1, 0x3)
	RSPRECHOST(0xc, 0xd)
	RSPRECHOST(0xe, 0xb)
	RSPRECHOST(0x6, 0x2)
	RSPRECHOST(0xa, 0x5)
	RSPRECHOST(0x0, 0x4)
	RSPRECHOST(0x8, 0xf)

	RSPRECHOSTHI(0x0, 0x9)
	RSPRECHOST(0x7, 0x5)
	RSPRECHOST(0x4, 0x2)
	RSPRECHOST(0xf, 0xa)
	RSPRECHOST(0x1, 0xe)
	RSPRECHOST(0xc, 0xb)
	RSPRECHOST(0x8, 0x6)
	RSPRECHOST(0xd, 0x3)

	RSPRECHOST(0xc, 0x2)
	RSPRECHOST(0xa, 0x6)
	RSPRECHOST(0xb, 0x0)
	RSPRECHOST(0x3, 0x8)
	RSPRECHOST(0xd, 0x4)
	RSPRECHOST(0x5, 0x7)
	RSPRECHOST(0xe, 0xf)
	RSPRECHOSTLO(0x9, 0x1)

	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(pre, prehost, 224 * 8, 0, cudaMemcpyHostToDevice, gpustream[thr_id]));
}

__host__ void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash)
{
	const uint32_t threadsperblock = 32;
	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	quark_blake512_gpu_hash_64 << <grid, block, 0, gpustream[thr_id] >> >(threads, startNounce, d_nonceVector, (uint2 *)d_outputHash);
}

__host__ void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash)
{

	const uint32_t threadsperblock = 32;
	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	quark_blake512_gpu_hash_80 << <grid, block, 0, gpustream[thr_id] >> >(threads, startNounce, (uint2 *)d_outputHash);
	CUDA_SAFE_CALL(cudaGetLastError());
}
