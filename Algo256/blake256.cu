/**
 * Blake-256 Cuda Kernel (Tested on SM 5.0)
 *
 * Tanguy Pruvot - Nov. 2014
 */

#define PRECALC64 1

#include "miner.h"

extern "C" {
#include "sph/sph_blake.h"
}

#include <cstdint>
#include <memory.h>


/* threads per block and throughput (intensity) */
#define TPB 128

/* added in sph_blake.c */
extern "C" int blake256_rounds = 14;

/* hash by cpu with blake 256 */
void blake256hash(void *output, const void *input, int8_t rounds = 14)
{
	uchar hash[64];
	sph_blake256_context ctx;

	blake256_rounds = rounds;

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 80);
	sph_blake256_close(&ctx, hash);

	memcpy(output, hash, 32);
}

#include "cuda_helper.h"

#if PRECALC64
__constant__ uint32_t _ALIGN(32) d_data[15];
static THREAD uint32_t *h_data;
#else
__constant__ static uint32_t _ALIGN(32) c_data[20];
/* midstate hash cache, this algo is run on 2 parts */
__device__ static uint32_t cache[8];
__device__ static uint32_t prevsum = 0;
/* crc32.c */
extern "C" uint32_t crc32_u32t(const uint32_t *buf, size_t size);
#endif

/* 8 adapters max */
static uint32_t *d_resNonce[MAX_GPUS];
static THREAD uint32_t *h_resNonce;

/* max count of found nonces in one call */
#define NBN 2
static uint32_t extra_results[MAX_GPUS][NBN] = { UINT32_MAX };

#if !PRECALC64
__device__ __constant__
static const uint32_t __align__(32) c_IV256[8] = {
	SPH_C32(0x6A09E667), SPH_C32(0xBB67AE85),
	SPH_C32(0x3C6EF372), SPH_C32(0xA54FF53A),
	SPH_C32(0x510E527F), SPH_C32(0x9B05688C),
	SPH_C32(0x1F83D9AB), SPH_C32(0x5BE0CD19)
};
#endif

#define GSPREC(a,b,c,d,x,y) { \
	v[a] += (m[x] ^ c_u256[y]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x1032); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
	v[a] += (m[y] ^ c_u256[x]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x0321); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
	}

/* Second part (64-80) msg never change, store it */
__device__ static
void blake256_compress(uint32_t *h, const uint32_t *block, const uint32_t T0, const int rounds)
{
	uint32_t /*_ALIGN(8)*/ m[16];
	uint32_t v[16];

	m[0] = block[0];
	m[1] = block[1];
	m[2] = block[2];
	m[3] = block[3];

	const uint32_t c_u256[16] = 
	{
		SPH_C32(0x243F6A88), SPH_C32(0x85A308D3),
		SPH_C32(0x13198A2E), SPH_C32(0x03707344),
		SPH_C32(0xA4093822), SPH_C32(0x299F31D0),
		SPH_C32(0x082EFA98), SPH_C32(0xEC4E6C89),
		SPH_C32(0x452821E6), SPH_C32(0x38D01377),
		SPH_C32(0xBE5466CF), SPH_C32(0x34E90C6C),
		SPH_C32(0xC0AC29B7), SPH_C32(0xC97C50DD),
		SPH_C32(0x3F84D5B5), SPH_C32(0xB5470917)
	};

	 const uint32_t c_Padding[16] = {
		0, 0, 0, 0,
		0x80000000UL, 0, 0, 0,
		0, 0, 0, 0,
		0, 1, 0, 640,
	};


	#pragma unroll
	for (uint32_t i = 4; i < 16; i++) 
	{
#if PRECALC64
		m[i] = c_Padding[i];
#else
		m[i] = (T0 == 0x200) ? block[i] : c_Padding[i];
#endif
	}

#pragma unroll
	for(uint32_t i = 0; i < 8; i++)
		v[i] = h[i];

	v[ 8] = c_u256[0];
	v[ 9] = c_u256[1];
	v[10] = c_u256[2];
	v[11] = c_u256[3];

	v[12] = c_u256[4] ^ T0;
	v[13] = c_u256[5] ^ T0;
	v[14] = c_u256[6];
	v[15] = c_u256[7];

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	GSPREC(0, 4, 0x8, 0xC,0,1);
	GSPREC(1, 5, 0x9, 0xD,2,3);
	GSPREC(2, 6, 0xA, 0xE, 4,5);
	GSPREC(3, 7, 0xB, 0xF, 6,7);
	GSPREC(0, 5, 0xA, 0xF, 8,9);
	GSPREC(1, 6, 0xB, 0xC, 10,11);
	GSPREC(2, 7, 0x8, 0xD, 12,13);
	GSPREC(3, 4, 0x9, 0xE, 14,15);
	//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	GSPREC(0, 4, 0x8, 0xC, 14, 10);
	GSPREC(1, 5, 0x9, 0xD, 4, 8);
	GSPREC(2, 6, 0xA, 0xE, 9, 15);
	GSPREC(3, 7, 0xB, 0xF, 13, 6);
	GSPREC(0, 5, 0xA, 0xF, 1, 12);
	GSPREC(1, 6, 0xB, 0xC, 0, 2);
	GSPREC(2, 7, 0x8, 0xD, 11, 7);
	GSPREC(3, 4, 0x9, 0xE, 5, 3);
	//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	GSPREC(0, 4, 0x8, 0xC, 11, 8);
	GSPREC(1, 5, 0x9, 0xD, 12, 0);
	GSPREC(2, 6, 0xA, 0xE, 5, 2);
	GSPREC(3, 7, 0xB, 0xF, 15, 13);
	GSPREC(0, 5, 0xA, 0xF, 10, 14);
	GSPREC(1, 6, 0xB, 0xC, 3, 6);
	GSPREC(2, 7, 0x8, 0xD, 7, 1);
	GSPREC(3, 4, 0x9, 0xE, 9, 4);
	//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	GSPREC(0, 4, 0x8, 0xC, 7, 9);
	GSPREC(1, 5, 0x9, 0xD, 3, 1);
	GSPREC(2, 6, 0xA, 0xE, 13, 12);
	GSPREC(3, 7, 0xB, 0xF, 11, 14);
	GSPREC(0, 5, 0xA, 0xF, 2, 6);
	GSPREC(1, 6, 0xB, 0xC, 5, 10);
	GSPREC(2, 7, 0x8, 0xD, 4, 0);
	GSPREC(3, 4, 0x9, 0xE, 15, 8);

	//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	GSPREC(0, 4, 0x8, 0xC, 9, 0);
	GSPREC(1, 5, 0x9, 0xD, 5, 7);
	GSPREC(2, 6, 0xA, 0xE, 2, 4);
	GSPREC(3, 7, 0xB, 0xF, 10, 15);
	GSPREC(0, 5, 0xA, 0xF, 14, 1);
	GSPREC(1, 6, 0xB, 0xC, 11, 12);
	GSPREC(2, 7, 0x8, 0xD, 6, 8);
	GSPREC(3, 4, 0x9, 0xE, 3, 13);
	//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	GSPREC(0, 4, 0x8, 0xC, 2, 12);
	GSPREC(1, 5, 0x9, 0xD, 6, 10);
	GSPREC(2, 6, 0xA, 0xE, 0, 11);
	GSPREC(3, 7, 0xB, 0xF, 8, 3);
	GSPREC(0, 5, 0xA, 0xF, 4, 13);
	GSPREC(1, 6, 0xB, 0xC, 7, 5);
	GSPREC(2, 7, 0x8, 0xD, 15, 14);
	GSPREC(3, 4, 0x9, 0xE, 1, 9);

	//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	GSPREC(0, 4, 0x8, 0xC, 12, 5);
	GSPREC(1, 5, 0x9, 0xD, 1, 15);
	GSPREC(2, 6, 0xA, 0xE, 14, 13);
	GSPREC(3, 7, 0xB, 0xF, 4, 10);
	GSPREC(0, 5, 0xA, 0xF, 0, 7);
	GSPREC(1, 6, 0xB, 0xC, 6, 3);
	GSPREC(2, 7, 0x8, 0xD, 9, 2);
	GSPREC(3, 4, 0x9, 0xE, 8, 11);

//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	GSPREC(0, 4, 0x8, 0xC, 13, 11);
	GSPREC(1, 5, 0x9, 0xD, 7, 14);
	GSPREC(2, 6, 0xA, 0xE, 12, 1);
	GSPREC(3, 7, 0xB, 0xF, 3, 9);
	GSPREC(0, 5, 0xA, 0xF, 5, 0);
	GSPREC(1, 6, 0xB, 0xC, 15, 4);
	GSPREC(2, 7, 0x8, 0xD, 8, 6);
	GSPREC(3, 4, 0x9, 0xE, 2, 10);
//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	GSPREC(0, 4, 0x8, 0xC, 6, 15);
	GSPREC(1, 5, 0x9, 0xD, 14, 9);
	GSPREC(2, 6, 0xA, 0xE, 11, 3);
	GSPREC(3, 7, 0xB, 0xF, 0, 8);
	GSPREC(0, 5, 0xA, 0xF, 12, 2);
	GSPREC(1, 6, 0xB, 0xC, 13, 7);
	GSPREC(2, 7, 0x8, 0xD, 1, 4);
	GSPREC(3, 4, 0x9, 0xE, 10, 5);
//	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	GSPREC(0, 4, 0x8, 0xC, 10, 2);
	GSPREC(1, 5, 0x9, 0xD, 8, 4);
	GSPREC(2, 6, 0xA, 0xE, 7, 6);
	GSPREC(3, 7, 0xB, 0xF, 1, 5);
	GSPREC(0, 5, 0xA, 0xF, 15, 11);
	GSPREC(1, 6, 0xB, 0xC, 9, 14);
	GSPREC(2, 7, 0x8, 0xD, 3, 12);
	GSPREC(3, 4, 0x9, 0xE, 13, 0);
//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	GSPREC(0, 4, 0x8, 0xC, 0, 1);
	GSPREC(1, 5, 0x9, 0xD, 2, 3);
	GSPREC(2, 6, 0xA, 0xE, 4, 5);
	GSPREC(3, 7, 0xB, 0xF, 6, 7);
	GSPREC(0, 5, 0xA, 0xF, 8, 9);
	GSPREC(1, 6, 0xB, 0xC, 10, 11);
	GSPREC(2, 7, 0x8, 0xD, 12, 13);
	GSPREC(3, 4, 0x9, 0xE, 14, 15);

//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	GSPREC(0, 4, 0x8, 0xC, 14, 10);
	GSPREC(1, 5, 0x9, 0xD, 4, 8);
	GSPREC(2, 6, 0xA, 0xE, 9, 15);
	GSPREC(3, 7, 0xB, 0xF, 13, 6);
	GSPREC(0, 5, 0xA, 0xF, 1, 12);
	GSPREC(1, 6, 0xB, 0xC, 0, 2);
	GSPREC(2, 7, 0x8, 0xD, 11, 7);
	GSPREC(3, 4, 0x9, 0xE, 5, 3);

//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	GSPREC(0, 4, 0x8, 0xC, 11, 8);
	GSPREC(1, 5, 0x9, 0xD, 12, 0);
	GSPREC(2, 6, 0xA, 0xE, 5, 2);
	GSPREC(3, 7, 0xB, 0xF, 15, 13);
	GSPREC(0, 5, 0xA, 0xF, 10, 14);
	GSPREC(1, 6, 0xB, 0xC, 3, 6);
	GSPREC(2, 7, 0x8, 0xD, 7, 1);
	GSPREC(3, 4, 0x9, 0xE, 9, 4);
	//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	GSPREC(0, 4, 0x8, 0xC, 7, 9);
	GSPREC(1, 5, 0x9, 0xD, 3, 1);
	GSPREC(2, 6, 0xA, 0xE, 13, 12);
	GSPREC(3, 7, 0xB, 0xF, 11, 14);
	GSPREC(0, 5, 0xA, 0xF, 2, 6);
	GSPREC(1, 6, 0xB, 0xC, 5, 10);
	GSPREC(2, 7, 0x8, 0xD, 4, 0);
//	GSPREC(3, 4, 0x9, 0xE, 15, 8);


#if PRECALC64
	// only compute h6 & 7
//	h[6U] ^= v[6U] ^ v[14U];
	h[7] ^= v[7] ^ v[15];
#else
	//#pragma unroll 16
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t j = i & 7U;
		h[j] ^= v[i];
	}
#endif
}


/* Second part (64-80) msg never change, store it */
__device__ static
void blake256_compress_8(uint32_t *const __restrict__ h, const uint32_t *const __restrict__ block)
{
	uint32_t /*_ALIGN(8)*/ m[16];
	uint32_t v[16];

	m[0] = block[0];
	m[1] = block[1];
	m[2] = block[2];
	m[3] = block[3];

	const uint32_t c_u256[16] = 
	{
		SPH_C32(0x243F6A88), SPH_C32(0x85A308D3),
		SPH_C32(0x13198A2E), SPH_C32(0x03707344),
		SPH_C32(0xA4093822), SPH_C32(0x299F31D0),
		SPH_C32(0x082EFA98), SPH_C32(0xEC4E6C89),
		SPH_C32(0x452821E6), SPH_C32(0x38D01377),
		SPH_C32(0xBE5466CF), SPH_C32(0x34E90C6C),
		SPH_C32(0xC0AC29B7), SPH_C32(0xC97C50DD),
		SPH_C32(0x3F84D5B5), SPH_C32(0xB5470917)
	};

	const uint32_t c_Padding[16] = {
		0, 0, 0, 0,
		0x80000000UL, 0, 0, 0,
		0, 0, 0, 0,
		0, 1, 0, 640,
	};


#pragma unroll
	for (int i = 4; i < 16; i++) 
	{
		m[i] = c_Padding[i];
	}

#pragma unroll
	for(int i = 0; i < 8; i++)
		v[i] = h[i];

	v[ 9] = c_u256[1];
	v[10] = c_u256[2];
	v[11] = c_u256[3];

	v[13] = c_u256[5] ^ 640;
	v[14] = c_u256[6];
	v[15] = c_u256[7];

	v[0] = d_data[11];
	v[4] = d_data[12];
	v[8] = d_data[13];
	v[12] = d_data[14];

	GSPREC(1, 5, 0x9, 0xD,2,3);
	GSPREC(2, 6, 0xA, 0xE, 4,5);
	GSPREC(3, 7, 0xB, 0xF, 6,7);
	GSPREC(0, 5, 0xA, 0xF, 8,9);
	GSPREC(1, 6, 0xB, 0xC, 10,11);
	GSPREC(2, 7, 0x8, 0xD, 12,13);
	GSPREC(3, 4, 0x9, 0xE, 14,15);
	//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	GSPREC(0, 4, 0x8, 0xC, 14, 10);
	GSPREC(1, 5, 0x9, 0xD, 4, 8);
	GSPREC(2, 6, 0xA, 0xE, 9, 15);
	GSPREC(3, 7, 0xB, 0xF, 13, 6);
	GSPREC(0, 5, 0xA, 0xF, 1, 12);
	GSPREC(1, 6, 0xB, 0xC, 0, 2);
	GSPREC(2, 7, 0x8, 0xD, 11, 7);
	GSPREC(3, 4, 0x9, 0xE, 5, 3);
	//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	GSPREC(0, 4, 0x8, 0xC, 11, 8);
	GSPREC(1, 5, 0x9, 0xD, 12, 0);
	GSPREC(2, 6, 0xA, 0xE, 5, 2);
	GSPREC(3, 7, 0xB, 0xF, 15, 13);
	GSPREC(0, 5, 0xA, 0xF, 10, 14);
	GSPREC(1, 6, 0xB, 0xC, 3, 6);
	GSPREC(2, 7, 0x8, 0xD, 7, 1);
	GSPREC(3, 4, 0x9, 0xE, 9, 4);
	//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	GSPREC(0, 4, 0x8, 0xC, 7, 9);
	GSPREC(1, 5, 0x9, 0xD, 3, 1);
	GSPREC(2, 6, 0xA, 0xE, 13, 12);
	GSPREC(3, 7, 0xB, 0xF, 11, 14);
	GSPREC(0, 5, 0xA, 0xF, 2, 6);
	GSPREC(1, 6, 0xB, 0xC, 5, 10);
	GSPREC(2, 7, 0x8, 0xD, 4, 0);
	GSPREC(3, 4, 0x9, 0xE, 15, 8);

	//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	GSPREC(0, 4, 0x8, 0xC, 9, 0);
	GSPREC(1, 5, 0x9, 0xD, 5, 7);
	GSPREC(2, 6, 0xA, 0xE, 2, 4);
	GSPREC(3, 7, 0xB, 0xF, 10, 15);
	GSPREC(0, 5, 0xA, 0xF, 14, 1);
	GSPREC(1, 6, 0xB, 0xC, 11, 12);
	GSPREC(2, 7, 0x8, 0xD, 6, 8);
	GSPREC(3, 4, 0x9, 0xE, 3, 13);
	//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	GSPREC(0, 4, 0x8, 0xC, 2, 12);
	GSPREC(1, 5, 0x9, 0xD, 6, 10);
	GSPREC(2, 6, 0xA, 0xE, 0, 11);
	GSPREC(3, 7, 0xB, 0xF, 8, 3);
	GSPREC(0, 5, 0xA, 0xF, 4, 13);
	GSPREC(1, 6, 0xB, 0xC, 7, 5);
	GSPREC(2, 7, 0x8, 0xD, 15, 14);
	GSPREC(3, 4, 0x9, 0xE, 1, 9);

	//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	GSPREC(0, 4, 0x8, 0xC, 12, 5);
	GSPREC(1, 5, 0x9, 0xD, 1, 15);
	GSPREC(2, 6, 0xA, 0xE, 14, 13);
	GSPREC(3, 7, 0xB, 0xF, 4, 10);
	GSPREC(0, 5, 0xA, 0xF, 0, 7);
	GSPREC(1, 6, 0xB, 0xC, 6, 3);
	GSPREC(2, 7, 0x8, 0xD, 9, 2);
	GSPREC(3, 4, 0x9, 0xE, 8, 11);

	//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	GSPREC(0, 4, 0x8, 0xC, 13, 11);
	GSPREC(1, 5, 0x9, 0xD, 7, 14);
	GSPREC(2, 6, 0xA, 0xE, 12, 1);
	GSPREC(3, 7, 0xB, 0xF, 3, 9);
	GSPREC(0, 5, 0xA, 0xF, 5, 0);
	GSPREC(1, 6, 0xB, 0xC, 15, 4);
	GSPREC(2, 7, 0x8, 0xD, 8, 6);
//	GSPREC(3, 4, 0x9, 0xE, 2, 10);
	//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },


#if PRECALC64
	// only compute h6 & 7
//	h[6] ^= v[6] ^ v[14];
	h[7] ^= v[7] ^ v[15];
#else
	//#pragma unroll 16
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t j = i & 7U;
		h[j] ^= v[i];
	}
#endif
}


#if !PRECALC64 /* original method */
__global__
void blake256_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce,
	const uint64_t highTarget, const int crcsum, const int rounds)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint32_t h[8];

		#pragma unroll
		for(int i=0; i<8; i++) {
			h[i] = c_IV256[i];
		}

		if (crcsum != prevsum) {
			prevsum = crcsum;
			blake256_compress(h, c_data, 512, rounds);
			#pragma unroll
			for(int i=0; i<8; i++) {
				cache[i] = h[i];
			}
		} else {
			#pragma unroll
			for(int i=0; i<8; i++) {
				h[i] = cache[i];
			}
		}

		// ------ Close: Bytes 64 to 80 ------

		uint32_t ending[4];
		ending[0] = c_data[16];
		ending[1] = c_data[17];
		ending[2] = c_data[18];
		ending[3] = nonce; /* our tested value */

		blake256_compress(h, ending, 640, rounds);

		// not sure why, h[7] is ok
		h[6] = cuda_swab32(h[6]);

		// compare count of leading zeros h[6] + h[7]
		uint64_t high64 = ((uint64_t*)h)[3];
		if (high64 <= highTarget)
#if NBN == 2
		/* keep the smallest nonce, + extra one if found */
		if (resNonce[0] > nonce) {
			// printf("%llx %llx \n", high64, highTarget);
			resNonce[1] = resNonce[0];
			resNonce[0] = nonce;
		}
		else
			resNonce[1] = nonce;
#else
		resNonce[0] = nonce;
#endif
	}
}

__host__
uint32_t blake256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, const uint64_t highTarget,
	const uint32_t crcsum, const int8_t rounds)
{
	uint32_t result = UINT32_MAX;

	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);
	/* Check error on Ctrl+C or kill to prevent segfaults on exit */
	if (cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
		return result;

	blake256_gpu_hash_80<<<grid, block, 0, gpustream[thr_id]>>>(threads, startNonce, d_resNonce[thr_id], highTarget, crcsum, (int) rounds);
	//cudaDeviceSynchronize();
	if (cudaSuccess == cudaMemcpyAsync(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		//cudaDeviceSynchronize(); /* seems no more required */
		result = h_resNonce[thr_id][0];
		for (int n=0; n < (NBN-1); n++)
			extra_results[thr_id][n] = h_resNonce[thr_id][n+1];
	}
	return result;
}

__host__
void blake256_cpu_setBlock_80(int thr_id, uint32_t *pdata, const uint32_t *ptarget)
{
	uint32_t data[20];
	memcpy(data, pdata, 80);
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(c_data, data, sizeof(data), 0, cudaMemcpyHostToDevice));
}
#else

/* ############################################################################################################################### */
/* Precalculated 1st 64-bytes block (midstate) method */

__global__
void blake256_gpu_hash_16(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce,
                          const uint32_t Target6, const uint32_t Target7, const int rounds, const bool trace)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint32_t _ALIGN(16) h[8];

		#pragma unroll
		for(int i=0; i < 8; i++) {
			h[i] = d_data[i];
		}

		// ------ Close: Bytes 64 to 80 ------

		uint32_t _ALIGN(16) ending[4];
		ending[0] = d_data[8];
		ending[1] = d_data[9];
		ending[2] = d_data[10];
		ending[3] = nonce; /* our tested value */

		blake256_compress(h, ending, 640, rounds);

		if (h[7] <= Target7)
		{
#if NBN == 2
			uint32_t tmp = atomicCAS(resNonce, 0xffffffff, nonce);
			if(tmp != 0xffffffff)
				resNonce[1] = nonce;
#else
			resNonce[0] = nonce;
#endif
#ifdef _DEBUG
			if (trace) {
				uint64_t high64 = ((uint64_t*)h)[3];
				printf("gpu:  %16llx\n", high64);
				printf("gpu: %08x.%08x\n", h[7], h[6]);
				printf("tgt:  %16llx\n", highTarget);
			}
#endif
		}
	}
}


__global__
void blake256_gpu_hash_16_8(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce,
                            const uint32_t Target6, const uint32_t Target7, const int rounds, const bool trace)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint32_t _ALIGN(16) h[8];

#pragma unroll
		for (int i = 0; i < 8; i++)
			h[i] = d_data[i];

		// ------ Close: Bytes 64 to 80 ------

		uint32_t _ALIGN(16) ending[4];
		ending[0] = d_data[8];
		ending[1] = d_data[9];
		ending[2] = d_data[10];
		ending[3] = nonce; /* our tested value */

		blake256_compress_8(h, ending);

		if (h[7] <= Target7)
		{
#if NBN == 2
			uint32_t tmp = atomicCAS(resNonce, 0xffffffff, nonce);
			if(tmp != 0xffffffff)
				resNonce[1] = nonce;
#else
			resNonce[0] = nonce;
#endif
#ifdef _DEBUG
			if (trace) {
				uint64_t high64 = ((uint64_t*)h)[3];
				printf("gpu:  %16llx\n", high64);
				printf("gpu: %08x.%08x\n", h[7], h[6]);
				printf("tgt:  %16llx\n", highTarget);
			}
#endif
		}
	}
}


__host__
static uint32_t blake256_cpu_hash_16(const int thr_id, const uint32_t threads, const uint32_t startNonce, const uint32_t Target6, const uint32_t Target7,
	const int8_t rounds)
{
	uint32_t result = UINT32_MAX;

	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);

	CUDA_SAFE_CALL(cudaMemsetAsync(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t), gpustream[thr_id]));

	if(rounds == 8)
		blake256_gpu_hash_16_8 << <grid, block, 0, gpustream[thr_id] >> > (threads, startNonce, d_resNonce[thr_id], Target6, Target7, (int)rounds, opt_tracegpu);
	else
	{
		if(rounds == 14)
			blake256_gpu_hash_16 << <grid, block, 0, gpustream[thr_id] >> > (threads, startNonce, d_resNonce[thr_id], Target6, Target7, (int)rounds, opt_tracegpu);
		else
			applog(LOG_ERR, "Number of blake rounds not supported");
	}
	CUDA_SAFE_CALL(cudaMemcpyAsync(h_resNonce, d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost, gpustream[thr_id])); cudaStreamSynchronize(gpustream[thr_id]);
	CUDA_SAFE_CALL(cudaStreamSynchronize(gpustream[thr_id]));
	result = h_resNonce[0];

	for (int n=0; n < (NBN-1); n++)
		extra_results[thr_id][n] = h_resNonce[n + 1];
	return result;
}

__host__
static void blake256mid(uint32_t *output, const uint32_t *input, int8_t rounds = 14)
{
	sph_blake256_context ctx;

	/* in sph_blake.c */
	blake256_rounds = rounds;

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 64);

	memcpy(output, (void*)ctx.H, 32);
}

__host__
static void blake256_cpu_setBlock_16(int thr_id, uint32_t *penddata, const uint32_t *midstate, const uint32_t *ptarget)
{
	memcpy(h_data, midstate, 32);
	h_data[8] = penddata[0];
	h_data[9] = penddata[1];
	h_data[10] = penddata[2];

	// precalc v[0], v[4], v[8], v[12]
	h_data[11] = h_data[0] + (h_data[8] ^ 0x85A308D3) + h_data[4];
	h_data[14] = ROTL32(0xA4093822 ^ 640 ^ h_data[11], 16);
	h_data[13] = 0x243F6A88 + h_data[14];
	h_data[12] = ROTR32(h_data[4] ^ h_data[13], 12);
	h_data[11] += (h_data[9] ^ 0x243F6A88) + h_data[12];
	h_data[14] = ROTR32(h_data[14] ^ h_data[11], 8);
	h_data[13] += h_data[14];
	h_data[12] = ROTR32(h_data[12] ^ h_data[13], 7);

	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(d_data, h_data, 15 * 4, 0, cudaMemcpyHostToDevice, gpustream[thr_id]));
}
#endif

extern int scanhash_blake256(int thr_id, uint32_t *pdata, uint32_t *ptarget,
	uint32_t max_nonce, uint32_t *hashes_done, int8_t blakerounds=14)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t _ALIGN(64) endiandata[20];
#if PRECALC64
	uint32_t _ALIGN(64) midstate[8];
#else
	uint32_t crcsum;
#endif
	unsigned int intensity = 28;
	uint32_t throughputmax = device_intensity(device_map[thr_id], __func__, 1U << intensity);
	uint32_t throughput = min(throughputmax, max_nonce - first_nonce) & 0xfffffc00;

	int rc = 0;

	if (opt_benchmark)
	{
		ptarget[7] = 0x00000000;
		ptarget[6] = 0xffffffff;
	}
	uint32_t target6 = ptarget[6];
	uint32_t target7 = swab32(ptarget[7]); // don't ask me why

	if (opt_tracegpu)
	{
		/* test call from util.c */
		throughput = 1;
		for (int k = 0; k < 20; k++)
			pdata[k] = swab32(pdata[k]);
	}

	static THREAD volatile bool init = false;
	if(!init)
	{
		if(throughputmax == intensity)
			applog(LOG_INFO, "GPU #%d: using default intensity %.3f", device_map[thr_id], throughput2intensity(throughputmax));
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		CUDA_SAFE_CALL(cudaDeviceReset());
		CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaschedule));
		CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));
		CUDA_SAFE_CALL(cudaMallocHost(&h_data, 15 * sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMallocHost(&h_resNonce, NBN * sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		mining_has_stopped[thr_id] = false;
		init = true;
	}

#if PRECALC64
	for (int k = 0; k < 16; k++)
		be32enc(&endiandata[k], pdata[k]);
	blake256mid(midstate, endiandata, blakerounds);
	blake256_cpu_setBlock_16(thr_id, &pdata[16], midstate, ptarget);
#else
	blake256_cpu_setBlock_80(thr_id, pdata, ptarget);
	crcsum = crc32_u32t(pdata, 64);
#endif /* PRECALC64 */

	do {
#if PRECALC64
		// GPU HASH (second block only, first is midstate)
		uint32_t foundNonce =	blake256_cpu_hash_16(thr_id, throughput, pdata[19], target6, target7, blakerounds);
#else
		// GPU FULL HASH
		uint32_t foundNonce =	blake256_cpu_hash_80(thr_id, throughput, pdata[19], targetHigh, crcsum, blakerounds);
#endif
		if(stop_mining) {mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);}
		if(foundNonce != UINT32_MAX)
		{
			uint32_t vhashcpu[8] = { 0 };

			for (int k=0; k < 19; k++)
				be32enc(&endiandata[k], pdata[k]);

			if(opt_verify)
			{
				be32enc(&endiandata[19], foundNonce);
				blake256hash(vhashcpu, endiandata, blakerounds);
			}
			//applog(LOG_BLUE, "%08x %16llx", vhashcpu[6], targetHigh);
			if (vhashcpu[7] <= target7 && fulltest(vhashcpu, ptarget))
			{
				if (opt_benchmark) applog(LOG_INFO, "GPU #%d Found nounce %08x", thr_id, foundNonce);
				rc = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				pdata[19] = foundNonce;
#if NBN > 1
				if (extra_results[thr_id][0] != UINT32_MAX)
				{
					if(opt_verify)
					{
						be32enc(&endiandata[19], extra_results[thr_id][0]);
						blake256hash(vhashcpu, endiandata, blakerounds);
					}
					if (vhashcpu[7] <= target7 && fulltest(vhashcpu, ptarget))
					{
						pdata[21] = extra_results[thr_id][0];
						if(opt_benchmark) applog(LOG_INFO, "GPU #%d Found second nounce %08x", thr_id, extra_results[thr_id][0]);
//						applog(LOG_BLUE, "1:%x 2:%x", foundNonce, extra_results[thr_id][0]);
						rc = 2;
					}
					else
					{
						if(vhashcpu[7]>target7)
							applog(LOG_ERR, "GPU #%d: result for second nonce %08x does not validate on CPU!", device_map[thr_id], extra_results[thr_id][0]);
					}
					extra_results[thr_id][0] = UINT32_MAX;
				}
#endif
				//applog_hash((uint8_t*)ptarget);
				//applog_compare_hash((uint8_t*)vhashcpu,(uint8_t*)ptarget);
				return rc;
			}
			else
			{
				if(opt_debug)
				{
					applog_hash((uchar*)ptarget);
					applog_compare_hash((uchar*)vhashcpu, (uchar*)ptarget);
				}
				if(vhashcpu[7]>target7)
					applog(LOG_ERR, "GPU #%d: result for nonce %08x does not validate on CPU!", device_map[thr_id], foundNonce);
			}
		}

		pdata[19] += throughput; CUDA_SAFE_CALL(cudaGetLastError());
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce ;

	return rc;
}
