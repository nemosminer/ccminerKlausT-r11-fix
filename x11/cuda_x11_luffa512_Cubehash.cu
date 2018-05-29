/*
* luffa_for_32.c
* Version 2.0 (Sep 15th 2009)
*
* Copyright (C) 2008-2009 Hitachi, Ltd. All rights reserved.
*
* Hitachi, Ltd. is the owner of this software and hereby grant
* the U.S. Government and any interested party the right to use
* this software for the purposes of the SHA-3 evaluation process,
* notwithstanding that this software is copyrighted.
*
* THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
* WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
* ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
* WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
* ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
* OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/

#include "cuda_helper.h"

typedef unsigned char BitSequence;

typedef struct
{
	uint32_t buffer[8]; /* Buffer to be hashed */
	uint32_t chainv[40];   /* Chaining values */
} hashState;

#define MULT2(a,j)\
    tmp = a[7+(8*j)];\
    a[7+(8*j)] = a[6+(8*j)];\
    a[6+(8*j)] = a[5+(8*j)];\
    a[5+(8*j)] = a[4+(8*j)];\
    a[4+(8*j)] = a[3+(8*j)] ^ tmp;\
    a[3+(8*j)] = a[2+(8*j)] ^ tmp;\
    a[2+(8*j)] = a[1+(8*j)];\
    a[1+(8*j)] = a[0+(8*j)] ^ tmp;\
    a[0+(8*j)] = tmp;

#define LROT ROTL32

#define CUBEHASH_ROUNDS 16 /* this is r for CubeHashr/b */
#define CUBEHASH_BLOCKBYTES 32 /* this is b for CubeHashr/b */

#define ROTATEUPWARDS7(a)  LROT(a,7)
#define ROTATEUPWARDS11(a) LROT(a,11)

#define TWEAK(a0,a1,a2,a3,j)\
    a0 = LROT(a0,j);\
    a1 = LROT(a1,j);\
    a2 = LROT(a2,j);\
    a3 = LROT(a3,j);

#define SUBCRUMB(a0,a1,a2,a3,a4)\
    a4  = a0;\
    a0 |= a1;\
    a2 ^= a3;\
    a1  = ~a1;\
    a0 ^= a3;\
    a3 &= a4;\
    a1 ^= a3;\
    a3 ^= a2;\
    a2 &= a0;\
    a0  = ~a0;\
    a2 ^= a1;\
    a1 |= a3;\
    a4 ^= a1;\
    a3 ^= a2;\
    a2 &= a1;\
    a1 ^= a0;\
    a0  = a4;

#define MIXWORD(a0,a4)\
    a4 ^= a0;\
    a0  = LROT(a0,2);\
    a0 ^= a4;\
    a4  = LROT(a4,14);\
    a4 ^= a0;\
    a0  = LROT(a0,10);\
    a0 ^= a4;\
    a4  = LROT(a4,1);

#define ADD_CONSTANT(a0,b0,c0,c1)\
    a0 ^= c0;\
    b0 ^= c1;

#define STEP(c0,c1)\
    SUBCRUMB(chainv[0],chainv[1],chainv[2],chainv[3],tmp);\
    SUBCRUMB(chainv[5],chainv[6],chainv[7],chainv[4],tmp);\
    MIXWORD(chainv[0],chainv[4]);\
    MIXWORD(chainv[1],chainv[5]);\
    MIXWORD(chainv[2],chainv[6]);\
    MIXWORD(chainv[3],chainv[7]);\
    ADD_CONSTANT(chainv[0],chainv[4],c0,c1);

// Precalculated chaining values
__device__ __constant__ uint32_t c_IV[40] =
{ 0x8bb0a761, 0xc2e4aa8b, 0x2d539bc9, 0x381408f8,
0x478f6633, 0x255a46ff, 0x581c37f7, 0x601c2e8e,
0x266c5f9d, 0xc34715d8, 0x8900670e, 0x51a540be,
0xe4ce69fb, 0x5089f4d4, 0x3cc0a506, 0x609bcb02,
0xa4e3cd82, 0xd24fd6ca, 0xc0f196dc, 0xcf41eafe,
0x0ff2e673, 0x303804f2, 0xa7b3cd48, 0x677addd4,
0x66e66a8a, 0x2303208f, 0x486dafb4, 0xc0d37dc6,
0x634d15af, 0xe5af6747, 0x10af7e38, 0xee7e6428,
0x01262e5d, 0xc92c2e64, 0x82fee966, 0xcea738d3,
0x867de2b0, 0xe0714818, 0xda6e831f, 0xa7062529 };



/* old chaining values
__device__ __constant__ uint32_t c_IV[40] = {
0x6d251e69,0x44b051e0,0x4eaa6fb4,0xdbf78465,
0x6e292011,0x90152df4,0xee058139,0xdef610bb,
0xc3b44b95,0xd9d2f256,0x70eee9a0,0xde099fa3,
0x5d9b0557,0x8fc944b3,0xcf1ccf0e,0x746cd581,
0xf7efc89d,0x5dba5781,0x04016ce5,0xad659c05,
0x0306194f,0x666d1836,0x24aa230a,0x8b264ae7,
0x858075d5,0x36d79cce,0xe571f7d7,0x204b1f67,
0x35870c6a,0x57e9e923,0x14bcb808,0x7cde72ce,
0x6c68e9be,0x5ec41e22,0xc825b7c7,0xaffb4363,
0xf5df3999,0x0fc688f1,0xb07224cc,0x03e86cea};
*/

__device__ __constant__ uint32_t c_CNS[80] = {
	0x303994a6, 0xe0337818, 0xc0e65299, 0x441ba90d,
	0x6cc33a12, 0x7f34d442, 0xdc56983e, 0x9389217f,
	0x1e00108f, 0xe5a8bce6, 0x7800423d, 0x5274baf4,
	0x8f5b7882, 0x26889ba7, 0x96e1db12, 0x9a226e9d,
	0xb6de10ed, 0x01685f3d, 0x70f47aae, 0x05a17cf4,
	0x0707a3d4, 0xbd09caca, 0x1c1e8f51, 0xf4272b28,
	0x707a3d45, 0x144ae5cc, 0xaeb28562, 0xfaa7ae2b,
	0xbaca1589, 0x2e48f1c1, 0x40a46f3e, 0xb923c704,
	0xfc20d9d2, 0xe25e72c1, 0x34552e25, 0xe623bb72,
	0x7ad8818f, 0x5c58a4a4, 0x8438764a, 0x1e38e2e7,
	0xbb6de032, 0x78e38b9d, 0xedb780c8, 0x27586719,
	0xd9847356, 0x36eda57f, 0xa2c78434, 0x703aace7,
	0xb213afa5, 0xe028c9bf, 0xc84ebe95, 0x44756f91,
	0x4e608a22, 0x7e8fce32, 0x56d858fe, 0x956548be,
	0x343b138f, 0xfe191be2, 0xd0ec4e3d, 0x3cb226e5,
	0x2ceb4882, 0x5944a28e, 0xb3ad2208, 0xa1c4c355,
	0xf0d2e9e3, 0x5090d577, 0xac11d7fa, 0x2d1925ab,
	0x1bcb66f2, 0xb46496ac, 0x6f2d9bc9, 0xd1925ab0,
	0x78602649, 0x29131ab6, 0x8edae952, 0x0fc053c3,
	0x3b6ba548, 0x3f014f0c, 0xedae9520, 0xfc053c31 };


/***************************************************/
static __device__ __forceinline__
void rnd512(uint32_t *const __restrict__ statebuffer, uint32_t *const __restrict__ statechainv)
{
	int i, j;
	uint32_t t[40];
	uint32_t chainv[8];
	uint32_t tmp;

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		t[i] = 0;
#pragma unroll 5
		for(j = 0; j<5; j++)
		{
			t[i] ^= statechainv[i + 8 * j];
		}
	}

	MULT2(t, 0);

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
#pragma unroll 8
		for(i = 0; i<8; i++)
		{
			statechainv[i + 8 * j] ^= t[i];
		}
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
#pragma unroll 8
		for(i = 0; i<8; i++)
		{
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
		MULT2(statechainv, j);
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
#pragma unroll 8
		for(i = 0; i<8; i++)
		{
			statechainv[8 * j + i] ^= t[8 * ((j + 1) % 5) + i];
		}
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
#pragma unroll 8
		for(i = 0; i<8; i++)
		{
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
		MULT2(statechainv, j);
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
#pragma unroll 8
		for(i = 0; i<8; i++)
		{
			statechainv[8 * j + i] ^= t[8 * ((j + 4) % 5) + i];
		}
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
#pragma unroll 8
		for(i = 0; i<8; i++)
		{
			statechainv[i + 8 * j] ^= statebuffer[i];
		}
		MULT2(statebuffer, 0);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		chainv[i] = statechainv[i];
	}

#pragma unroll 1
	for(i = 0; i<=14; i+=2)
	{
		STEP(c_CNS[i], c_CNS[i + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		statechainv[i] = chainv[i];
		chainv[i] = statechainv[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);

#pragma unroll 1
	for(i = 0; i<=14; i+=2)
	{
		STEP(c_CNS[i + 16], c_CNS[i + 16 + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		statechainv[i + 8] = chainv[i];
		chainv[i] = statechainv[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

#pragma unroll 1
	for(i = 0; i<=14; i+=2)
	{
		STEP(c_CNS[i + 32], c_CNS[i + 32 + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		statechainv[i + 16] = chainv[i];
		chainv[i] = statechainv[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

#pragma unroll 1
	for(i = 0; i<=14; i+=2)
	{
		STEP(c_CNS[i + 48], c_CNS[i + 48 + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		statechainv[i + 24] = chainv[i];
		chainv[i] = statechainv[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

#pragma unroll 1
	for(i = 0; i<=14; i+=2)
	{
		STEP(c_CNS[i + 64], c_CNS[i + 64 + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		statechainv[i + 32] = chainv[i];
	}
}


static __device__ __forceinline__
void rnd512_finalfirst(uint32_t *const statechainv)
{
	int i, j;
	uint32_t t[40];
	uint32_t chainv[8];
	uint32_t tmp;

#pragma unroll 8
	for (i = 0; i<8; i++)
	{
		t[i] = 0;
#pragma unroll 5
		for (j = 0; j<5; j++)
		{
			t[i] ^= statechainv[i + 8 * j];
		}
	}

	MULT2(t, 0);

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			statechainv[i + 8 * j] ^= t[i];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
		MULT2(statechainv, j);
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			statechainv[8 * j + i] ^= t[8 * ((j + 1) % 5) + i];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
		MULT2(statechainv, j);
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			statechainv[8 * j + i] ^= t[8 * ((j + 4) % 5) + i];
		}
	}

	statechainv[0 + 8 * 0] ^= 0x80000000;
	statechainv[1 + 8 * 1] ^= 0x80000000;
	statechainv[2 + 8 * 2] ^= 0x80000000;
	statechainv[3 + 8 * 3] ^= 0x80000000;
	statechainv[4 + 8 * 4] ^= 0x80000000;


#pragma unroll 8
	for (i = 0; i<8; i++) {
		chainv[i] = statechainv[i];
	}

#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i)], c_CNS[(2 * i) + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		statechainv[i] = chainv[i];
		chainv[i] = statechainv[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);

#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 16], c_CNS[(2 * i) + 16 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		statechainv[i + 8] = chainv[i];
		chainv[i] = statechainv[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 32], c_CNS[(2 * i) + 32 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		statechainv[i + 16] = chainv[i];
		chainv[i] = statechainv[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 48], c_CNS[(2 * i) + 48 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		statechainv[i + 24] = chainv[i];
		chainv[i] = statechainv[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 64], c_CNS[(2 * i) + 64 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		statechainv[i + 32] = chainv[i];
	}
}


static __device__ __forceinline__
void rnd512_first(uint32_t state[40], uint32_t buffer[8])
{
	int i, j;
	uint32_t chainv[8];
	uint32_t tmp;

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
		state[0 + 8 * j] ^= buffer[0];

#pragma unroll 7
		for(i = 1; i<8; i++)
		{
			state[i + 8 * j] ^= buffer[i];
		}
		MULT2(buffer, 0);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		chainv[i] = state[i];
	}

#pragma unroll 1
	for(i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i)], c_CNS[(2 * i) + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		state[i] = chainv[i];
		chainv[i] = state[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);

#pragma unroll 1
	for(i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 16], c_CNS[(2 * i) + 16 + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		state[i + 8] = chainv[i];
		chainv[i] = state[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

#pragma unroll 1
	for(i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 32], c_CNS[(2 * i) + 32 + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		state[i + 16] = chainv[i];
		chainv[i] = state[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

#pragma unroll 1
	for(i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 48], c_CNS[(2 * i) + 48 + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		state[i + 24] = chainv[i];
		chainv[i] = state[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

#pragma unroll 1
	for(i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 64], c_CNS[(2 * i) + 64 + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		state[i + 32] = chainv[i];
	}
}

/***************************************************/
static __device__ __forceinline__
void rnd512_nullhash(uint32_t *const state)
{
	int i, j;
	uint32_t t[40];
	uint32_t chainv[8];
	uint32_t tmp;

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		t[i] = state[i + 8 * 0];
#pragma unroll 4
		for(j = 1; j<5; j++)
		{
			t[i] ^= state[i + 8 * j];
		}
	}

	MULT2(t, 0);

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
#pragma unroll 8
		for(i = 0; i<8; i++)
		{
			state[i + 8 * j] ^= t[i];
		}
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
#pragma unroll 8
		for(i = 0; i<8; i++)
		{
			t[i + 8 * j] = state[i + 8 * j];
		}
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
		MULT2(state, j);
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
#pragma unroll 8
		for(i = 0; i<8; i++)
		{
			state[8 * j + i] ^= t[8 * ((j + 1) % 5) + i];
		}
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
#pragma unroll 8
		for(i = 0; i<8; i++)
		{
			t[i + 8 * j] = state[i + 8 * j];
		}
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
		MULT2(state, j);
	}

#pragma unroll 5
	for(j = 0; j<5; j++)
	{
#pragma unroll 8
		for(i = 0; i<8; i++)
		{
			state[8 * j + i] ^= t[8 * ((j + 4) % 5) + i];
		}
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		chainv[i] = state[i];
	}

#pragma unroll 1
	for(i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i)], c_CNS[(2 * i) + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		state[i] = chainv[i];
		chainv[i] = state[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);

#pragma unroll 1
	for(i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 16], c_CNS[(2 * i) + 16 + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		state[i + 8] = chainv[i];
		chainv[i] = state[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

#pragma unroll 1
	for(i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 32], c_CNS[(2 * i) + 32 + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		state[i + 16] = chainv[i];
		chainv[i] = state[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

#pragma unroll 1
	for(i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 48], c_CNS[(2 * i) + 48 + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		state[i + 24] = chainv[i];
		chainv[i] = state[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

#pragma unroll 1
	for(i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 64], c_CNS[(2 * i) + 64 + 1]);
	}

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		state[i + 32] = chainv[i];
	}
}
static __device__ __forceinline__
void Update512(uint32_t *const __restrict__ statebuffer, uint32_t *const __restrict__ statechainv, const uint32_t *const __restrict__ data)
{
#pragma unroll 8
	for(int i = 0; i < 8; i++)
		statebuffer[i] = cuda_swab32(data[i]);
	rnd512_first(statechainv, statebuffer);

#pragma unroll 8
	for(int i = 0; i < 8; i++)
		statebuffer[i] = cuda_swab32(data[i + 8]);
	rnd512(statebuffer, statechainv);
}


/***************************************************/
static __device__ __forceinline__
void finalization512(uint32_t *const __restrict__ statechainv, uint32_t *const __restrict__ b)
{
	int i, j;
	rnd512_finalfirst(statechainv);
	/*---- blank round with m=0 ----*/
	rnd512_nullhash(statechainv);

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		b[i] = statechainv[i + 8 * 0];
#pragma unroll 4
		for(j = 1; j<5; j++)
		{
			b[i] ^= statechainv[i + 8 * j];
		}
		b[i] = cuda_swab32((b[i]));
	}

	rnd512_nullhash(statechainv);

#pragma unroll 8
	for(i = 0; i<8; i++)
	{
		b[8 + i] = statechainv[i + 8 * 0];
#pragma unroll 4
		for(j = 1; j<5; j++)
		{
			b[8 + i] ^= statechainv[i + 8 * j];
		}
		b[8 + i] = cuda_swab32((b[8 + i]));
	}
}

#define ROUND_EVEN   \
		xg = (x0 + xg); \
		x0 = ROTL32(x0, 7); \
		xh = (x1 + xh); \
		x1 = ROTL32(x1, 7); \
		xi = (x2 + xi); \
		x2 = ROTL32(x2, 7); \
		xj = (x3 + xj); \
		x3 = ROTL32(x3, 7); \
		xk = (x4 + xk); \
		x4 = ROTL32(x4, 7); \
		xl = (x5 + xl); \
		x5 = ROTL32(x5, 7); \
		xm = (x6 + xm); \
		x6 = ROTL32(x6, 7); \
		xn = (x7 + xn); \
		x7 = ROTL32(x7, 7); \
		xo = (x8 + xo); \
		x8 = ROTL32(x8, 7); \
		xp = (x9 + xp); \
		x9 = ROTL32(x9, 7); \
		xq = (xa + xq); \
		xa = ROTL32(xa, 7); \
		xr = (xb + xr); \
		xb = ROTL32(xb, 7); \
		xs = (xc + xs); \
		xc = ROTL32(xc, 7); \
		xt = (xd + xt); \
		xd = ROTL32(xd, 7); \
		xu = (xe + xu); \
		xe = ROTL32(xe, 7); \
		xv = (xf + xv); \
		xf = ROTL32(xf, 7); \
		x8 ^= xg; \
		x9 ^= xh; \
		xa ^= xi; \
		xb ^= xj; \
		xc ^= xk; \
		xd ^= xl; \
		xe ^= xm; \
		xf ^= xn; \
		x0 ^= xo; \
		x1 ^= xp; \
		x2 ^= xq; \
		x3 ^= xr; \
		x4 ^= xs; \
		x5 ^= xt; \
		x6 ^= xu; \
		x7 ^= xv; \
		xi = (x8 + xi); \
		x8 = ROTL32(x8, 11); \
		xj = (x9 + xj); \
		x9 = ROTL32(x9, 11); \
		xg = (xa + xg); \
		xa = ROTL32(xa, 11); \
		xh = (xb + xh); \
		xb = ROTL32(xb, 11); \
		xm = (xc + xm); \
		xc = ROTL32(xc, 11); \
		xn = (xd + xn); \
		xd = ROTL32(xd, 11); \
		xk = (xe + xk); \
		xe = ROTL32(xe, 11); \
		xl = (xf + xl); \
		xf = ROTL32(xf, 11); \
		xq = (x0 + xq); \
		x0 = ROTL32(x0, 11); \
		xr = (x1 + xr); \
		x1 = ROTL32(x1, 11); \
		xo = (x2 + xo); \
		x2 = ROTL32(x2, 11); \
		xp = (x3 + xp); \
		x3 = ROTL32(x3, 11); \
		xu = (x4 + xu); \
		x4 = ROTL32(x4, 11); \
		xv = (x5 + xv); \
		x5 = ROTL32(x5, 11); \
		xs = (x6 + xs); \
		x6 = ROTL32(x6, 11); \
		xt = (x7 + xt); \
		x7 = ROTL32(x7, 11); \
		xc ^= xi; \
		xd ^= xj; \
		xe ^= xg; \
		xf ^= xh; \
		x8 ^= xm; \
		x9 ^= xn; \
		xa ^= xk; \
		xb ^= xl; \
		x4 ^= xq; \
		x5 ^= xr; \
		x6 ^= xo; \
		x7 ^= xp; \
		x0 ^= xu; \
		x1 ^= xv; \
		x2 ^= xs; \
		x3 ^= xt; 

#define ROUND_ODD    \
		xj = (xc + xj); \
		xc = ROTL32(xc, 7); \
		xi = (xd + xi); \
		xd = ROTL32(xd, 7); \
		xh = (xe + xh); \
		xe = ROTL32(xe, 7); \
		xg = (xf + xg); \
		xf = ROTL32(xf, 7); \
		xn = (x8 + xn); \
		x8 = ROTL32(x8, 7); \
		xm = (x9 + xm); \
		x9 = ROTL32(x9, 7); \
		xl = (xa + xl); \
		xa = ROTL32(xa, 7); \
		xk = (xb + xk); \
		xb = ROTL32(xb, 7); \
		xr = (x4 + xr); \
		x4 = ROTL32(x4, 7); \
		xq = (x5 + xq); \
		x5 = ROTL32(x5, 7); \
		xp = (x6 + xp); \
		x6 = ROTL32(x6, 7); \
		xo = (x7 + xo); \
		x7 = ROTL32(x7, 7); \
		xv = (x0 + xv); \
		x0 = ROTL32(x0, 7); \
		xu = (x1 + xu); \
		x1 = ROTL32(x1, 7); \
		xt = (x2 + xt); \
		x2 = ROTL32(x2, 7); \
		xs = (x3 + xs); \
		x3 = ROTL32(x3, 7); \
		x4 ^= xj; \
		x5 ^= xi; \
		x6 ^= xh; \
		x7 ^= xg; \
		x0 ^= xn; \
		x1 ^= xm; \
		x2 ^= xl; \
		x3 ^= xk; \
		xc ^= xr; \
		xd ^= xq; \
		xe ^= xp; \
		xf ^= xo; \
		x8 ^= xv; \
		x9 ^= xu; \
		xa ^= xt; \
		xb ^= xs; \
		xh = (x4 + xh); \
		x4 = ROTL32(x4, 11); \
		xg = (x5 + xg); \
		x5 = ROTL32(x5, 11); \
		xj = (x6 + xj); \
		x6 = ROTL32(x6, 11); \
		xi = (x7 + xi); \
		x7 = ROTL32(x7, 11); \
		xl = (x0 + xl); \
		x0 = ROTL32(x0, 11); \
		xk = (x1 + xk); \
		x1 = ROTL32(x1, 11); \
		xn = (x2 + xn); \
		x2 = ROTL32(x2, 11); \
		xm = (x3 + xm); \
		x3 = ROTL32(x3, 11); \
		xp = (xc + xp); \
		xc = ROTL32(xc, 11); \
		xo = (xd + xo); \
		xd = ROTL32(xd, 11); \
		xr = (xe + xr); \
		xe = ROTL32(xe, 11); \
		xq = (xf + xq); \
		xf = ROTL32(xf, 11); \
		xt = (x8 + xt); \
		x8 = ROTL32(x8, 11); \
		xs = (x9 + xs); \
		x9 = ROTL32(x9, 11); \
		xv = (xa + xv); \
		xa = ROTL32(xa, 11); \
		xu = (xb + xu); \
		xb = ROTL32(xb, 11); \
		x0 ^= xh; \
		x1 ^= xg; \
		x2 ^= xj; \
		x3 ^= xi; \
		x4 ^= xl; \
		x5 ^= xk; \
		x6 ^= xn; \
		x7 ^= xm; \
		x8 ^= xp; \
		x9 ^= xo; \
		xa ^= xr; \
		xb ^= xq; \
		xc ^= xt; \
		xd ^= xs; \
		xe ^= xv; \
		xf ^= xu; 

#define SIXTEEN_ROUNDS \
		for (int j = 0; j < 8; j ++) { \
			ROUND_EVEN; \
			ROUND_ODD;}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(256, 4)
#else
__launch_bounds__(256, 3)
#endif
void x11_luffaCubehash512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *const g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if(thread < threads)
	{
		uint32_t *const Hash = (uint32_t*)&g_hash[8 * thread];

		uint32_t statebuffer[8];
		uint32_t statechainv[40] =
		{
			0x8bb0a761, 0xc2e4aa8b, 0x2d539bc9, 0x381408f8,
			0x478f6633, 0x255a46ff, 0x581c37f7, 0x601c2e8e,
			0x266c5f9d, 0xc34715d8, 0x8900670e, 0x51a540be,
			0xe4ce69fb, 0x5089f4d4, 0x3cc0a506, 0x609bcb02,
			0xa4e3cd82, 0xd24fd6ca, 0xc0f196dc, 0xcf41eafe,
			0x0ff2e673, 0x303804f2, 0xa7b3cd48, 0x677addd4,
			0x66e66a8a, 0x2303208f, 0x486dafb4, 0xc0d37dc6,
			0x634d15af, 0xe5af6747, 0x10af7e38, 0xee7e6428,
			0x01262e5d, 0xc92c2e64, 0x82fee966, 0xcea738d3,
			0x867de2b0, 0xe0714818, 0xda6e831f, 0xa7062529
		};

		Update512(statebuffer, statechainv, Hash);
		finalization512(statechainv, Hash);
		//Cubehash

		uint32_t x0 = 0x2AEA2A61 ^ Hash[0];
		uint32_t x1 = 0x50F494D4 ^ Hash[1];
		uint32_t x2 = 0x2D538B8B ^ Hash[2];
		uint32_t x3 = 0x4167D83E ^ Hash[3];
		uint32_t x4 = 0x3FEE2313 ^ Hash[4];
		uint32_t x5 = 0xC701CF8C ^ Hash[5];
		uint32_t x6 = 0xCC39968E ^ Hash[6];
		uint32_t x7 = 0x50AC5695 ^ Hash[7];
		uint32_t x8 = 0x4D42C787, x9 = 0xA647A8B3, xa = 0x97CF0BEF, xb = 0x825B4537;
		uint32_t xc = 0xEEF864D2, xd = 0xF22090C4, xe = 0xD0E5CD33, xf = 0xA23911AE;
		uint32_t xg = 0xFCD398D9 + x0, xh = 0x148FE485 + x1, xi = 0x1B017BEF + x2, xj = 0xB6444532 + x3;
		uint32_t xk = 0x6A536159 + x4, xl = 0x2FF5781C + x5, xm = 0x91FA7934 + x6, xn = 0x0DBADEA9 + x7;
		uint32_t xo = 0xD65C8A2B + x8, xp = 0xA5A70E75 + x9, xq = 0xB1C62456 + xa, xr = 0xBC796576 + xb;
		uint32_t xs = 0x1921C8F7 + xc, xt = 0xE7989AF1 + xd, xu = 0x7795D246 + xe, xv = 0xD43E3B44 + xf;


			x0 = ROTL32(x0, 7);
			x1 = ROTL32(x1, 7);
			x2 = ROTL32(x2, 7);
			x3 = ROTL32(x3, 7); 
			x4 = ROTL32(x4, 7);
			x5 = ROTL32(x5, 7);
			x6 = ROTL32(x6, 7);
			x7 = ROTL32(x7, 7);
			x8 = ROTL32(x8, 7);
			x9 = ROTL32(x9, 7);
			xa = ROTL32(xa, 7);
			xb = ROTL32(xb, 7);
			xc = ROTL32(xc, 7);
			xd = ROTL32(xd, 7);
			xe = ROTL32(xe, 7);
			xf = ROTL32(xf, 7);
			x8 ^= xg;
			x9 ^= xh;
			xa ^= xi;
			xb ^= xj;
			xc ^= xk;
			xd ^= xl;
			xe ^= xm;
			xf ^= xn;
			x0 ^= xo;
			x1 ^= xp;
			x2 ^= xq;
			x3 ^= xr;
			x4 ^= xs;
			x5 ^= xt;
			x6 ^= xu;
			x7 ^= xv;
			xi = (x8 + xi);
			x8 = ROTL32(x8, 11);
			xj = (x9 + xj);
			x9 = ROTL32(x9, 11);
			xg = (xa + xg);
			xa = ROTL32(xa, 11);
			xh = (xb + xh);
			xb = ROTL32(xb, 11);
			xm = (xc + xm);
			xc = ROTL32(xc, 11);
			xn = (xd + xn);
			xd = ROTL32(xd, 11);
			xk = (xe + xk);
			xe = ROTL32(xe, 11);
			xl = (xf + xl);
			xf = ROTL32(xf, 11);
			xq = (x0 + xq);
			x0 = ROTL32(x0, 11);
			xr = (x1 + xr);
			x1 = ROTL32(x1, 11);
			xo = (x2 + xo);
			x2 = ROTL32(x2, 11);
			xp = (x3 + xp);
			x3 = ROTL32(x3, 11);
			xu = (x4 + xu);
			x4 = ROTL32(x4, 11);
			xv = (x5 + xv);
			x5 = ROTL32(x5, 11);
			xs = (x6 + xs);
			x6 = ROTL32(x6, 11);
			xt = (x7 + xt);
			x7 = ROTL32(x7, 11);
			xc ^= xi;
			xd ^= xj;
			xe ^= xg;
			xf ^= xh;
			x8 ^= xm;
			x9 ^= xn;
			xa ^= xk;
			xb ^= xl;
			x4 ^= xq;
			x5 ^= xr;
			x6 ^= xo;
			x7 ^= xp;
			x0 ^= xu;
			x1 ^= xv;
			x2 ^= xs;
			x3 ^= xt;

		xj = (xc + xj);
			xc = ROTL32(xc, 7);
			xi = (xd + xi);
			xd = ROTL32(xd, 7);
			xh = (xe + xh);
			xe = ROTL32(xe, 7);
			xg = (xf + xg);
			xf = ROTL32(xf, 7);
			xn = (x8 + xn);
			x8 = ROTL32(x8, 7);
			xm = (x9 + xm);
			x9 = ROTL32(x9, 7);
			xl = (xa + xl);
			xa = ROTL32(xa, 7);
			xk = (xb + xk);
			xb = ROTL32(xb, 7);
			xr = (x4 + xr);
			x4 = ROTL32(x4, 7);
			xq = (x5 + xq);
			x5 = ROTL32(x5, 7);
			xp = (x6 + xp);
			x6 = ROTL32(x6, 7);
			xo = (x7 + xo);
			x7 = ROTL32(x7, 7);
			xv = (x0 + xv);
			x0 = ROTL32(x0, 7);
			xu = (x1 + xu);
			x1 = ROTL32(x1, 7);
			xt = (x2 + xt);
			x2 = ROTL32(x2, 7);
			xs = (x3 + xs);
			x3 = ROTL32(x3, 7);
			x4 ^= xj;
			x5 ^= xi;
			x6 ^= xh;
			x7 ^= xg;
			x0 ^= xn;
			x1 ^= xm;
			x2 ^= xl;
			x3 ^= xk;
			xc ^= xr;
			xd ^= xq;
			xe ^= xp;
			xf ^= xo;
			x8 ^= xv;
			x9 ^= xu;
			xa ^= xt;
			xb ^= xs;
			xh = (x4 + xh);
			x4 = ROTL32(x4, 11);
			xg = (x5 + xg);
			x5 = ROTL32(x5, 11);
			xj = (x6 + xj);
			x6 = ROTL32(x6, 11);
			xi = (x7 + xi);
			x7 = ROTL32(x7, 11);
			xl = (x0 + xl);
			x0 = ROTL32(x0, 11);
			xk = (x1 + xk);
			x1 = ROTL32(x1, 11);
			xn = (x2 + xn);
			x2 = ROTL32(x2, 11);
			xm = (x3 + xm);
			x3 = ROTL32(x3, 11);
			xp = (xc + xp);
			xc = ROTL32(xc, 11);
			xo = (xd + xo);
			xd = ROTL32(xd, 11);
			xr = (xe + xr);
			xe = ROTL32(xe, 11);
			xq = (xf + xq);
			xf = ROTL32(xf, 11);
			xt = (x8 + xt);
			x8 = ROTL32(x8, 11);
			xs = (x9 + xs);
			x9 = ROTL32(x9, 11);
			xv = (xa + xv);
			xa = ROTL32(xa, 11);
			xu = (xb + xu);
			xb = ROTL32(xb, 11);
			x0 ^= xh;
			x1 ^= xg;
			x2 ^= xj;
			x3 ^= xi;
			x4 ^= xl;
			x5 ^= xk;
			x6 ^= xn;
			x7 ^= xm;
			x8 ^= xp;
			x9 ^= xo; 
			xa ^= xr;
			xb ^= xq;
			xc ^= xt;
			xd ^= xs;
			xe ^= xv;
			xf ^= xu;

		for (int j = 1; j < 8; j++)
		{
			ROUND_EVEN;
			ROUND_ODD;
		}
		x0 ^= (Hash[8]);
		x1 ^= (Hash[9]);
		x2 ^= (Hash[10]);
		x3 ^= (Hash[11]);
		x4 ^= (Hash[12]);
		x5 ^= (Hash[13]);
		x6 ^= (Hash[14]);
		x7 ^= (Hash[15]);


		for (int j = 0; j < 8; j++)
		{
			ROUND_EVEN;
			ROUND_ODD;
		}
		x0 ^= 0x80;
	
		for (int j = 0; j < 8; j++)
		{
			ROUND_EVEN;
			ROUND_ODD;
		}
		xv ^= 1;

		for(int i = 3; i < 13; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				ROUND_EVEN;
				ROUND_ODD;
			}
		}

		Hash[0] = x0;
		Hash[1] = x1;
		Hash[2] = x2;
		Hash[3] = x3;
		Hash[4] = x4;
		Hash[5] = x5;
		Hash[6] = x6;
		Hash[7] = x7;
		Hash[8] = x8;
		Hash[9] = x9;
		Hash[10] = xa;
		Hash[11] = xb;
		Hash[12] = xc;
		Hash[13] = xd;
		Hash[14] = xe;
		Hash[15] = xf;
	}
}

__host__ void x11_luffaCubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x11_luffaCubehash512_gpu_hash_64 << <grid, block, 0, gpustream[thr_id]>> >(threads, startNounce, (uint64_t*)d_hash);
	CUDA_SAFE_CALL(cudaGetLastError());
}
