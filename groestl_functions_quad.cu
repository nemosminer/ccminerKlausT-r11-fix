#include "cuda_helper.h"

static __device__ __forceinline__ void G256_AddRoundConstantQ_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0, const int round)
{
	x0 = ~x0;
	x1 = ~x1;
	x2 = ~x2;
	x3 = ~x3;
	x4 = ~x4;
	x5 = ~x5;
	x6 = ~x6;
	x7 = ~x7;

	const uint32_t andmask1 = ((-((threadIdx.x & 0x03) == 3)) & 0xffff0000);

	x0 ^= ((-(round & 0x01)) & andmask1);
	x1 ^= ((-(round & 0x02)) & andmask1);
	x2 ^= ((-(round & 0x04)) & andmask1);
	x3 ^= ((-(round & 0x08)) & andmask1);
	x4 ^= (0xAAAA0000 & andmask1);
	x5 ^= (0xCCCC0000 & andmask1);
	x6 ^= (0xF0F00000 & andmask1);
	x7 ^= (0xFF000000 & andmask1);
}

static __device__ __forceinline__ void G256_AddRoundConstantP_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0, const int round)
{
	const uint32_t andmask1 = ((threadIdx.x & 0x03) - 1) >> 16;

	x4 ^= (0xAAAA & andmask1);
	x5 ^= (0xCCCC & andmask1);
	x6 ^= (0xF0F0 & andmask1);
	x7 ^= (0xFF00 & andmask1);

	x0 ^= ((-(round & 0x01)) & andmask1);
	x1 ^= ((-((round & 0x02) >> 1)) & andmask1);
	x2 ^= ((-((round & 0x04) >> 2)) & andmask1);
	x3 ^= ((-((round & 0x08) >> 3)) & andmask1);
}

static __device__ __forceinline__ void G16mul_quad(uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0,
	const uint32_t &y3, const uint32_t &y2, const uint32_t &y1, const uint32_t &y0)
{
    uint32_t t0,t1,t2;
    
    t0 = ((x2 ^ x0) ^ (x3 ^ x1)) & ((y2 ^ y0) ^ (y3 ^ y1));
    t1 = ((x2 ^ x0) & (y2 ^ y0)) ^ t0;
    t2 = ((x3 ^ x1) & (y3 ^ y1)) ^ t0 ^ t1;

    t0 = (x2^x3) & (y2^y3);
    x3 = (x3 & y3) ^ t0 ^ t1;
    x2 = (x2 & y2) ^ t0 ^ t2;

    t0 = (x0^x1) & (y0^y1);
    x1 = (x1 & y1) ^ t0 ^ t1;
    x0 = (x0 & y0) ^ t0 ^ t2;
}

static __device__ __forceinline__ void G256_inv_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0)
{
    uint32_t t0,t1,t2,t3,t4,t5,t6,a,b;

    t3 = x7;
    t2 = x6;
    t1 = x5;
    t0 = x4;

    G16mul_quad(t3, t2, t1, t0, x3, x2, x1, x0);

    a = (x4 ^ x0);
    t0 ^= a;
    t2 ^= (x7 ^ x3) ^ (x5 ^ x1); 
    t1 ^= (x5 ^ x1) ^ a;
    t3 ^= (x6 ^ x2) ^ a;

    b = t0 ^ t1;
    t4 = (t2 ^ t3) & b;
    a = t4 ^ t3 ^ t1;
    t5 = (t3 & t1) ^ a;
    t6 = (t2 & t0) ^ a ^ (t2 ^ t0);

    t4 = (t5 ^ t6) & b;
    t1 = (t6 & t1) ^ t4;
    t0 = (t5 & t0) ^ t4;

    t4 = (t5 ^ t6) & (t2^t3);
    t3 = (t6 & t3) ^ t4;
    t2 = (t5 & t2) ^ t4;

    G16mul_quad(x3, x2, x1, x0, t1, t0, t3, t2);

    G16mul_quad(x7, x6, x5, x4, t1, t0, t3, t2);
}

static __device__ __forceinline__ void transAtoX_quad(uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &x4, uint32_t &x5, uint32_t &x6, uint32_t &x7)
{
    uint32_t t0, t1;
    t0 = x0 ^ x1 ^ x2;
    t1 = x5 ^ x6;
    x2 = t0 ^ t1 ^ x7;
    x6 = t0 ^ x3 ^ x6;
    x3 = x0 ^ x1 ^ x3 ^ x4 ^ x7;    
    x4 = x0 ^ x4 ^ t1;
    x2 = t0 ^ t1 ^ x7;
    x1 = x0 ^ x1 ^ t1;
    x7 = x0 ^ t1 ^ x7;
    x5 = x0 ^ t1;
}

static __device__ __forceinline__ void transXtoA_quad(uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &x4, uint32_t &x5, uint32_t &x6, uint32_t &x7)
{
    uint32_t t0,t2,t3,t5;

    x1 ^= x4;
    t0 = x1 ^ x6;
    x1 ^= x5;

    t2 = x0 ^ x2;
    x2 = x3 ^ x5;
    t2 ^= x2 ^ x6;
    x2 ^= x7;
    t3 = x4 ^ x2 ^ x6;

    t5 = x0 ^ x6;
    x4 = x3 ^ x7;
    x0 = x3 ^ x5;

    x6 = t0;    
    x3 = t2;
    x7 = t3;    
    x5 = t5;    
}

static __device__ __forceinline__ void sbox_quad(uint32_t *const r)
{
    transAtoX_quad(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);

    G256_inv_quad(r[2], r[4], r[1], r[7], r[3], r[0], r[5], r[6]);

    transXtoA_quad(r[7], r[1], r[4], r[2], r[6], r[5], r[0], r[3]);
    
    r[0] = ~r[0];
    r[1] = ~r[1];
    r[5] = ~r[5];
    r[6] = ~r[6];
}

static __device__ __forceinline__ void G256_ShiftBytesP_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0)
{
    uint32_t t0,t1;

	uint32_t tpos = threadIdx.x & 0x03;
	uint32_t shift1 = tpos << 1;
	uint32_t shift2 = shift1 + 1 + ((tpos == 3) << 2);

    t0 = __byte_perm(x0, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x0, 0, 0x3232)>>shift2;
    x0 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x1, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x1, 0, 0x3232)>>shift2;
    x1 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x2, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x2, 0, 0x3232)>>shift2;
    x2 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x3, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x3, 0, 0x3232)>>shift2;
    x3 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x4, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x4, 0, 0x3232)>>shift2;
    x4 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x5, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x5, 0, 0x3232)>>shift2;
    x5 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x6, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x6, 0, 0x3232)>>shift2;
    x6 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x7, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x7, 0, 0x3232)>>shift2;
    x7 = __byte_perm(t0, t1, 0x5410);
}

static __device__ __forceinline__ void G256_ShiftBytesQ_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0)
{
    uint32_t t0,t1;

	uint32_t tpos = threadIdx.x & 0x03;
	uint32_t shift1 = (1 - (tpos >> 1)) + ((tpos & 0x01) << 2);
	uint32_t shift2 = shift1 + 2 + ((tpos == 1) << 2);

    t0 = __byte_perm(x0, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x0, 0, 0x3232)>>shift2;
    x0 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x1, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x1, 0, 0x3232)>>shift2;
    x1 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x2, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x2, 0, 0x3232)>>shift2;
    x2 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x3, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x3, 0, 0x3232)>>shift2;
    x3 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x4, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x4, 0, 0x3232)>>shift2;
    x4 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x5, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x5, 0, 0x3232)>>shift2;
    x5 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x6, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x6, 0, 0x3232)>>shift2;
    x6 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x7, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x7, 0, 0x3232)>>shift2;
    x7 = __byte_perm(t0, t1, 0x5410);
}

static __device__ __forceinline__ void G256_MixFunction_quad(uint32_t *r)
{
#define SHIFT64_16(hi, lo)    __byte_perm(lo, hi, 0x5432)
#define A(v, u)             __shfl_sync(0xffffffff, (int)r[v], ((threadIdx.x+u)&0x03), 4)
#define S(idx, l)            SHIFT64_16( A(idx, (l+1)), A(idx, l) )

#define DOUBLE_ODD(i, bc)        ( S(i, (bc)) ^ A(i, (bc) + 1) )
#define DOUBLE_EVEN(i, bc)        ( S(i, (bc)) ^ A(i, (bc)    ) )

#define SINGLE_ODD(i, bc)        ( S(i, (bc)) )
#define SINGLE_EVEN(i, bc)        ( A(i, (bc)) )
    uint32_t b[8];

	b[0] = (S(0, (1)) ^ A(0, (1) + 1)) ^ DOUBLE_EVEN(0, 3);
	b[1] = (S(1, (1)) ^ A(1, (1) + 1)) ^ DOUBLE_EVEN(1, 3);
	b[2] = (S(2, (1)) ^ A(2, (1) + 1)) ^ DOUBLE_EVEN(2, 3);
	b[3] = (S(3, (1)) ^ A(3, (1) + 1)) ^ DOUBLE_EVEN(3, 3);
	b[4] = (S(4, (1)) ^ A(4, (1) + 1)) ^ DOUBLE_EVEN(4, 3);
	b[5] = (S(5, (1)) ^ A(5, (1) + 1)) ^ DOUBLE_EVEN(5, 3);
	b[6] = (S(6, (1)) ^ A(6, (1) + 1)) ^ DOUBLE_EVEN(6, 3);
	b[7] = (S(7, (1)) ^ A(7, (1) + 1)) ^ DOUBLE_EVEN(7, 3);

	uint32_t tmp = b[7];
	b[7] = b[6] ^ (S(7, (3)) ^ A(7, (3) + 1)) ^ DOUBLE_ODD(7, 4) ^ SINGLE_ODD(7, 6);
	b[6] = b[5] ^ (S(6, (3)) ^ A(6, (3) + 1)) ^ DOUBLE_ODD(6, 4) ^ SINGLE_ODD(6, 6);
	b[5] = b[4] ^ (S(5, (3)) ^ A(5, (3) + 1)) ^ DOUBLE_ODD(5, 4) ^ SINGLE_ODD(5, 6);
	b[4] = b[3] ^ (S(4, (3)) ^ A(4, (3) + 1)) ^ DOUBLE_ODD(4, 4) ^ SINGLE_ODD(4, 6) ^ tmp;
	b[3] = b[2] ^ (S(3, (3)) ^ A(3, (3) + 1)) ^ DOUBLE_ODD(3, 4) ^ SINGLE_ODD(3, 6) ^ tmp;
	b[2] = b[1] ^ (S(2, (3)) ^ A(2, (3) + 1)) ^ DOUBLE_ODD(2, 4) ^ SINGLE_ODD(2, 6);
	b[1] = b[0] ^ (S(1, (3)) ^ A(1, (3) + 1)) ^ DOUBLE_ODD(1, 4) ^ SINGLE_ODD(1, 6) ^ tmp;
	b[0] = tmp ^ (S(0, (3)) ^ A(0, (3) + 1)) ^ DOUBLE_ODD(0, 4) ^ SINGLE_ODD(0, 6);

	tmp = b[7];
	r[7] = b[6] ^ DOUBLE_EVEN(7, 2) ^ DOUBLE_EVEN(7, 3) ^ SINGLE_EVEN(7, 5);
	r[6] = b[5] ^ DOUBLE_EVEN(6, 2) ^ DOUBLE_EVEN(6, 3) ^ SINGLE_EVEN(6, 5);
	r[5] = b[4] ^ DOUBLE_EVEN(5, 2) ^ DOUBLE_EVEN(5, 3) ^ SINGLE_EVEN(5, 5);
	r[4] = b[3] ^ DOUBLE_EVEN(4, 2) ^ DOUBLE_EVEN(4, 3) ^ SINGLE_EVEN(4, 5) ^ tmp;
	r[3] = b[2] ^ DOUBLE_EVEN(3, 2) ^ DOUBLE_EVEN(3, 3) ^ SINGLE_EVEN(3, 5) ^ tmp;
	r[2] = b[1] ^ DOUBLE_EVEN(2, 2) ^ DOUBLE_EVEN(2, 3) ^ SINGLE_EVEN(2, 5);
	r[1] = b[0] ^ DOUBLE_EVEN(1, 2) ^ DOUBLE_EVEN(1, 3) ^ SINGLE_EVEN(1, 5)^tmp;
	r[0] = tmp ^ DOUBLE_EVEN(0, 2) ^ DOUBLE_EVEN(0, 3) ^ SINGLE_EVEN(0, 5);

#undef S
#undef A
#undef SHIFT64_16
#undef t
#undef X
}

static __device__ __forceinline__ void groestl512_perm_P_quad(uint32_t *const r)
{
#if __CUDA_ARCH__ > 500
	const uint32_t andmask1 = ((threadIdx.x & 0x03) - 1) >> 16;

	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);

	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	r[0] ^= andmask1;
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);

	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	r[1] ^= andmask1;
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);

	for (int round = 3; round<14; round++)
	{
		G256_AddRoundConstantP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0], round);
		sbox_quad(r);
		G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
		G256_MixFunction_quad(r);
	}

#else
	for (int round = 0; round<14; round++)
	{
		G256_AddRoundConstantP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0], round);
		sbox_quad(r);
		G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
		G256_MixFunction_quad(r);
	}
#endif

/*



r[4] ^= (0xAAAA & andmask1);
r[5] ^= (0xCCCC & andmask1);
r[6] ^= (0xF0F0 & andmask1);
r[7] ^= (0xFF00 & andmask1);
r[0] ^= andmask1;
r[1] ^= andmask1;
sbox_quad(r);
G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
G256_MixFunction_quad(r);

r[4] ^= (0xAAAA & andmask1);
r[5] ^= (0xCCCC & andmask1);
r[6] ^= (0xF0F0 & andmask1);
r[7] ^= (0xFF00 & andmask1);
r[2] ^= andmask1;
sbox_quad(r);	
G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);


	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	r[2] ^= andmask1;
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);
	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	r[0] ^= andmask1;
	r[2] ^= andmask1;
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);
	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	r[1] ^= andmask1;
	r[2] ^= andmask1;
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);
	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	r[0] ^= andmask1;
	r[1] ^= andmask1;
	r[2] ^= andmask1;
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);
	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	r[3] ^= andmask1;
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);
	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	r[0] ^= andmask1;
	r[3] ^= andmask1;
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);
	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	r[1] ^= andmask1;
	r[3] ^= andmask1;
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);
	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	r[0] ^= andmask1;
	r[1] ^= andmask1;
	r[3] ^= andmask1;
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);
	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	r[2] ^= andmask1;
	r[3] ^= andmask1;
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);
	r[4] ^= (0xAAAA & andmask1);
	r[5] ^= (0xCCCC & andmask1);
	r[6] ^= (0xF0F0 & andmask1);
	r[7] ^= (0xFF00 & andmask1);
	r[0] ^= andmask1;
	r[2] ^= andmask1;
	r[3] ^= andmask1;
	sbox_quad(r);
	G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
	G256_MixFunction_quad(r);
	*/
}

static __device__ __forceinline__ void groestl512_perm_Q_quad(uint32_t *const r)
{    
	for (int round = 0; round<14; round++)
    {
        G256_AddRoundConstantQ_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0], round);
        sbox_quad(r);
        G256_ShiftBytesQ_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
        G256_MixFunction_quad(r);
    }
}

static __device__ __forceinline__ void groestl512_progressMessage_quad(uint32_t *const __restrict__ state, uint32_t *const __restrict__ message)
{
	state[0] = message[0];
	state[1] = message[1];
	state[2] = message[2];
	state[3] = message[3];
	state[4] = message[4];
	state[5] = message[5];
	state[6] = message[6];
	state[7] = message[7];

    if ((threadIdx.x & 0x03) == 3) state[ 1] ^= 0x00008000;
    groestl512_perm_P_quad(state);
    if ((threadIdx.x & 0x03) == 3) state[ 1] ^= 0x00008000;
    groestl512_perm_Q_quad(message);

	state[0] ^= message[0];
	state[1] ^= message[1];
	state[2] ^= message[2];
	state[3] ^= message[3];
	state[4] ^= message[4];
	state[5] ^= message[5];
	state[6] ^= message[6];
	state[7] ^= message[7];

	message[0] = state[0];
	message[1] = state[1];
	message[2] = state[2];
	message[3] = state[3];
	message[4] = state[4];
	message[5] = state[5];
	message[6] = state[6];
	message[7] = state[7];

	groestl512_perm_P_quad(message);
		
	state[0] ^= message[0];
	state[1] ^= message[1];
	state[2] ^= message[2];
	state[3] ^= message[3];
	state[4] ^= message[4];
	state[5] ^= message[5];
	state[6] ^= message[6];
	state[7] ^= message[7];
}
