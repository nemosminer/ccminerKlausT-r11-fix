extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_skein.h"
#include "sph/sph_keccak.h"
#include "sph/sph_cubehash.h"
#include "lyra2/Lyra2.h"
}

#include "miner.h"
#include "cuda_helper.h"
extern "C" {
#include "SHA3api_ref.h"
}
extern void blakeKeccak256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash);
extern void blake256_cpu_setBlock_80(int thr_id, uint32_t *pdata);

extern void keccak256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);
extern void keccak256_cpu_init(int thr_id, uint32_t threads);

extern void skein256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);
extern void skein256_cpu_init(int thr_id, uint32_t threads);

extern void skeinCube256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);


#ifdef ORG
extern void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);
#else
extern void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, bool eco_mode);
#endif
extern void lyra2v2_cpu_init(int thr_id, uint64_t* matrix);

extern void bmw256_cpu_init(int thr_id);
extern void bmw256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *resultnonces, uint32_t target);

extern void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash);

void lyra2v2_hash(void *state, const void *input)
{
	sph_blake256_context      ctx_blake;
	sph_keccak256_context     ctx_keccak;
	sph_skein256_context      ctx_skein;
	sph_bmw256_context        ctx_bmw;
	sph_cubehash256_context   ctx_cube;

	uint32_t hashA[8], hashB[8];

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashB, 32);
	sph_cubehash256_close(&ctx_cube, hashA);


	LYRA2(hashB, 32, hashA, 32, hashA, 32, 1, 4, 4, LYRA2_NOBUG);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashB, 32);
	sph_skein256_close(&ctx_skein, hashA);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashA, 32);
	sph_cubehash256_close(&ctx_cube, hashB);

/*
	sph_bmw256_init(&ctx_bmw);
	sph_bmw256(&ctx_bmw, hashB, 32);
	sph_bmw256_close(&ctx_bmw, hashA);
*/
	BMWHash(256, (const BitSequence*)hashB, 256, (BitSequence*)hashA);

	memcpy(state, hashA, 32);
}

int scanhash_lyra2v2(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done)
{
	static THREAD uint64_t *d_hash = nullptr;
	static THREAD uint64_t *d_hash2 = nullptr;

	const uint32_t first_nonce = pdata[19];
#ifdef ORG
	uint32_t intensity = 256 * 256 * 8;
#else
	double intensity = 19.0;
#endif

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device_map[thr_id]);
#ifdef ORG
	if(strstr(props.name, "Titan"))
	{
		intensity = 256 * 256 * 15;
#if defined _WIN64 || defined _LP64
		intensity = 256 * 256 * 22;
#endif
	}
	else if(strstr(props.name, "1080"))
	{
		intensity = 256 * 256 * 15;
#if defined _WIN64 || defined _LP64
		intensity = 256 * 256 * 22;
#endif
	}
	else if(strstr(props.name, "1070"))
	{
		intensity = 256 * 256 * 15;
#if defined _WIN64 || defined _LP64
		intensity = 256 * 256 * 22;
#endif
	}
	else if(strstr(props.name, "970"))
	{
		intensity = 256 * 256 * 15;
#if defined _WIN64 || defined _LP64
		intensity = 256 * 256 * 22;
#endif
	}
	else if (strstr(props.name, "980"))
	{
		intensity = 256 * 256 * 15;
#if defined _WIN64 || defined _LP64
		intensity = 256 * 256 * 22;
#endif
	}
	else if (strstr(props.name, "750 Ti"))
	{
		intensity = 256 * 256 * 12;
	}
	else if (strstr(props.name, "750"))
	{
		intensity = 256 * 256 * 5;
	}
	else if (strstr(props.name, "960"))
	{
		intensity = 256 * 256 * 8;
	}
	uint32_t throughputmax = device_intensity(device_map[thr_id], __func__, intensity);
#else
	// TITAN series
	if (strstr(props.name, "TITAN V")) intensity = 23.0;			// Volta(NVIDIA TITAN V)
	else if (strstr(props.name, "GTX TITAN X")) intensity = 21.75;	// Maxwell(GTX TITAN X)
	else if (strstr(props.name, "TITAN X")) intensity = 22.5;		// Pascal(NVIDIA TITAN X/NVIDIA TITAN Xp)
	else if (strstr(props.name, "TITAN Z")) intensity = 20.75;		// Kepler(GTX TITAN Z)
	else if (strstr(props.name, "TITAN")) intensity = 19.5;			// Kepler(GTX TITAN/GTX TITAN Black)
																	// Pascal
	else if (strstr(props.name, "1080")) {
		if (strstr(props.name, "Ti")) intensity = 22.5;			// GTX 1080Ti
		else intensity = 22.25;									// GTX 1080
	}
	else if (strstr(props.name, "1070")) {
		if (strstr(props.name, "Ti")) intensity = 22.0;			// GTX 1070Ti
		else intensity = 21.75;									// GTX 1070
	}
	else if (strstr(props.name, "1060")) intensity = 21.0;		// GTX 1060
	else if (strstr(props.name, "1050")) intensity = 20.0;		// GTX 1050Ti/GTX 1050
	else if (strstr(props.name, "1030")) intensity = 19.25;		// GT 1030
																// Maxwell
	else if (strstr(props.name, "1080")) {
		if (strstr(props.name, "Ti")) intensity = 21.5;			// GTX 980Ti
		else intensity = 21.25;									// GTX 980
	}
	else if (strstr(props.name, "970")) intensity = 21.0;		// GTX 970
	else if (strstr(props.name, "960")) intensity = 20.25;		// GTX 960
	else if (strstr(props.name, "950")) intensity = 19.75;		// GTX 950
	else if (strstr(props.name, "750")) {
		if (strstr(props.name, "Ti")) intensity = 19.0;			// GTX 750Ti
		else intensity = 18.75;									// GTX 750
	}
	// Kepler
	else if (strstr(props.name, "780")) {
		if (strstr(props.name, "Ti")) intensity = 19.75;		// GTX 780Ti
		else intensity = 19.5;									// GTX 780
	}
	else if (strstr(props.name, "770")) intensity = 19.25;
	else if (strstr(props.name, "760")) intensity = 18.75;
	else if (strstr(props.name, "740")) intensity = 17.0;
	else if (strstr(props.name, "730")) intensity = 17.0;
	else if (strstr(props.name, "720")) intensity = 15.75;
	else if (strstr(props.name, "710")) intensity = 16.0;
	else if (strstr(props.name, "690")) intensity = 20.0;
	else if (strstr(props.name, "680")) intensity = 19.0;
	else if (strstr(props.name, "670")) intensity = 18.75;
	else if (strstr(props.name, "660")) {
		if (strstr(props.name, "Ti")) intensity = 18.75;		// GTX 660Ti
		else intensity = 18.25;									// GTX 660
	}
	else if (strstr(props.name, "650")) {
		if (strstr(props.name, "Ti")) intensity = 18.0;			// GTX 660Ti BOOST/GTX 660Ti
		else intensity = 17.0;									// GTX 660
	}
	else if (strstr(props.name, "645")) intensity = 17.5;
	else if (strstr(props.name, "640")) intensity = 17.0;
	else if (strstr(props.name, "635")) intensity = 17.0;
	else if (strstr(props.name, "630")) intensity = 15.75;
	// Tesla series
	else if (strstr(props.name, "V100")) intensity = 23.0;
	else if (strstr(props.name, "P100")) intensity = 22.25;
	else if (strstr(props.name, "P40")) intensity = 22.5;
	else if (strstr(props.name, "P4")) intensity = 21.5;
	else if (strstr(props.name, "M60")) intensity = 22.25;
	else if (strstr(props.name, "M6")) intensity = 20.75;
	else if (strstr(props.name, "M40")) intensity = 21.75;
	else if (strstr(props.name, "M4")) intensity = 20.25;
	// Quadro series
	else if (strstr(props.name, "GP100")) intensity = 22.25;
	else if (strstr(props.name, "P6000")) intensity = 22.5;
	else if (strstr(props.name, "P5000")) intensity = 22.25;
	else if (strstr(props.name, "P4000")) intensity = 21.5;
	else if (strstr(props.name, "P2000")) intensity = 20.75;
	else if (strstr(props.name, "P1000")) intensity = 20.0;
	else if (strstr(props.name, "P600")) intensity = 18.75;
	else if (strstr(props.name, "P400")) intensity = 18.0;
	else if (strstr(props.name, "M6000")) intensity = 21.75;
	else if (strstr(props.name, "M5000")) intensity = 21.5;
	else if (strstr(props.name, "M4000")) intensity = 21.0;
	else if (strstr(props.name, "M2000")) intensity = 19.75;
	else if (strstr(props.name, "K6000")) intensity = 19.0;
	else if (strstr(props.name, "K5200")) intensity = 19.0;
	else if (strstr(props.name, "K5000")) intensity = 18.0;
	else if (strstr(props.name, "K600")) intensity = 15.0;
	else if (strstr(props.name, "K420")) intensity = 15.0;
	uint32_t throughputmax = (uint32_t)((1.0 + (intensity - (int)intensity))*(1UL << (int)intensity));
	throughputmax = device_intensity(device_map[thr_id], __func__, throughputmax);
#endif
	uint32_t throughput = min(throughputmax, max_nonce - first_nonce) & 0xfffffe00;

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x004f;

	static THREAD bool init = false;
	if (!init)
	{ 
#ifdef ORG
		if(throughputmax == intensity)
			applog(LOG_INFO, "GPU #%d: using default intensity %.3f", device_map[thr_id], throughput2intensity(throughputmax));
#else
		intensity = throughput2intensity(throughputmax);
		applog(LOG_WARNING, "Using intensity %2.2f (%d threads)", intensity, throughputmax);
#endif
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		CUDA_SAFE_CALL(cudaDeviceReset());
		CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaschedule));
		CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));
#if defined WIN32 && !defined _WIN64
		// 2GB limit for cudaMalloc
#ifdef ORG
		if(throughputmax > 0x7fffffffULL / (16 * 4 * 4 * sizeof(uint64_t)))
#else
		if(throughputmax > 0x7fffffffULL / (4 * 4 * sizeof(uint64_t)))
#endif
		{
			applog(LOG_ERR, "intensity too high");
			mining_has_stopped[thr_id] = true;
			cudaStreamDestroy(gpustream[thr_id]);
			proper_exit(2);
		}
#endif
#ifdef ORG
		CUDA_SAFE_CALL(cudaMalloc(&d_hash2, 16ULL  * 4 * 4 * sizeof(uint64_t) * throughputmax));
#else
		CUDA_SAFE_CALL(cudaMalloc(&d_hash2, 3ULL * 4 * sizeof(uint64_t) * throughputmax));
#endif
		CUDA_SAFE_CALL(cudaMalloc(&d_hash, 8ULL * sizeof(uint32_t) * throughputmax));

		bmw256_cpu_init(thr_id);
		lyra2v2_cpu_init(thr_id, d_hash2);
		mining_has_stopped[thr_id] = false;

		init = true; 
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	blake256_cpu_setBlock_80(thr_id, pdata);

	do {
		uint32_t foundNonce[2] = { 0, 0 };

		blakeKeccak256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash);
//		keccak256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash);
		cubehash256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash);
#ifdef ORG
		lyra2v2_cpu_hash_32(thr_id, throughput, pdata[19], d_hash);
#else
		lyra2v2_cpu_hash_32(thr_id, throughput, pdata[19], d_hash, opt_eco_mode);
#endif
		skein256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash);
		cubehash256_cpu_hash_32(thr_id, throughput,pdata[19], d_hash);
		bmw256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash, foundNonce, ptarget[7]);
		if(stop_mining)
		{
			mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);
		}
		if(foundNonce[0] != 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8]={0};
			if(opt_verify)
			{
				be32enc(&endiandata[19], foundNonce[0]);
				lyra2v2_hash(vhash64, endiandata);
			}
			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				// check if there was some other ones...
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (foundNonce[1] != 0)
				{
					if(opt_verify)
					{
						be32enc(&endiandata[19], foundNonce[1]);
						lyra2v2_hash(vhash64, endiandata);
					}
					if(vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
					{
						pdata[21] = foundNonce[1];
						res++;
						if(opt_benchmark)  applog(LOG_INFO, "GPU #%d Found second nonce %08x", thr_id, foundNonce[1]);
					}
					else
					{
						if(vhash64[7] != Htarg) // don't show message if it is equal but fails fulltest
							applog(LOG_WARNING, "GPU #%d: result does not validate on CPU!", device_map[thr_id]);
					}
				}
				pdata[19] = foundNonce[0];
				if (opt_benchmark) applog(LOG_INFO, "GPU #%d Found nonce % 08x", thr_id, foundNonce[0]);
				return res;
			}
			else
			{
				if (vhash64[7] != Htarg) // don't show message if it is equal but fails fulltest
					applog(LOG_WARNING, "GPU #%d: result does not validate on CPU!", device_map[thr_id]);
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce ;
	return 0;
}
