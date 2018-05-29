extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
}

#include "miner.h"
#include "cuda_helper.h"

extern void quark_blake512_cpu_init(int thr_id);
extern void quark_blake512_cpu_setBlock_80(int thr_id, uint64_t *pdata);
extern void quark_blake512_cpu_setBlock_80_multi(int thr_id, uint64_t *pdata);
extern void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_80_multi(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_bmw512_cpu_init(int thr_id, uint32_t threads);
extern void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_bmw512_cpu_hash_64_quark(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_groestl512_cpu_init(int thr_id, uint32_t threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_skein512_cpu_init(int thr_id);
extern void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_keccak512_cpu_init(int thr_id);
extern void quark_keccakskein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_keccak512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t target, uint32_t *h_found);
extern void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_jh512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t target, uint32_t *h_found);
extern void  quark_jh512_cpu_init(int thr_id);


extern void quark_compactTest_cpu_init(int thr_id, uint32_t threads);
extern void quark_compactTest_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, const uint32_t *inpHashes, const uint32_t *d_validNonceTable,
											uint32_t *d_nonces1, uint32_t *nrm1,
											uint32_t *d_nonces2, uint32_t *nrm2);
extern void quark_compactTest_single_false_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable,
											uint32_t *d_nonces1, uint32_t *nrm1);

extern uint32_t cuda_check_hash_branch(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash);
extern void cuda_check_quarkcoin(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, uint32_t *foundnonces);

// Original Quarkhash Funktion aus einem miner Quelltext
void quarkhash(void *state, const void *input)
{
    sph_blake512_context ctx_blake;
    sph_bmw512_context ctx_bmw;
    sph_groestl512_context ctx_groestl;
    sph_jh512_context ctx_jh;
    sph_keccak512_context ctx_keccak;
    sph_skein512_context ctx_skein;
    
    unsigned char hash[64];

    sph_blake512_init(&ctx_blake);
    sph_blake512 (&ctx_blake, input, 80);
    sph_blake512_close(&ctx_blake, (void*) hash);
    
    sph_bmw512_init(&ctx_bmw);
    sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
    sph_bmw512_close(&ctx_bmw, (void*) hash);

    if (hash[0] & 0x8)
    {
        sph_groestl512_init(&ctx_groestl);
        sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
        sph_groestl512_close(&ctx_groestl, (void*) hash);
    }
    else
    {
        sph_skein512_init(&ctx_skein);
        sph_skein512 (&ctx_skein, (const void*) hash, 64);
        sph_skein512_close(&ctx_skein, (void*) hash);
    }
    
    sph_groestl512_init(&ctx_groestl);
    sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
    sph_groestl512_close(&ctx_groestl, (void*) hash);

    sph_jh512_init(&ctx_jh);
    sph_jh512 (&ctx_jh, (const void*) hash, 64);
    sph_jh512_close(&ctx_jh, (void*) hash);

    if (hash[0] & 0x8)
    {
        sph_blake512_init(&ctx_blake);
        sph_blake512 (&ctx_blake, (const void*) hash, 64);
        sph_blake512_close(&ctx_blake, (void*) hash);
    }
    else
    {
        sph_bmw512_init(&ctx_bmw);
        sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
        sph_bmw512_close(&ctx_bmw, (void*) hash);
    }

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
    sph_keccak512_close(&ctx_keccak, (void*) hash);

    sph_skein512_init(&ctx_skein);
    sph_skein512 (&ctx_skein, (const void*) hash, 64);
    sph_skein512_close(&ctx_skein, (void*) hash);

    if (hash[0] & 0x8)
    {
        sph_keccak512_init(&ctx_keccak);
        sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
        sph_keccak512_close(&ctx_keccak, (void*) hash);
    }
    else
    {
        sph_jh512_init(&ctx_jh);
        sph_jh512 (&ctx_jh, (const void*) hash, 64);
        sph_jh512_close(&ctx_jh, (void*) hash);
    }

    memcpy(state, hash, 32);
}

extern int scanhash_quark(int thr_id, uint32_t *pdata,
    uint32_t *ptarget, uint32_t max_nonce,
    uint32_t *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	uint32_t intensity = 1 << 22;
	intensity = intensity + ((1 << 22)*9/10);
	uint32_t throughputmax = device_intensity(device_map[thr_id], __func__, intensity); // 256*4096
	uint32_t throughput = min(throughputmax, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x0000003f;

	static THREAD uint32_t *foundnonces = nullptr;
	static THREAD uint32_t *d_hash = nullptr;
	static THREAD uint32_t *d_branch1Nonces = nullptr;
	static THREAD uint32_t *d_branch2Nonces = nullptr;
	static THREAD uint32_t *d_branch3Nonces = nullptr;

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
//		}

#if defined WIN32 && !defined _WIN64
		// 2GB limit for cudaMalloc
		if(throughputmax > 0x7fffffffULL / (16 * sizeof(uint32_t)))
		{
			applog(LOG_ERR, "intensity too high");
			mining_has_stopped[thr_id] = true;
			cudaStreamDestroy(gpustream[thr_id]);
			proper_exit(2);
		}
#endif

		// Konstanten kopieren, Speicher belegen
		CUDA_SAFE_CALL(cudaMalloc(&d_hash, 16ULL * sizeof(uint32_t) * throughputmax));
		CUDA_SAFE_CALL(cudaMallocHost(&foundnonces, 4 * 4));
//		CUDA_SAFE_CALL(cudaMalloc(&d_branch1Nonces, sizeof(uint32_t)*throughput));
//		CUDA_SAFE_CALL(cudaMalloc(&d_branch2Nonces, sizeof(uint32_t)*throughput));
		uint32_t noncebuffersize = throughputmax * 7 / 10;
		uint32_t noncebuffersize2 = (throughputmax * 7 / 10)*7/10;

		CUDA_SAFE_CALL(cudaMalloc(&d_branch1Nonces, sizeof(uint32_t)*noncebuffersize2));
		CUDA_SAFE_CALL(cudaMalloc(&d_branch2Nonces, sizeof(uint32_t)*noncebuffersize2));
		CUDA_SAFE_CALL(cudaMalloc(&d_branch3Nonces, sizeof(uint32_t)*noncebuffersize));
		quark_blake512_cpu_init(thr_id);
		quark_compactTest_cpu_init(thr_id, throughputmax);
		quark_keccak512_cpu_init(thr_id);
		quark_jh512_cpu_init(thr_id);
		CUDA_SAFE_CALL(cudaGetLastError());
		mining_has_stopped[thr_id] = false;
		init = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	quark_blake512_cpu_setBlock_80(thr_id, (uint64_t *)endiandata);

	do {

		uint32_t nrm1 = 0, nrm2 = 0, nrm3 = 0;

		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash);
		quark_bmw512_cpu_hash_64_quark(thr_id, throughput, pdata[19], NULL, d_hash);

		quark_compactTest_single_false_cpu_hash_64(thr_id, throughput, pdata[19], d_hash, NULL,
			d_branch3Nonces, &nrm3);

		// nur den Skein Branch weiterverfolgen
		quark_skein512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces, d_hash);

		// das ist der unbedingte Branch für Groestl512
		quark_groestl512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces, d_hash);

		// das ist der unbedingte Branch für JH512
		quark_jh512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces, d_hash);

		// quarkNonces in branch1 und branch2 aufsplitten gemäss if (hash[0] & 0x8)
		quark_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash, d_branch3Nonces,
			d_branch1Nonces, &nrm1,
			d_branch2Nonces, &nrm2);

		// das ist der bedingte Branch für Blake512
		quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces, d_hash);

		// das ist der bedingte Branch für Bmw512
		quark_bmw512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces, d_hash);

		quark_keccakskein512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces, d_hash);


		// quarkNonces in branch1 und branch2 aufsplitten gemäss if (hash[0] & 0x8)
		quark_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash, d_branch3Nonces,
			d_branch1Nonces, &nrm1,
			d_branch3Nonces, &nrm2);

		quark_keccak512_cpu_hash_64_final(thr_id, nrm1, pdata[19], d_branch1Nonces, d_hash, ptarget[7], foundnonces);
		quark_jh512_cpu_hash_64_final(thr_id, nrm2, pdata[19], d_branch3Nonces, d_hash, ptarget[7], foundnonces+2);
		CUDA_SAFE_CALL(cudaStreamSynchronize(gpustream[thr_id]));
		if(foundnonces[0] == 0xffffffff)
		{
			foundnonces[0] = foundnonces[2];
			foundnonces[1] = foundnonces[3];
		}
		else
		{
			if(foundnonces[1] == 0xffffffff)
				foundnonces[1] = foundnonces[2];
		}

		if(stop_mining)
		{
			mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);
		}

		if (foundnonces[0] != 0xffffffff)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8]={0};
			if(opt_verify){ be32enc(&endiandata[19], foundnonces[0]);
			quarkhash(vhash64, endiandata);

			} if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				if(opt_benchmark)  applog(LOG_INFO, "GPU #%d: Found nonce $%08X", device_map[thr_id], foundnonces[0]);
				// check if there was some other ones...
				if (foundnonces[1] != 0xffffffff)
				{
					if(opt_verify){ be32enc(&endiandata[19], foundnonces[1]);
					quarkhash(vhash64, endiandata);

					} if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
					{
						pdata[21] = foundnonces[1];
						res++;
						if(opt_benchmark)  applog(LOG_INFO, "GPU #%d: Found second nonce $%08X", device_map[thr_id], foundnonces[1]);
					}
					else
					{
						if (vhash64[7] != Htarg) // don't show message if it is equal but fails fulltest
							applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", device_map[thr_id], foundnonces[1]);
					}
				}
				pdata[19] = foundnonces[0];

				return res;
			}
			else
			{
				if (vhash64[7] != Htarg) // don't show message if it is equal but fails fulltest
					applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", device_map[thr_id], foundnonces[0]);
			}
		}
		pdata[19] += throughput; CUDA_SAFE_CALL(cudaGetLastError());
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce ;
	return 0;
}
