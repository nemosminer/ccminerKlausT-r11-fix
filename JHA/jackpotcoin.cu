extern "C"
{
#include "sph/sph_keccak.h"
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_jh.h"
#include "sph/sph_skein.h"
}

#include "miner.h"
#include "cuda_helper.h"

extern void jackpot_keccak512_cpu_init(int thr_id, uint32_t threads);
extern void jackpot_keccak512_cpu_setBlock(int thr_id, void *pdata, size_t inlen);
extern void jackpot_keccak512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_groestl512_cpu_init(int thr_id, uint32_t threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_skein512_cpu_init(int thr_id);
extern void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void jackpot_compactTest_cpu_init(int thr_id, uint32_t threads);
extern void jackpot_compactTest_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable, 
											uint32_t *d_nonces1, uint32_t *nrm1,
											uint32_t *d_nonces2, uint32_t *nrm2);

extern uint32_t cuda_check_hash_branch(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash);

// Original jackpothash Funktion aus einem miner Quelltext
unsigned int jackpothash(void *state, const void *input)
{
    sph_blake512_context     ctx_blake;
    sph_groestl512_context   ctx_groestl;
    sph_jh512_context        ctx_jh;
    sph_keccak512_context    ctx_keccak;
    sph_skein512_context     ctx_skein;

    uint32_t hash[16];

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512 (&ctx_keccak, input, 80);
    sph_keccak512_close(&ctx_keccak, hash);

    unsigned int round;
    for (round = 0; round < 3; round++) {
        if (hash[0] & 0x01) {
           sph_groestl512_init(&ctx_groestl);
           sph_groestl512 (&ctx_groestl, (&hash), 64);
           sph_groestl512_close(&ctx_groestl, (&hash));
        }
        else {
           sph_skein512_init(&ctx_skein);
           sph_skein512 (&ctx_skein, (&hash), 64);
           sph_skein512_close(&ctx_skein, (&hash));
        }
        if (hash[0] & 0x01) {
           sph_blake512_init(&ctx_blake);
           sph_blake512 (&ctx_blake, (&hash), 64);
           sph_blake512_close(&ctx_blake, (&hash));
        }
        else {
           sph_jh512_init(&ctx_jh);
           sph_jh512 (&ctx_jh, (&hash), 64);
           sph_jh512_close(&ctx_jh, (&hash));
        }
    }
    memcpy(state, hash, 32);

    return round;
}

extern int scanhash_jackpot(int thr_id, uint32_t *pdata,
    uint32_t *ptarget, uint32_t max_nonce,
    uint32_t *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	uint32_t throughputmax = device_intensity(device_map[thr_id], __func__, 1U << 20);
	uint32_t throughput = min(throughputmax, (max_nonce - first_nonce)) & 0xfffffc00;

	if (opt_benchmark)
		ptarget[7] = 0x000f;

	static THREAD uint32_t *d_hash = nullptr;
	static THREAD uint32_t *d_jackpotNonces = nullptr;
	static THREAD uint32_t *d_branch1Nonces = nullptr;
	static THREAD uint32_t *d_branch2Nonces = nullptr;
	static THREAD uint32_t *d_branch3Nonces = nullptr;
	static THREAD volatile bool init = false;

	if (!init)
	{
		if(throughputmax == 1<<20)
			applog(LOG_INFO, "GPU #%d: using default intensity 20", device_map[thr_id]);
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		CUDA_SAFE_CALL(cudaDeviceReset());
		CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaschedule));
		CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));

		CUDA_SAFE_CALL(cudaMalloc(&d_hash, 16ULL * sizeof(uint32_t) * throughputmax));

		jackpot_keccak512_cpu_init(thr_id, throughputmax);
		jackpot_compactTest_cpu_init(thr_id, throughputmax);
		quark_groestl512_cpu_init(thr_id, throughputmax);
		quark_skein512_cpu_init(thr_id);
		cuda_check_cpu_init(thr_id, throughputmax);

		cudaMalloc(&d_branch1Nonces, sizeof(uint32_t)*throughputmax * 1.25/2);
		cudaMalloc(&d_branch2Nonces, sizeof(uint32_t)*throughputmax * 1.25/2);
		cudaMalloc(&d_branch3Nonces, sizeof(uint32_t)*throughputmax * 1.25); // 25% more than we need, just in case

		CUDA_SAFE_CALL(cudaMalloc(&d_jackpotNonces, sizeof(uint32_t)*throughputmax * 2));
		mining_has_stopped[thr_id] = false;

		init = true;
	}

	uint32_t endiandata[22];
	for (int k=0; k < 22; k++)
		be32enc(&endiandata[k], pdata[k]);

	jackpot_keccak512_cpu_setBlock(thr_id, (void*)endiandata, 80);
	cuda_check_cpu_setTarget(ptarget, thr_id);

	do {
		// erstes Keccak512 Hash mit CUDA
		jackpot_keccak512_cpu_hash(thr_id, throughput, pdata[19], d_hash);

		uint32_t nrm1, nrm2, nrm3;

		// Runde 1 (ohne Gröstl)

		jackpot_compactTest_cpu_hash_64(thr_id, throughput, pdata[19], d_hash, NULL,
				d_branch1Nonces, &nrm1,
				d_branch3Nonces, &nrm3);

		// verfolge den skein-pfad weiter
		quark_skein512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces, d_hash);

		// noch schnell Blake & JH
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash, d_branch3Nonces,
			d_branch1Nonces, &nrm1,
			d_branch2Nonces, &nrm2);

		if (nrm1+nrm2 == nrm3) {
			quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces, d_hash);
			quark_jh512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces, d_hash);
		}

		// Runde 3 (komplett)

		// jackpotNonces in branch1/2 aufsplitten gemäss if (hash[0] & 0x01)
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash, d_branch3Nonces,
			d_branch1Nonces, &nrm1,
			d_branch2Nonces, &nrm2);

		if (nrm1+nrm2 == nrm3) {
			quark_groestl512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces, d_hash);
			quark_skein512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces, d_hash);
		}

		// jackpotNonces in branch1/2 aufsplitten gemäss if (hash[0] & 0x01)
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash, d_branch3Nonces,
			d_branch1Nonces, &nrm1,
			d_branch2Nonces, &nrm2);

		if (nrm1+nrm2 == nrm3) {
			quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces, d_hash);
			quark_jh512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces, d_hash);
		}

		// Runde 3 (komplett)

		// jackpotNonces in branch1/2 aufsplitten gemäss if (hash[0] & 0x01)
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash, d_branch3Nonces,
			d_branch1Nonces, &nrm1,
			d_branch2Nonces, &nrm2);

		if (nrm1+nrm2 == nrm3) {
			quark_groestl512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces, d_hash);
			quark_skein512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces, d_hash);
		}

		// jackpotNonces in branch1/2 aufsplitten gemäss if (hash[0] & 0x01)
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash, d_branch3Nonces,
			d_branch1Nonces, &nrm1,
			d_branch2Nonces, &nrm2);

		if (nrm1+nrm2 == nrm3) {
			quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces, d_hash);
			quark_jh512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces, d_hash);
		}

		uint32_t foundNonce = cuda_check_hash_branch(thr_id, nrm3, pdata[19], d_branch3Nonces, d_hash);
		if(stop_mining) {mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);}
		if(foundNonce != 0xffffffff)
		{
			unsigned int rounds;
			uint32_t vhash64[8]={0};
			uint32_t Htarg = ptarget[7];
			if(opt_verify){ be32enc(&endiandata[19], foundNonce);

			// diese jackpothash Funktion gibt die Zahl der Runden zurück
			rounds = jackpothash(vhash64, endiandata);

			} if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				int res = 1;
				uint32_t secNonce = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash, foundNonce);
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (secNonce != 0)
				{
					if(opt_verify){ be32enc(&endiandata[19], secNonce);
					rounds = jackpothash(vhash64, endiandata);

					} if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
					{
						pdata[21] = secNonce;
						res++;
					}
					else
					{
						if(opt_verify)
							applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU (%d rounds)!", device_map[thr_id], secNonce, rounds);
					}
				}
				pdata[19] = foundNonce;
				return res;
			}
			else {
				if(opt_verify)
					applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU (%d rounds)!", device_map[thr_id], foundNonce, rounds);
			}
		}

		pdata[19] += throughput; CUDA_SAFE_CALL(cudaGetLastError());
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce ;
	return 0;
}
