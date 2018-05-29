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

static uint32_t *d_hash[MAX_GPUS];

// Speicher zur Generierung der Noncevektoren f�r die bedingten Hashes
static uint32_t *d_branch1Nonces[MAX_GPUS];
static uint32_t *d_branch2Nonces[MAX_GPUS];
static uint32_t *d_branch3Nonces[MAX_GPUS];

extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_bmw512_cpu_init(int thr_id, uint32_t threads);
extern void quark_bmw512_cpu_setBlock_80(int thr_id, void *pdata);
extern void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash);
extern void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash);

extern void quark_groestl512_cpu_init(int thr_id, uint32_t threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_doublegroestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_skein512_cpu_init(int thr_id);
extern void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_keccak512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_jh512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_compactTest_cpu_init(int thr_id, uint32_t threads);
extern void quark_compactTest_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable,
											uint32_t *d_nonces1, uint32_t *nrm1,
											uint32_t *d_nonces2, uint32_t *nrm2);
extern void quark_compactTest_single_false_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable,
											uint32_t *d_nonces1, uint32_t *nrm1);

extern uint32_t cuda_check_hash_branch(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash);
extern void cuda_check_quarkcoin(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, uint32_t *foundnonces);

/* CPU Hash */
extern "C" void animehash(void *state, const void *input)
{
    sph_blake512_context ctx_blake;
    sph_bmw512_context ctx_bmw;
    sph_groestl512_context ctx_groestl;
    sph_jh512_context ctx_jh;
    sph_keccak512_context ctx_keccak;
    sph_skein512_context ctx_skein;
    
    unsigned char hash[64];

    sph_bmw512_init(&ctx_bmw);
    // ZBMW;
    sph_bmw512 (&ctx_bmw, (const void*) input, 80);
    sph_bmw512_close(&ctx_bmw, (void*) hash);

    sph_blake512_init(&ctx_blake);
    // ZBLAKE;
    sph_blake512 (&ctx_blake, hash, 64);
    sph_blake512_close(&ctx_blake, (void*) hash);
    
    if (hash[0] & 0x8)
    {
        sph_groestl512_init(&ctx_groestl);
        // ZGROESTL;
        sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
        sph_groestl512_close(&ctx_groestl, (void*) hash);
    }
    else
    {
        sph_skein512_init(&ctx_skein);
        // ZSKEIN;
        sph_skein512 (&ctx_skein, (const void*) hash, 64);
        sph_skein512_close(&ctx_skein, (void*) hash);
    }
    
    sph_groestl512_init(&ctx_groestl);
    // ZGROESTL;
    sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
    sph_groestl512_close(&ctx_groestl, (void*) hash);

    sph_jh512_init(&ctx_jh);
    // ZJH;
    sph_jh512 (&ctx_jh, (const void*) hash, 64);
    sph_jh512_close(&ctx_jh, (void*) hash);

    if (hash[0] & 0x8)
    {
        sph_blake512_init(&ctx_blake);
        // ZBLAKE;
        sph_blake512 (&ctx_blake, (const void*) hash, 64);
        sph_blake512_close(&ctx_blake, (void*) hash);
    }
    else
    {
        sph_bmw512_init(&ctx_bmw);
        // ZBMW;
        sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
        sph_bmw512_close(&ctx_bmw, (void*) hash);
    }

    sph_keccak512_init(&ctx_keccak);
    // ZKECCAK;
    sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
    sph_keccak512_close(&ctx_keccak, (void*) hash);

    sph_skein512_init(&ctx_skein);
    // SKEIN;
    sph_skein512 (&ctx_skein, (const void*) hash, 64);
    sph_skein512_close(&ctx_skein, (void*) hash);

    if (hash[0] & 0x8)
    {
        sph_keccak512_init(&ctx_keccak);
        // ZKECCAK;
        sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
        sph_keccak512_close(&ctx_keccak, (void*) hash);
    }
    else
    {
        sph_jh512_init(&ctx_jh);
        // ZJH;
        sph_jh512 (&ctx_jh, (const void*) hash, 64);
        sph_jh512_close(&ctx_jh, (void*) hash);
    }

    memcpy(state, hash, 32);
}

/*
struct HashPredicate
{    
    HashPredicate(uint32_t *hashes, uint32_t startNonce) :
        m_hashes(hashes),
        m_startNonce(startNonce)
        { }

    __device__
    bool operator()(const uint32_t x)
    {
        uint32_t *const Hash = &m_hashes[(x - m_startNonce)*16];
        return hash[0] & 0x8;
    }

    uint32_t *m_hashes;
    uint32_t  m_startNonce;
};
*/

static volatile bool init[MAX_GPUS] = { false };

extern int scanhash_anime(int thr_id, uint32_t *pdata,
    uint32_t *ptarget, uint32_t max_nonce,
    uint32_t *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, 1 << 20); // 256*256*8
	throughput = min(throughput, (max_nonce - first_nonce)) & 0xfffffc00;

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	if (!init[thr_id])
	{
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));
		get_cuda_arch(&cuda_arch[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput));

		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id);
		quark_bmw512_cpu_init(thr_id, throughput);
		cuda_check_cpu_init(thr_id, throughput);
		quark_compactTest_cpu_init(thr_id, throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_branch1Nonces[thr_id], sizeof(uint32_t)*throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_branch2Nonces[thr_id], sizeof(uint32_t)*throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_branch3Nonces[thr_id], sizeof(uint32_t)*throughput));

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	quark_bmw512_cpu_setBlock_80(thr_id, (void*)endiandata);
	cuda_check_cpu_setTarget(ptarget, thr_id);

	do {
		uint32_t nrm1=0, nrm2=0, nrm3=0;

		// erstes BMW512 Hash mit CUDA
		quark_bmw512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);

		// das ist der unbedingte Branch f�r Blake512
		quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id]);

		quark_compactTest_single_false_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], NULL,
				d_branch3Nonces[thr_id], &nrm3);
		
		// nur den Skein Branch weiterverfolgen
		quark_skein512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id]);

		// das ist der unbedingte Branch f�r Groestl512
		quark_groestl512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id]);

		// das ist der unbedingte Branch f�r JH512
		quark_jh512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id]);

		// quarkNonces in branch1 und branch2 aufsplitten gem�ss if (hash[0] & 0x8)
		quark_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2);

		// das ist der bedingte Branch f�r Blake512
		quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id]);

		// das ist der bedingte Branch f�r Bmw512
		quark_bmw512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id]);

		// das ist der unbedingte Branch f�r Keccak512
		quark_keccak512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id]);

		// das ist der unbedingte Branch f�r Skein512
		quark_skein512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id]);

		// quarkNonces in branch1 und branch2 aufsplitten gem�ss if (hash[0] & 0x8)
		quark_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2);

		quark_keccak512_cpu_hash_64_final(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id]);
		quark_jh512_cpu_hash_64_final(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id]);

		uint32_t foundnonces[2];
		cuda_check_quarkcoin(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], foundnonces);
		if(stop_mining) {mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);}
		if(foundnonces[0] != 0xffffffff)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundnonces[0]);
			animehash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				// check if there was some other ones...
				if (foundnonces[1] != 0xffffffff)
				{
					be32enc(&endiandata[19], foundnonces[1]);
					animehash(vhash64, endiandata);
					if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
					{

						pdata[21] = foundnonces[1];
						res++;
						if (opt_benchmark)  applog(LOG_INFO, "GPU #%d: Found second nonce $%08X", device_map[thr_id], foundnonces[1]);
					}
				}
				pdata[19] = foundnonces[0];
				if (opt_benchmark) applog(LOG_INFO, "GPU #%d: Found nonce $%08X", device_map[thr_id], foundnonces[0]);
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

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
