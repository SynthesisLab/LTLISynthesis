#include <set>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <warpcore/hash_set.cuh>

using UINT_64 = std::uint64_t;

const std::size_t maxNumOfTraces = 64;

__constant__ uint8_t d_pTraceLen[maxNumOfTraces];
__constant__ uint8_t d_cTraceLen[maxNumOfTraces];

inline
cudaError_t checkCuda(cudaError_t res) {
#ifndef MEASUREMENT_MODE
    if (res != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(res));
        assert(res == cudaSuccess);
    }
#endif
    return res;
}

// Finding the left and right indices that makes the final LTL to bring to the host later
__global__ void generateResIndices(
    const int index,
    const int alphabetSize,
    const int* d_leftIdx,
    const int* d_rightIdx,
    int* d_FinalLTLIdx)
{

    int resIdx = 0;
    while (d_FinalLTLIdx[resIdx] != -1) resIdx++;
    int queue[600];
    queue[0] = index;
    int head = 0;
    int tail = 1;
    while (head < tail) {
        int ltl = queue[head];
        int l = d_leftIdx[ltl];
        int r = d_rightIdx[ltl];
        d_FinalLTLIdx[resIdx++] = ltl;
        d_FinalLTLIdx[resIdx++] = l;
        d_FinalLTLIdx[resIdx++] = r;
        if (l >= alphabetSize) queue[tail++] = l;
        if (r >= alphabetSize) queue[tail++] = r;
        head++;
    }

}

__device__ void makeRlxUnqChkCSs(
    UINT_64* CS,
    UINT_64& hCS,
    UINT_64& lCS,
    const int numOfTraces,
    const int RlxUnqChkTyp,
    const int lenSum)
{

    if (lenSum > 126) {

        // We need a relaxed uniqueness check

        switch (RlxUnqChkTyp) {

        case 1: {
            // Not handled with LTLI
            const int stride = lenSum / 126;
            int j = 0;
            for (int i = 0; i < numOfTraces; ++i) {
                for (int k = 0; k < (d_pTraceLen[i] + d_cTraceLen[i]); k += stride, ++j) {
                    if (j < 63) {
                        if (CS[i] & ((UINT_64)1 << k)) lCS |= (UINT_64)1 << j;
                    } else if (j < 126) {
                        if (CS[i] & ((UINT_64)1 << k)) hCS |= (UINT_64)1 << (j - 63);
                    } else break;
                }
            }
            break;
        }

        case 2: {
            // Not handled with LTLI 
            int j = 0;
            for (int i = 0; i < numOfTraces; ++i) {
                UINT_64 bitPtr = 1;
                int maxbitsForThisTrace = (126 * (d_pTraceLen[i] + d_cTraceLen[i]) + lenSum) / lenSum;
                for (int k = 0; k < maxbitsForThisTrace; ++k, ++j, bitPtr <<= 1) {
                    if (j < 63) {
                        if (CS[i] & bitPtr) lCS |= (UINT_64)1 << j;
                    } else if (j < 126) {
                        if (CS[i] & bitPtr) hCS |= (UINT_64)1 << (j - 63);
                    } else break;
                }
            }
            break;
        }

        case 3: {
            // Handled with LTLI
            for (int i = 0; i < numOfTraces; ++i) {
                UINT_64 pcs = CS[2 * i];
                UINT_64 ccs = CS[2 * i + 1];
                pcs = (pcs ^ (pcs >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
                pcs = (pcs ^ (pcs >> 27)) * UINT64_C(0x94d049bb133111eb);
                pcs = pcs ^ (pcs >> 31);
                ccs = (ccs ^ (ccs >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
                ccs = (ccs ^ (ccs >> 27)) * UINT64_C(0x94d049bb133111eb);
                ccs = ccs ^ (ccs >> 31);
                hCS ^= pcs; lCS ^= ccs;
            }
            break;
        }

        }

    } else {

        // The result will be minimal
        // Handled with LTLI

        int j = 0;
        for (int i = 0; i < numOfTraces; ++i) {
            UINT_64 bitPtr = 1;
            for (int k = 0; k < d_pTraceLen[i]; ++k, ++j, bitPtr <<= 1) {
                if (j < 63) {
                    if (CS[2 * i] & bitPtr) lCS |= (UINT_64)1 << j;
                } else if (j < 126) {
                    if (CS[2 * i] & bitPtr) hCS |= (UINT_64)1 << (j - 63);
                } else break;
            }
            bitPtr = 1;
            for (int k = 0; k < d_cTraceLen[i]; ++k, ++j, bitPtr <<= 1) {
                if (j < 63) {
                    if (CS[2 * i + 1] & bitPtr) lCS |= (UINT_64)1 << j;
                } else if (j < 126) {
                    if (CS[2 * i + 1] & bitPtr) hCS |= (UINT_64)1 << (j - 63);
                } else break;
            }
        }

    }

}

// Initialising the hashSets with the alphabet before starting the enumeration
template<class hash_set_t>
__global__ void hashSetsInitialisation(
    const int numOfTraces,
    const int RlxUnqChkTyp,
    const int lenSum,
    hash_set_t cHashSet, hash_set_t iHashSet,
    UINT_64* d_LTLcache)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    UINT_64 CS[maxNumOfTraces * 2];

    for (int i = 0; i < numOfTraces; ++i) {
        CS[2 * i] = d_LTLcache[2 * (tid * numOfTraces + i)];
        CS[2 * i + 1] = d_LTLcache[2 * (tid * numOfTraces + i) + 1];
    }

    UINT_64 hCS{}, lCS{};
    makeRlxUnqChkCSs(CS, hCS, lCS, numOfTraces, RlxUnqChkTyp, lenSum);

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    int H = cHashSet.insert(hCS, group); int L = cHashSet.insert(lCS, group);
    H = (H > 0) ? H : -H; L = (L > 0) ? L : -L;
    UINT_64 HL = H; HL <<= 32; HL |= L;
    iHashSet.insert(HL, group);

}

enum class Op { Not, And, Or, Next, Finally, Globally, Until };

template<Op op>
__device__ void applyOperator(
    UINT_64* CS,
    UINT_64* d_LTLcache,
    int ldx, int rdx,
    int numOfTraces)
{

    if constexpr (op == Op::Not) {
        for (int i = 0; i < numOfTraces; ++i) {
            UINT_64 prefixNegationFixer = ((UINT_64)1 << d_pTraceLen[i]) - 1;
            UINT_64 cycleNegationFixer = ((UINT_64)1 << d_cTraceLen[i]) - 1;
            CS[2 * i] = ~d_LTLcache[2 * (ldx * numOfTraces + i)] & prefixNegationFixer;
            CS[2 * i + 1] = ~d_LTLcache[2 * (ldx * numOfTraces + i) + 1] & cycleNegationFixer;
        }
    } else if constexpr (op == Op::And) {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[2 * i] = d_LTLcache[2 * (ldx * numOfTraces + i)] & d_LTLcache[2 * (rdx * numOfTraces + i)];
            CS[2 * i + 1] = d_LTLcache[2 * (ldx * numOfTraces + i) + 1] & d_LTLcache[2 * (rdx * numOfTraces + i) + 1];
        }
    } else if constexpr (op == Op::Or) {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[2 * i] = d_LTLcache[2 * (ldx * numOfTraces + i)] | d_LTLcache[2 * (rdx * numOfTraces + i)];
            CS[2 * i + 1] = d_LTLcache[2 * (ldx * numOfTraces + i) + 1] | d_LTLcache[2 * (rdx * numOfTraces + i) + 1];
        }
    } else if constexpr (op == Op::Next) {
        for (int i = 0; i < numOfTraces; ++i) {
            UINT_64 cycleFirstBit = d_LTLcache[2 * (ldx * numOfTraces + i) + 1] & 1;
            CS[2 * i] = d_LTLcache[2 * (ldx * numOfTraces + i)] >> 1 | (cycleFirstBit << (d_pTraceLen[i] - 1));
            CS[2 * i + 1] = d_LTLcache[2 * (ldx * numOfTraces + i) + 1] >> 1 | (cycleFirstBit << (d_cTraceLen[i] - 1));
        }
    } else if constexpr (op == Op::Finally) {
        for (int i = 0; i < numOfTraces; ++i) {
            UINT_64 pcs = d_LTLcache[2 * (ldx * numOfTraces + i)];
            UINT_64 ccs = d_LTLcache[2 * (ldx * numOfTraces + i) + 1];
            if (ccs == 0) {
                pcs |= pcs >> 1; pcs |= pcs >> 2; pcs |= pcs >> 4;
                pcs |= pcs >> 8; pcs |= pcs >> 16; pcs |= pcs >> 32;
            } else {
                pcs = ((UINT_64)1 << d_pTraceLen[i]) - 1;
                ccs = ((UINT_64)1 << d_cTraceLen[i]) - 1;
            }
            CS[2 * i] = pcs; CS[2 * i + 1] = ccs;
        }
    } else if constexpr (op == Op::Globally) {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[2 * i] = d_LTLcache[2 * (ldx * numOfTraces + i)];
            CS[2 * i + 1] = d_LTLcache[2 * (ldx * numOfTraces + i) + 1];
            UINT_64 pcs = ~CS[2 * i] & (((UINT_64)1 << d_pTraceLen[i]) - 1);
            UINT_64 ccs = ~CS[2 * i + 1] & (((UINT_64)1 << d_cTraceLen[i]) - 1);
            if (ccs == 0) {
                pcs |= pcs >> 1; pcs |= pcs >> 2; pcs |= pcs >> 4;
                pcs |= pcs >> 8; pcs |= pcs >> 16; pcs |= pcs >> 32;
            } else {
                pcs = ((UINT_64)1 << d_pTraceLen[i]) - 1;
                ccs = ((UINT_64)1 << d_cTraceLen[i]) - 1;
            }
            CS[2 * i] &= ~pcs;
            CS[2 * i + 1] &= ~ccs;
        }
    } else if constexpr (op == Op::Until) {
        for (int i = 0; i < numOfTraces; ++i) {
            __uint128_t cl = d_LTLcache[2 * (ldx * numOfTraces + i) + 1]; cl <<= d_cTraceLen[i];
            cl += d_LTLcache[2 * (ldx * numOfTraces + i) + 1];
            __uint128_t cr = d_LTLcache[2 * (rdx * numOfTraces + i) + 1]; cr <<= d_cTraceLen[i];
            cr += d_LTLcache[2 * (rdx * numOfTraces + i) + 1];
            cr |= cl & (cr >> 1);  cl &= cl >> 1;
            cr |= cl & (cr >> 2);  cl &= cl >> 2;
            cr |= cl & (cr >> 4);  cl &= cl >> 4;
            cr |= cl & (cr >> 8);  cl &= cl >> 8;
            cr |= cl & (cr >> 16); cl &= cl >> 16;
            cr |= cl & (cr >> 32); cl &= cl >> 32;
            cr |= cl & (cr >> 64);
            CS[2 * i + 1] = static_cast<UINT_64>(cr);
            UINT_64 pl = d_LTLcache[2 * (ldx * numOfTraces + i)];
            UINT_64 propagatedBit = ((CS[2 * i + 1] & 1) << (d_pTraceLen[i] - 1)) & pl;
            UINT_64 pr = d_LTLcache[2 * (rdx * numOfTraces + i)] | propagatedBit;
            pr |= pl & (pr >> 1);  pl &= pl >> 1;
            pr |= pl & (pr >> 2);  pl &= pl >> 2;
            pr |= pl & (pr >> 4);  pl &= pl >> 4;
            pr |= pl & (pr >> 8);  pl &= pl >> 8;
            pr |= pl & (pr >> 16); pl &= pl >> 16;
            pr |= pl & (pr >> 32);
            CS[2 * i] = pr;
        }
    } else {
        [] <bool flag = false>() { static_assert(flag, "Unhandled operator"); }();
    }

}

__device__ void processOnTheFly(
    UINT_64* CS,
    int tid,
    int numOfTraces, int numOfP,
    int ldx, int rdx,
    int* d_temp_leftIdx, int* d_temp_rightIdx,
    int* d_FinalLTLIdx)
{

    bool found = true;
    for (int i = 0; found && i < numOfP; ++i) if (!(CS[2 * i] & 1)) found = false;
    for (int i = numOfP; found && i < numOfTraces; ++i) if (CS[2 * i] & 1) found = false;

    if (found) {
        d_temp_leftIdx[tid] = ldx; d_temp_rightIdx[tid] = rdx;
        atomicCAS(d_FinalLTLIdx, -1, tid);
    }

}

template <typename hash_set_t>
__device__ bool processUniqueCS(
    UINT_64* CS,
    const int numOfTraces,
    const int RlxUnqChkTyp,
    const int lenSum,
    hash_set_t& cHashSet, hash_set_t& iHashSet)
{

    UINT_64 hCS{}, lCS{};
    makeRlxUnqChkCSs(CS, hCS, lCS, numOfTraces, RlxUnqChkTyp, lenSum);

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    int H = cHashSet.insert(hCS, group); int L = cHashSet.insert(lCS, group);
    H = (H > 0) ? H : -H; L = (L > 0) ? L : -L;
    UINT_64 HL = H; HL <<= 32; HL |= L;
    return (iHashSet.insert(HL, group) > 0) ? false : true;

}

__device__ void insertInCache(
    bool CS_is_unique,
    UINT_64* CS,
    int tid,
    int numOfTraces, int numOfP,
    int ldx, int rdx,
    UINT_64* d_temp_LTLcache,
    int* d_temp_leftIdx, int* d_temp_rightIdx,
    int* d_FinalLTLIdx)
{

    if (CS_is_unique) {

        for (int i = 0; i < numOfTraces; ++i) {
            d_temp_LTLcache[2 * (tid * numOfTraces + i)] = CS[2 * i];
            d_temp_LTLcache[2 * (tid * numOfTraces + i) + 1] = CS[2 * i + 1];
        }
        d_temp_leftIdx[tid] = ldx; d_temp_rightIdx[tid] = rdx;

        bool found = true;
        for (int i = 0; found && i < numOfP; ++i) if (!(CS[2 * i] & 1)) found = false;
        for (int i = numOfP; found && i < numOfTraces; ++i) if (CS[2 * i] & 1) found = false;
        if (found) atomicCAS(d_FinalLTLIdx, -1, tid);

    } else {

        for (int i = 0; i < numOfTraces; ++i) {
            d_temp_LTLcache[2 * (tid * numOfTraces + i)] = (UINT_64)-1;
            d_temp_LTLcache[2 * (tid * numOfTraces + i) + 1] = (UINT_64)-1;
        }
        d_temp_leftIdx[tid] = -1; d_temp_rightIdx[tid] = -1;

    }

}

template<Op op, class hash_set_t>
__global__ void processOperator(
    const int idx1, const int idx2,
    const int idx3, const int idx4,
    const int numOfP, const int numOfN,
    const int RlxUnqChkTyp,
    const int lenSum,
    const bool onTheFly,
    UINT_64* d_LTLcache, UINT_64* d_temp_LTLcache,
    int* d_temp_leftIdx, int* d_temp_rightIdx,
    hash_set_t cHashSet, hash_set_t iHashSet,
    int* d_FinalLTLIdx)
{

    const int tid = (blockDim.x * blockIdx.x + threadIdx.x);
    const int numOfTraces = numOfP + numOfN;
    constexpr bool isUnary = (op == Op::Not || op == Op::Next || op == Op::Finally || op == Op::Globally);
    int maxTid = isUnary ? (idx2 - idx1 + 1) : (idx4 - idx3 + 1) * (idx2 - idx1 + 1);

    if (tid < maxTid) {

        int ldx = isUnary ? idx1 + tid : idx1 + tid / (idx4 - idx3 + 1);
        int rdx = isUnary ? 0 : idx3 + tid % (idx4 - idx3 + 1);
        const int modifiedTid = (op == Op::Until) ? (tid * 2) : tid;
        UINT_64 CS[maxNumOfTraces * 2];
        applyOperator<op>(CS, d_LTLcache, ldx, rdx, numOfTraces);

        if (onTheFly) {
            processOnTheFly(
                CS, modifiedTid, numOfTraces, numOfP, ldx, rdx,
                d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
            );
        } else {
            bool CS_is_unique =
                processUniqueCS(CS, numOfTraces, RlxUnqChkTyp, lenSum, cHashSet, iHashSet);
            insertInCache(
                CS_is_unique, CS, modifiedTid, numOfTraces, numOfP, ldx, rdx,
                d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
            );
        }

        if (op == Op::Until) {

            applyOperator<Op::Until>(CS, d_LTLcache, rdx, ldx, numOfTraces);

            if (onTheFly) {
                processOnTheFly(
                    CS, modifiedTid + 1, numOfTraces, numOfP, rdx, ldx,
                    d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
                );
            } else {
                bool CS_is_unique =
                    processUniqueCS(CS, numOfTraces, RlxUnqChkTyp, lenSum, cHashSet, iHashSet);
                insertInCache(
                    CS_is_unique, CS, modifiedTid + 1, numOfTraces, numOfP, rdx, ldx,
                    d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
                );
            }

        }

    }

}

// Generating the final RE string recursively
// When all the left and right indices are ready in the host
std::string toString(
    int index,
    std::map<int, std::pair<int, int>>& indicesMap,
    const std::set<char>& alphabet,
    const int* startPoints)
{

    if (index < alphabet.size()) {
        std::string s(1, *next(alphabet.begin(), index));
        return s;
    }
    int i = 0;
    while (index >= startPoints[i]) { i++; }
    i--;

    if (i % 7 == 0) {
        std::string res = toString(indicesMap[index].first, indicesMap, alphabet, startPoints);
        return "~(" + res + ")";
    }

    if (i % 7 == 1) {
        std::string left = toString(indicesMap[index].first, indicesMap, alphabet, startPoints);
        std::string right = toString(indicesMap[index].second, indicesMap, alphabet, startPoints);
        return "(" + left + ")" + "&" + "(" + right + ")";
    }

    if (i % 7 == 2) {
        std::string left = toString(indicesMap[index].first, indicesMap, alphabet, startPoints);
        std::string right = toString(indicesMap[index].second, indicesMap, alphabet, startPoints);
        return "(" + left + ")" + "|" + "(" + right + ")";
    }

    if (i % 7 == 3) {
        std::string res = toString(indicesMap[index].first, indicesMap, alphabet, startPoints);
        return "X(" + res + ")";
    }

    if (i % 7 == 4) {
        std::string res = toString(indicesMap[index].first, indicesMap, alphabet, startPoints);
        return "F(" + res + ")";
    }

    if (i % 7 == 5) {
        std::string res = toString(indicesMap[index].first, indicesMap, alphabet, startPoints);
        return "G(" + res + ")";
    }

    std::string left = toString(indicesMap[index].first, indicesMap, alphabet, startPoints);
    std::string right = toString(indicesMap[index].second, indicesMap, alphabet, startPoints);
    return "(" + left + ")" + "U" + "(" + right + ")";

}

// Bringing the left and right indices of the LTL from device to host
// If LTL is found, this index is from the temp memory               (temp = true)
// For printing other LTLs if needed, indices are in the main memory (temp = false)
std::string LTLtoString(
    bool temp,
    const int FinalLTLIdx,
    const int lastIdx,
    const std::set<char>& alphabet,
    const int* startPoints,
    const int* d_leftIdx,
    const int* d_rightIdx,
    const int* d_temp_leftIdx,
    const int* d_temp_rightIdx)
{

    auto* LIdx = new int[1];
    auto* RIdx = new int[1];

    if (temp) {
        checkCuda(cudaMemcpy(LIdx, d_temp_leftIdx + FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(RIdx, d_temp_rightIdx + FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        checkCuda(cudaMemcpy(LIdx, d_leftIdx + FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(RIdx, d_rightIdx + FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
    }

    auto alphabetSize = static_cast<int> (alphabet.size());

    int* d_resIndices;
    checkCuda(cudaMalloc(&d_resIndices, 600 * sizeof(int)));

    thrust::device_ptr<int> d_resIndices_ptr(d_resIndices);
    thrust::fill(d_resIndices_ptr, d_resIndices_ptr + 600, -1);

    if (*LIdx >= alphabetSize) generateResIndices << <1, 1 >> > (*LIdx, alphabetSize, d_leftIdx, d_rightIdx, d_resIndices);
    if (*RIdx >= alphabetSize) generateResIndices << <1, 1 >> > (*RIdx, alphabetSize, d_leftIdx, d_rightIdx, d_resIndices);

    int resIndices[600];
    checkCuda(cudaMemcpy(resIndices, d_resIndices, 600 * sizeof(int), cudaMemcpyDeviceToHost));

    std::map<int, std::pair<int, int>> indicesMap;

    if (temp) indicesMap.insert(std::make_pair(INT_MAX - 1, std::make_pair(*LIdx, *RIdx)));
    else      indicesMap.insert(std::make_pair(FinalLTLIdx, std::make_pair(*LIdx, *RIdx)));

    int i = 0;
    while (resIndices[i] != -1 && i + 2 < 600) {
        int ltl = resIndices[i];
        int l = resIndices[i + 1];
        int r = resIndices[i + 2];
        indicesMap.insert(std::make_pair(ltl, std::make_pair(l, r)));
        i += 3;
    }

    if (i + 2 >= 600) return "Size of the output is too big";

    cudaFree(d_resIndices);

    if (temp) return toString(INT_MAX - 1, indicesMap, alphabet, startPoints);
    else      return toString(FinalLTLIdx, indicesMap, alphabet, startPoints);

}

// Transfering the unique CSs from temp to main LTLcache
void storeUniqueLTLs(
    int N,
    int& lastIdx,
    const int numOfTraces,
    const int LTLcacheCapacity,
    bool& onTheFly,
    UINT_64* d_LTLcache,
    UINT_64* d_temp_LTLcache,
    int* d_leftIdx,
    int* d_rightIdx,
    int* d_temp_leftIdx,
    int* d_temp_rightIdx)
{

    thrust::device_ptr<UINT_64> new_end_ptr;
    thrust::device_ptr<UINT_64> d_LTLcache_ptr(d_LTLcache + 2 * lastIdx * numOfTraces);
    thrust::device_ptr<UINT_64> d_temp_LTLcache_ptr(d_temp_LTLcache);
    thrust::device_ptr<int> d_leftIdx_ptr(d_leftIdx + lastIdx);
    thrust::device_ptr<int> d_rightIdx_ptr(d_rightIdx + lastIdx);
    thrust::device_ptr<int> d_temp_leftIdx_ptr(d_temp_leftIdx);
    thrust::device_ptr<int> d_temp_rightIdx_ptr(d_temp_rightIdx);

    new_end_ptr = // End of d_temp_LTLcache
        thrust::remove(d_temp_LTLcache_ptr, d_temp_LTLcache_ptr + 2 * N * numOfTraces, (UINT_64)-1);
    thrust::remove(d_temp_leftIdx_ptr, d_temp_leftIdx_ptr + N, -1);
    thrust::remove(d_temp_rightIdx_ptr, d_temp_rightIdx_ptr + N, -1);

    // It stores all (or a part of) unique CSs until language cache gets full
    // If language cache gets full, it makes onTheFly mode on
    int numberOfNewUniqueLTLs = static_cast<int>(new_end_ptr - d_temp_LTLcache_ptr) / (2 * numOfTraces);
    if (lastIdx + numberOfNewUniqueLTLs > LTLcacheCapacity) {
        N = LTLcacheCapacity - lastIdx;
        onTheFly = true;
    } else N = numberOfNewUniqueLTLs;

    thrust::copy_n(d_temp_LTLcache_ptr, 2 * N * numOfTraces, d_LTLcache_ptr);
    thrust::copy_n(d_temp_leftIdx_ptr, N, d_leftIdx_ptr);
    thrust::copy_n(d_temp_rightIdx_ptr, N, d_rightIdx_ptr);

    lastIdx += N;

}

int costOf(const int index, const int* startPoints) {
    int i = 0;
    while (index >= startPoints[i]) { i++; }
    return((i - 1) / 4);
}

std::string LTLI(
    const unsigned short* costFun,
    const unsigned short maxCost,
    const unsigned int RlxUnqChkTyp,
    const unsigned int NegType,
    const std::set<char> alphabet,
    int& LTLcost,
    std::uint64_t& allLTLs,
    const std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> pos,
    const std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> neg) {

    // --------------------------------
    // Generating and checking alphabet
    // --------------------------------

    const int numOfP = pos.size();
    const int numOfN = neg.size();
    const int numOfTraces = numOfP + numOfN;

    int maxLenOfTraces{};
    uint8_t* pTraceLen = new uint8_t[numOfTraces];
    uint8_t* cTraceLen = new uint8_t[numOfTraces];

    int TLIdx{};
    int lenSum{};

    for (const auto& trace : pos) {
        lenSum += trace.first.size() + trace.second.size();
        pTraceLen[TLIdx] = trace.first.size();
        cTraceLen[TLIdx++] = trace.second.size();
        if (trace.first.size() + trace.second.size() > maxLenOfTraces) {
            maxLenOfTraces = trace.first.size() + trace.second.size();
        }
    }
    for (const auto& trace : neg) {
        lenSum += trace.first.size() + trace.second.size();
        pTraceLen[TLIdx] = trace.first.size();
        cTraceLen[TLIdx++] = trace.second.size();
        if (trace.first.size() + trace.second.size() > maxLenOfTraces) {
            maxLenOfTraces = trace.first.size() + trace.second.size();
        }
    }

    if (numOfTraces > maxNumOfTraces || maxLenOfTraces > sizeof(UINT_64) * 8 - 1) {
        printf("In this version, The input can have:\n");
        printf("1) At most %zu traces. It is currently %d.\n", maxNumOfTraces, numOfTraces);
        printf("2) Max(len(trace)) = %d. It is currently %d.\n", static_cast<int>(sizeof(UINT_64) * 8 - 1), maxLenOfTraces);
        return "see_the_error";
    }

    // Copying the length of traces into the constant memory
    checkCuda(cudaMemcpyToSymbol(d_pTraceLen, pTraceLen, numOfTraces * sizeof(uint8_t)));
    checkCuda(cudaMemcpyToSymbol(d_cTraceLen, cTraceLen, numOfTraces * sizeof(uint8_t)));

    const int alphabetSize = static_cast<int>(alphabet.size());

    auto* LTLcache = new UINT_64[alphabetSize * numOfTraces * 2];

    // Index of the last free position in the LTLcache
    int lastIdx{};

#ifndef MEASUREMENT_MODE
    printf("Cost %-2d | (A) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
        costFun[0], allLTLs, 0, alphabetSize);
#endif

    int index{};
    for (int i = 0; i < alphabetSize; ++i) {
        bool found = true;
        std::string ch(1, *next(alphabet.begin(), i));
        for (const auto& trace : pos) {
            UINT_64 binTrace{};
            UINT_64 idx = 1;
            for (const auto& token : trace.first) {
                for (const auto& c : token) {
                    if (c == ch[0]) {
                        binTrace |= idx;
                        break;
                    }
                }
                idx <<= 1;
            }
            LTLcache[index++] = binTrace;
            if (!(binTrace & 1)) found = false;
            binTrace = 0;
            idx = 1;
            for (const auto& token : trace.second) {
                for (const auto& c : token) {
                    if (c == ch[0]) {
                        binTrace |= idx;
                        break;
                    }
                }
                idx <<= 1;
            }
            LTLcache[index++] = binTrace;
        }
        for (const auto& trace : neg) {
            UINT_64 binTrace{};
            UINT_64 idx = 1;
            for (const auto& token : trace.first) {
                for (const auto& c : token) {
                    if (c == ch[0]) {
                        binTrace |= idx;
                        break;
                    }
                }
                idx <<= 1;
            }
            LTLcache[index++] = binTrace;
            if (binTrace & 1) found = false;
            binTrace = 0;
            idx = 1;
            for (const auto& token : trace.second) {
                for (const auto& c : token) {
                    if (c == ch[0]) {
                        binTrace |= idx;
                        break;
                    }
                }
                idx <<= 1;
            }
            LTLcache[index++] = binTrace;
        }
        allLTLs++; lastIdx++;
        if (found) return ch;
    }

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    // Cost function
    int c1 = costFun[0]; // cost of p
    int c2 = costFun[1]; // cost of ~
    int c3 = costFun[2]; // cost of &
    int c4 = costFun[3]; // cost of |
    int c5 = costFun[4]; // cost of X
    int c6 = costFun[5]; // cost of F
    int c7 = costFun[6]; // cost of G
    int c8 = costFun[7]; // cost of U

    int maxAllocationSize;
    cudaDeviceGetAttribute(&maxAllocationSize, cudaDevAttrMaxPitch, 0);

    const int LTLcacheCapacity = maxAllocationSize / (2 * numOfTraces * sizeof(UINT_64)) * 1.5;
    const int temp_LTLcacheCapacity = LTLcacheCapacity / 2;

    // const int LTLcacheCapacity = 2000000;
    // const int temp_LTLcacheCapacity = 100000000;

    // 7 for ~, &, |, X, F, G, U
    int* startPoints = new int[(maxCost + 2) * 7]();
    startPoints[c1 * 7 + 6] = lastIdx;
    startPoints[(c1 + 1) * 7] = lastIdx;

    int* d_FinalLTLIdx;
    int* FinalLTLIdx = new int[1]; *FinalLTLIdx = -1;
    checkCuda(cudaMalloc(&d_FinalLTLIdx, sizeof(int)));
    checkCuda(cudaMemcpy(d_FinalLTLIdx, FinalLTLIdx, sizeof(int), cudaMemcpyHostToDevice));

    UINT_64* d_LTLcache, * d_temp_LTLcache;
    int* d_leftIdx, * d_rightIdx, * d_temp_leftIdx, * d_temp_rightIdx;
    checkCuda(cudaMalloc(&d_leftIdx, LTLcacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_rightIdx, LTLcacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_leftIdx, temp_LTLcacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_rightIdx, temp_LTLcacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_LTLcache, 2 * LTLcacheCapacity * numOfTraces * sizeof(UINT_64)));
    checkCuda(cudaMalloc(&d_temp_LTLcache, 2 * temp_LTLcacheCapacity * numOfTraces * sizeof(UINT_64)));

    using hash_set_t = warpcore::HashSet<
        UINT_64,         // Key type
        UINT_64(0) - 1,  // Empty key
        UINT_64(0) - 2,  // Tombstone key
        warpcore::probing_schemes::QuadraticProbing<warpcore::hashers::MurmurHash <UINT_64>>>;

    hash_set_t cHashSet(2 * LTLcacheCapacity);
    hash_set_t iHashSet(2 * LTLcacheCapacity);

    checkCuda(cudaMemcpy(d_LTLcache, LTLcache, 2 * alphabetSize * numOfTraces * sizeof(UINT_64), cudaMemcpyHostToDevice));
    hashSetsInitialisation<hash_set_t> << <1, alphabetSize >> > (numOfTraces, RlxUnqChkTyp, lenSum, cHashSet, iHashSet, d_LTLcache);

    // ----------------------------
    // Enumeration of the next LTLs
    // ----------------------------

    bool onTheFly = false, lastRound = false;
    int shortageCost = -1;

    for (LTLcost = c1 + 1; LTLcost <= maxCost; ++LTLcost) {

        // Once it uses a previous cost that is not fully stored, it should continue as the last round
        if (onTheFly) {
            int dif = LTLcost - shortageCost;
            if (dif == c2 || dif == c1 + c3 || dif == c1 + c4 || dif == c5 || dif == c6 || dif == c7 || dif == c1 + c8) lastRound = true;
        }

        // Negation (~)
        // NegType = 1 is for negation of phi
        // NegType = 2 is for negation of char only
        if ((NegType == 1 && LTLcost - c2 >= c1) || (NegType == 2 && LTLcost - c2 == c1)) {

            int Idx1 = startPoints[(LTLcost - c2) * 7];
            int Idx2 = startPoints[(LTLcost - c2 + 1) * 7] - 1;
            int N = Idx2 - Idx1 + 1;

            if (N) {
                int x = Idx1, y;
                do {
                    y = x + std::min(temp_LTLcacheCapacity - 1, Idx2 - x);
                    N = y - x + 1;
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (~) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLcost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::Not, hash_set_t> << <Blc, 1024 >> > (
                        x, y, 0, 0, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { startPoints[LTLcost * 7 + 1] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(N, lastIdx, numOfTraces, LTLcacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx2);
            }

        }
        startPoints[LTLcost * 7 + 1] = lastIdx;

        // Intersection (&)
        for (int i = c1; 2 * i <= LTLcost - c3; ++i) {

            int Idx1 = startPoints[i * 7];
            int Idx2 = startPoints[(i + 1) * 7] - 1;
            int Idx3 = startPoints[(LTLcost - i - c3) * 7];
            int Idx4 = startPoints[(LTLcost - i - c3 + 1) * 7] - 1;
            int N = (Idx4 - Idx3 + 1) * (Idx2 - Idx1 + 1);

            if (N) {
                int x = Idx3, y;
                do {
                    y = x + std::min(temp_LTLcacheCapacity / (Idx2 - Idx1 + 1) - 1, Idx4 - x);
                    N = (y - x + 1) * (Idx2 - Idx1 + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (&) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLcost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::And, hash_set_t> << <Blc, 1024 >> > (
                        Idx1, Idx2, x, y, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { startPoints[LTLcost * 7 + 2] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(N, lastIdx, numOfTraces, LTLcacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx4);
            }

        }
        startPoints[LTLcost * 7 + 2] = lastIdx;

        // Union (|)
        for (int i = c1; 2 * i <= LTLcost - c4; ++i) {

            int Idx1 = startPoints[i * 7];
            int Idx2 = startPoints[(i + 1) * 7] - 1;
            int Idx3 = startPoints[(LTLcost - i - c4) * 7];
            int Idx4 = startPoints[(LTLcost - i - c4 + 1) * 7] - 1;
            int N = (Idx4 - Idx3 + 1) * (Idx2 - Idx1 + 1);

            if (N) {
                int x = Idx3, y;
                do {
                    y = x + std::min(temp_LTLcacheCapacity / (Idx2 - Idx1 + 1) - 1, Idx4 - x);
                    N = (y - x + 1) * (Idx2 - Idx1 + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (|) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLcost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::Or, hash_set_t> << <Blc, 1024 >> > (
                        Idx1, Idx2, x, y, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { startPoints[LTLcost * 7 + 3] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(N, lastIdx, numOfTraces, LTLcacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx4);
            }

        }
        startPoints[LTLcost * 7 + 3] = lastIdx;

        // Next (X)
        if (LTLcost - c5 >= c1) {

            int Idx1 = startPoints[(LTLcost - c5) * 7];
            int Idx2 = startPoints[(LTLcost - c5 + 1) * 7] - 1;
            int N = Idx2 - Idx1 + 1;

            if (N) {
                int x = Idx1, y;
                do {
                    y = x + std::min(temp_LTLcacheCapacity - 1, Idx2 - x);
                    N = y - x + 1;
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (X) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLcost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::Next, hash_set_t> << <Blc, 1024 >> > (
                        x, y, 0, 0, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { startPoints[LTLcost * 7 + 4] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(N, lastIdx, numOfTraces, LTLcacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx2);
            }

        }
        startPoints[LTLcost * 7 + 4] = lastIdx;

        // Finally (F)
        if (LTLcost - c6 >= c1) {

            int Idx1 = startPoints[(LTLcost - c6) * 7];
            int Idx2 = startPoints[(LTLcost - c6 + 1) * 7] - 1;
            int N = Idx2 - Idx1 + 1;

            if (N) {
                int x = Idx1, y;
                do {
                    y = x + std::min(temp_LTLcacheCapacity - 1, Idx2 - x);
                    N = y - x + 1;
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (F) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLcost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::Finally, hash_set_t> << <Blc, 1024 >> > (
                        x, y, 0, 0, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { startPoints[LTLcost * 7 + 5] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(N, lastIdx, numOfTraces, LTLcacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx2);
            }

        }
        startPoints[LTLcost * 7 + 5] = lastIdx;

        // Globally (G)
        if (LTLcost - c7 >= c1) {

            int Idx1 = startPoints[(LTLcost - c7) * 7];
            int Idx2 = startPoints[(LTLcost - c7 + 1) * 7] - 1;
            int N = Idx2 - Idx1 + 1;

            if (N) {
                int x = Idx1, y;
                do {
                    y = x + std::min(temp_LTLcacheCapacity - 1, Idx2 - x);
                    N = y - x + 1;
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (G) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLcost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::Globally, hash_set_t> << <Blc, 1024 >> > (
                        x, y, 0, 0, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += N;
                    if (*FinalLTLIdx != -1) { startPoints[LTLcost * 7 + 6] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(N, lastIdx, numOfTraces, LTLcacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx2);
            }

        }
        startPoints[LTLcost * 7 + 6] = lastIdx;

        // Until (U)
        for (int i = c1; 2 * i <= LTLcost - c8; ++i) {

            int Idx1 = startPoints[i * 7];
            int Idx2 = startPoints[(i + 1) * 7] - 1;
            int Idx3 = startPoints[(LTLcost - i - c8) * 7];
            int Idx4 = startPoints[(LTLcost - i - c8 + 1) * 7] - 1;
            int N = (Idx4 - Idx3 + 1) * (Idx2 - Idx1 + 1);

            if (N) {
                int x = Idx3, y;
                do {
                    y = x + std::min(temp_LTLcacheCapacity / (2 * (Idx2 - Idx1 + 1)) - 1, Idx4 - x); // 2 is for until only (lUr and rUl)
                    N = (y - x + 1) * (Idx2 - Idx1 + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (U) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLcost, allLTLs, lastIdx, 2 * N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<Op::Until, hash_set_t> << <Blc, 1024 >> > (
                        Idx1, Idx2, x, y, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
                        d_LTLcache, d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx,
                        cHashSet, iHashSet, d_FinalLTLIdx
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalLTLIdx, d_FinalLTLIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allLTLs += 2 * N;
                    if (*FinalLTLIdx != -1) { startPoints[(LTLcost + 1) * 7] = INT_MAX; goto exitEnumeration; }
                    if (!onTheFly) storeUniqueLTLs(2 * N, lastIdx, numOfTraces, LTLcacheCapacity, onTheFly, d_LTLcache, d_temp_LTLcache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < Idx4);
            }

        }
        startPoints[(LTLcost + 1) * 7] = lastIdx;

        if (lastRound) break;
        if (onTheFly && shortageCost == -1) shortageCost = LTLcost;

    }

    if (LTLcost == maxCost + 1) LTLcost--;

exitEnumeration:

    std::string output;
    bool isLTLFromTempLTLcache = true;

    if (*FinalLTLIdx != -1) {
        output = LTLtoString(isLTLFromTempLTLcache, *FinalLTLIdx, lastIdx, alphabet, startPoints,
            d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
    } else {
        output = "not_found";
    }

    /*
    const int NNN = 1000;
    auto *cache2 = new UINT_64[numOfTraces * NNN];
    auto *left_indices = new int[NNN];
    auto *right_indices = new int[NNN];
    checkCuda( cudaMemcpy(cache2, d_LTLcache, numOfTraces * NNN * sizeof(UINT_64), cudaMemcpyDeviceToHost) );
    checkCuda( cudaMemcpy(left_indices, d_leftIdx, NNN * sizeof(int), cudaMemcpyDeviceToHost) );
    checkCuda( cudaMemcpy(right_indices, d_rightIdx, NNN * sizeof(int), cudaMemcpyDeviceToHost) );

    for (int i = 0; i < NNN; ++i) {
        std::cout << i << " --> \t \t";
        std::string out = LTLtoString(false, i, lastIdx, alphabet, startPoints,
        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
        std::cout << out << "\t \t";
        for (int j = 0; j < numOfTraces; ++j) {
            std::cout << cache2[i * numOfTraces + j] << "\t";
        }
        std::cout << std::endl;
    }
    */

    // Cleanup
    cudaFree(d_LTLcache);
    cudaFree(d_FinalLTLIdx);
    cudaFree(d_temp_LTLcache);
    cudaFree(d_leftIdx);
    cudaFree(d_rightIdx);
    cudaFree(d_temp_leftIdx);
    cudaFree(d_temp_rightIdx);

    return output;

}

// Reading the input file
bool readFile(
    const std::string& fileName,
    std::set<char>& alphabet,
    std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>>& pos,
    std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>>& neg)
{

    std::ifstream file(fileName);
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        std::string line;
        char ch;
        bool foundNewline = false;
        while (!foundNewline && file.tellg() > 0) {
            file.seekg(-2, std::ios::cur);
            file.get(ch);
            if (ch == '\n') foundNewline = true;
        }
        std::getline(file, line);
        std::string alpha;
        for (auto& c : line)
            if (c >= 'a' && c <= 'z') {
                alphabet.insert(c);
                alpha += c;
            }
        file.seekg(0, std::ios::beg);
        while (std::getline(file, line)) {
            std::vector<std::string> ptrace;
            std::vector<std::string> ctrace;
            bool in_cycle = false;
            if (line != "---") {
                std::string token;
                int j{};
                for (char& c : line) {
                    if (c == ';') {
                        if (in_cycle) ctrace.push_back(token);
                        else ptrace.push_back(token);
                        token = ""; j = 0;
                    } else if (c == '|') in_cycle = true;
                    else if (c == ',') continue;
                    else {
                        if (c == '1') token += alpha[j];
                        j++;
                    }
                }
                ctrace.push_back(token);
                if (ptrace.empty()) ptrace = ctrace; // Copy cycle into prefix if prefix is empty
                pos.push_back({ ptrace, ctrace });
            } else break;
        }
        while (std::getline(file, line)) {
            std::vector<std::string> ptrace;
            std::vector<std::string> ctrace;
            bool in_cycle = false;
            if (line != "---") {
                std::string token;
                int j{};
                for (char& c : line) {
                    if (c == ';') {
                        if (in_cycle) ctrace.push_back(token);
                        else ptrace.push_back(token);
                        token = ""; j = 0;
                    } else if (c == '|') in_cycle = true;
                    else if (c == ',') continue;
                    else {
                        if (c == '1') token += alpha[j];
                        j++;
                    }
                }
                ctrace.push_back(token);
                if (ptrace.empty()) ptrace = ctrace; // Copy cycle into prefix if prefix is empty
                neg.push_back({ ptrace, ctrace });
            } else break;
        }
        file.close();
        return true;
    } else printf("Failed to open the input file.\n");

    return false;

}

int main(int argc, char* argv[]) {

    // -----------------
    // Reading the input
    // -----------------

    if (argc != 13) {
        printf("Arguments should be in the form of\n");
        printf("-----------------------------------------------------------------\n");
        printf("%s <input_file_address> <c1> <c2> <c3> <c4> <c5> <c6> <c7> <c8> <maxCost> <RlxUnqChkTyp> <NegType>\n", argv[0]);
        printf("-----------------------------------------------------------------\n");
        printf("\nFor example\n");
        printf("-----------------------------------------------------------------\n");
        printf("%s ./input.txt 1 1 1 1 1 1 1 1 500 3 2\n", argv[0]);
        printf("-----------------------------------------------------------------\n");
        return 0;
    }

    bool argError = false;
    for (int i = 2; i < argc - 2; ++i) {
        if (std::atoi(argv[i]) <= 0 || std::atoi(argv[i]) > SHRT_MAX) {
            printf("Argument number %d, \"%s\", should be a positive short integer.\n", i, argv[i]);
            argError = true;
        }
    }
    if (std::atoi(argv[11]) < 1 || std::atoi(argv[11]) > 3) {
        printf("Argument number 11, RlxUnqChkTyp = \"%s\", should be 1, 2, or 3.\n", argv[11]);
        argError = true;
    }
    if (std::atoi(argv[12]) < 1 || std::atoi(argv[12]) > 2) {
        printf("Argument number 12, NegType = \"%s\", should be 1, or 2.\n", argv[12]);
        argError = true;
    }

    if (argError) return 0;

    std::string fileName = argv[1];
    std::set<char> alphabet;
    std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> pos, neg;
    if (!readFile(fileName, alphabet, pos, neg)) return 0;
    unsigned short costFun[8];
    for (int i = 0; i < 8; i++)
        costFun[i] = std::atoi(argv[i + 2]);
    unsigned short maxCost = std::atoi(argv[10]);
    unsigned int RlxUnqChkTyp = std::atoi(argv[11]);
    unsigned int NegType = std::atoi(argv[12]);

    // --------------------------------------
    // Linear Temporal Logic Inference (LTLI)
    // --------------------------------------

#ifdef MEASUREMENT_MODE
    auto start = std::chrono::high_resolution_clock::now();
#endif

    std::uint64_t allLTLs{}; int LTLcost = costFun[0];
    std::string output = LTLI(costFun, maxCost, RlxUnqChkTyp, NegType, alphabet, LTLcost, allLTLs, pos, neg);
    if (output == "see_the_error") return 0;

#ifdef MEASUREMENT_MODE
    auto stop = std::chrono::high_resolution_clock::now();
#endif

    // -------------------
    // Printing the output
    // -------------------

    printf("\nPositive: \n");
    for (const auto& trace : pos) {
        printf("\t");
        for (const std::string& t : trace.first) {
            std::string s;
            for (const char& c : t) {
                s += c; s += ", ";
            }
            printf("{%s}\t", s.substr(0, s.length() - 2).c_str());
        }
        printf("|||\t");
        for (const std::string& t : trace.second) {
            std::string s;
            for (const char& c : t) {
                s += c; s += ", ";
            }
            printf("{%s}\t", s.substr(0, s.length() - 2).c_str());
        }
        printf("\n");
    }

    printf("\nNegative: \n");
    for (const auto& trace : neg) {
        printf("\t");
        for (const std::string& t : trace.first) {
            std::string s;
            for (const char& c : t) {
                s += c; s += ", ";
            }
            printf("{%s}\t", s.substr(0, s.length() - 2).c_str());
        }
        printf("|||\t");
        for (const std::string& t : trace.second) {
            std::string s;
            for (const char& c : t) {
                s += c; s += ", ";
            }
            printf("{%s}\t", s.substr(0, s.length() - 2).c_str());
        }
        printf("\n");
    }

    printf("\nCost Function: p:%u, ~:%u, &:%u, |:%u, X:%u, F:%u, G:%u, U:%u",
        costFun[0], costFun[1], costFun[2], costFun[3], costFun[4], costFun[5], costFun[6], costFun[7]);
    printf("\nNumber of Traces: %d", static_cast<int>(pos.size() + neg.size()));
#ifdef MEASUREMENT_MODE
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("\nNumber of All LTLs: %lu", allLTLs);
    printf("\nCost of Final LTL: %d", LTLcost);
    printf("\nRunning Time: %f s", (double)duration * 0.000001);
#endif
    printf("\n\nLTL: \"%s\"\n", output.c_str());

    return 0;

}