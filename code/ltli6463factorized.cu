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

__constant__ char d_traceLen[maxNumOfTraces];

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

        // we need an relaxed uniqueness check

        switch (RlxUnqChkTyp) {

        case 1: {

            const int stride = lenSum / 126;

            int j = 0;
            for (int i = 0; i < numOfTraces; ++i) {
                for (int k = 0; k < d_traceLen[i]; k += stride, ++j) {
                    if (j < 63) {
                        if (CS[i] & ((UINT_64)1 << k)) lCS |= (UINT_64)1 << j;
                    }
                    else if (j < 126) {
                        if (CS[i] & ((UINT_64)1 << k)) hCS |= (UINT_64)1 << (j - 63);
                    }
                    else break;
                }
            }

            break;
        }

        case 2: {

            int j = 0;
            for (int i = 0; i < numOfTraces; ++i) {
                UINT_64 bitPtr = 1;
                int maxbitsForThisTrace = (126 * d_traceLen[i] + lenSum) / lenSum;
                for (int k = 0; k < maxbitsForThisTrace; ++k, ++j, bitPtr <<= 1) {
                    if (j < 63) {
                        if (CS[i] & bitPtr) lCS |= (UINT_64)1 << j;
                    }
                    else if (j < 126) {
                        if (CS[i] & bitPtr) hCS |= (UINT_64)1 << (j - 63);
                    }
                    else break;
                }
            }

            break;
        }

        case 3: {

            for (int i = 0; i < numOfTraces; ++i) {
                UINT_64 x = CS[i];
                x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
                x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
                x = x ^ (x >> 31);
                if (i < numOfTraces / 2) hCS ^= x; else lCS ^= x;
            }

            break;
        }

        }

    }
    else {

        // the result will be minimal

        int j = 0;
        for (int i = 0; i < numOfTraces; ++i) {
            UINT_64 bitPtr = 1;
            for (int k = 0; k < d_traceLen[i]; ++k, ++j, bitPtr <<= 1) {
                if (j < 63) {
                    if (CS[i] & bitPtr) lCS |= (UINT_64)1 << j;
                }
                else if (j < 126) {
                    if (CS[i] & bitPtr) hCS |= (UINT_64)1 << (j - 63);
                }
                else break;
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
    UINT_64 CS[maxNumOfTraces];

    for (int i = 0; i < numOfTraces; ++i)
        CS[i] = d_LTLcache[tid * numOfTraces + i];

    UINT_64 hCS{}, lCS{};
    makeRlxUnqChkCSs(CS, hCS, lCS, numOfTraces, RlxUnqChkTyp, lenSum);

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    int H = cHashSet.insert(hCS, group); int L = cHashSet.insert(lCS, group);
    H = (H > 0) ? H : -H; L = (L > 0) ? L : -L;
    UINT_64 HL = H; HL <<= 32; HL |= L;
    iHashSet.insert(HL, group);

}

__device__ void applyOperator(
    char op,
    UINT_64* CS,
    UINT_64* d_LTLcache,
    int ldx, int rdx,
    int numOfTraces)
{

    switch (op) {
    case '~': {
        for (int i = 0; i < numOfTraces; ++i) {
            UINT_64 negationFixer = ((UINT_64)1 << d_traceLen[i]) - 1;
            CS[i] = ~d_LTLcache[ldx * numOfTraces + i] & negationFixer;
        }
        break;
    }
    case '&': {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * numOfTraces + i] & d_LTLcache[rdx * numOfTraces + i];
        }
        break;
    }
    case '|': {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * numOfTraces + i] | d_LTLcache[rdx * numOfTraces + i];
        }
        break;
    }
    case 'X': {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * numOfTraces + i] >> 1;
        }
        break;
    }
    case 'F': {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * numOfTraces + i];
            CS[i] |= CS[i] >> 1; CS[i] |= CS[i] >> 2; CS[i] |= CS[i] >> 4;
            CS[i] |= CS[i] >> 8; CS[i] |= CS[i] >> 16; CS[i] |= CS[i] >> 32;
        }
        break;
    }
    case 'G': {
        for (int i = 0; i < numOfTraces; ++i) {
            CS[i] = d_LTLcache[ldx * numOfTraces + i];
            UINT_64 cs = ~CS[i] & (((UINT_64)1 << d_traceLen[i]) - 1);
            cs |= cs >> 1; cs |= cs >> 2; cs |= cs >> 4;
            cs |= cs >> 8; cs |= cs >> 16; cs |= cs >> 32;
            CS[i] &= ~cs;
        }
        break;
    }
    case 'U': {
        for (int i = 0; i < numOfTraces; ++i) {
            UINT_64 l = d_LTLcache[ldx * numOfTraces + i];
            UINT_64 r = d_LTLcache[rdx * numOfTraces + i];
            r |= l & (r >> 1);  l &= l >> 1;
            r |= l & (r >> 2);  l &= l >> 2;
            r |= l & (r >> 4);  l &= l >> 4;
            r |= l & (r >> 8);  l &= l >> 8;
            r |= l & (r >> 16); l &= l >> 16;
            r |= l & (r >> 32);
            CS[i] = r;
        }
        break;
    }
    default:
        printf("Opérateur non pris en charge: %c\n", op);
        break;
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
    for (int i = 0; found && i < numOfP; ++i) if (!(CS[i] & 1)) found = false;
    for (int i = numOfP; found && i < numOfTraces; ++i) if (CS[i] & 1) found = false;

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

        for (int i = 0; i < numOfTraces; ++i)
            d_temp_LTLcache[tid * numOfTraces + i] = CS[i];
        d_temp_leftIdx[tid] = ldx; d_temp_rightIdx[tid] = rdx;

        bool found = true;
        for (int i = 0; found && i < numOfP; ++i) if (!(CS[i] & 1)) found = false;
        for (int i = numOfP; found && i < numOfTraces; ++i) if (CS[i] & 1) found = false;
        if (found) atomicCAS(d_FinalLTLIdx, -1, tid);

    }
    else {

        for (int i = 0; i < numOfTraces; ++i)
            d_temp_LTLcache[tid * numOfTraces + i] = (UINT_64)-1;
        d_temp_leftIdx[tid] = -1; d_temp_rightIdx[tid] = -1;

    }

}

template<class hash_set_t>
__global__ void processOperator(
    const char op,
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

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int numOfTraces = numOfP + numOfN;
    bool isUnary = (op == '~' || op == 'X' || op == 'F' || op == 'G');
    int maxTid = isUnary ? (idx2 - idx1 + 1) : ((idx4 - idx3 + 1) * (idx2 - idx1 + 1));

    if (tid < maxTid) {

        int ldx = isUnary ? (idx1 + tid) : (idx1 + tid / (idx4 - idx3 + 1));
        int rdx = isUnary ? 0 : (idx3 + tid % (idx4 - idx3 + 1));

        if (op == 'U') {

            UINT_64 CS1[maxNumOfTraces], CS2[maxNumOfTraces];
            applyOperator('U', CS1, d_LTLcache, ldx, rdx, numOfTraces);
            applyOperator('U', CS2, d_LTLcache, rdx, ldx, numOfTraces);

            if (onTheFly) {
                processOnTheFly(
                    CS1, tid * 2, numOfTraces, numOfP, ldx, rdx,
                    d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
                );
                processOnTheFly(
                    CS2, tid * 2 + 1, numOfTraces, numOfP, rdx, ldx,
                    d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
                );
            }

            else {
                bool CS1_is_unique =
                    processUniqueCS(CS1, numOfTraces, RlxUnqChkTyp, lenSum, cHashSet, iHashSet);
                bool CS2_is_unique =
                    processUniqueCS(CS2, numOfTraces, RlxUnqChkTyp, lenSum, cHashSet, iHashSet);
                insertInCache(
                    CS1_is_unique, CS1, tid * 2, numOfTraces, numOfP, ldx, rdx,
                    d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
                );
                insertInCache(
                    CS2_is_unique, CS2, tid * 2 + 1, numOfTraces, numOfP, rdx, ldx,
                    d_temp_LTLcache, d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
                );
            }

        }

        else {

            UINT_64 CS[maxNumOfTraces];
            applyOperator(op, CS, d_LTLcache, ldx, rdx, numOfTraces);

            if (onTheFly) {
                processOnTheFly(
                    CS, tid, numOfTraces, numOfP, ldx, rdx,
                    d_temp_leftIdx, d_temp_rightIdx, d_FinalLTLIdx
                );
            }

            else {
                bool CS_is_unique =
                    processUniqueCS(CS, numOfTraces, RlxUnqChkTyp, lenSum, cHashSet, iHashSet);
                insertInCache(
                    CS_is_unique, CS, tid, numOfTraces, numOfP, ldx, rdx,
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
    }
    else {
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
    thrust::device_ptr<UINT_64> d_LTLcache_ptr(d_LTLcache + numOfTraces * lastIdx);
    thrust::device_ptr<UINT_64> d_temp_LTLcache_ptr(d_temp_LTLcache);
    thrust::device_ptr<int> d_leftIdx_ptr(d_leftIdx + lastIdx);
    thrust::device_ptr<int> d_rightIdx_ptr(d_rightIdx + lastIdx);
    thrust::device_ptr<int> d_temp_leftIdx_ptr(d_temp_leftIdx);
    thrust::device_ptr<int> d_temp_rightIdx_ptr(d_temp_rightIdx);

    new_end_ptr = // end of d_temp_LTLcache
        thrust::remove(d_temp_LTLcache_ptr, d_temp_LTLcache_ptr + numOfTraces * N, (UINT_64)-1);
    thrust::remove(d_temp_leftIdx_ptr, d_temp_leftIdx_ptr + N, -1);
    thrust::remove(d_temp_rightIdx_ptr, d_temp_rightIdx_ptr + N, -1);

    // It stores all (or a part of) unique CSs until language cahce gets full
    // If language cache gets full, it makes onTheFly mode on
    int numberOfNewUniqueLTLs = static_cast<int>(new_end_ptr - d_temp_LTLcache_ptr) / numOfTraces;
    if (lastIdx + numberOfNewUniqueLTLs > LTLcacheCapacity) {
        N = LTLcacheCapacity - lastIdx;
        onTheFly = true;
    }
    else N = numberOfNewUniqueLTLs;

    thrust::copy_n(d_temp_LTLcache_ptr, numOfTraces * N, d_LTLcache_ptr);
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
    const std::vector<std::vector<std::string>> pos,
    const std::vector<std::vector<std::string>> neg) {

    // --------------------------------
    // Generating and checking alphabet
    // --------------------------------

    const int numOfP = pos.size();
    const int numOfN = neg.size();
    const int numOfTraces = numOfP + numOfN;

    int maxLenOfTraces{};
    auto* traceLen = new char[numOfTraces];

    int TLIdx{};
    int lenSum{};
    for (const auto& trace : pos) {
        lenSum += trace.size();
        traceLen[TLIdx++] = trace.size();
        if (trace.size() > maxLenOfTraces) maxLenOfTraces = trace.size();
    }
    for (const auto& trace : neg) {
        lenSum += trace.size();
        traceLen[TLIdx++] = trace.size();
        if (trace.size() > maxLenOfTraces) maxLenOfTraces = trace.size();
    }

    if (numOfTraces > maxNumOfTraces || maxLenOfTraces > sizeof(UINT_64) * 8 - 1) {
        printf("In this version, The input can have:\n");
        printf("1) At most %zu traces. It is currently %d.\n", maxNumOfTraces, numOfTraces);
        printf("2) Max(len(trace)) = %d. It is currently %d.\n", static_cast<int>(sizeof(UINT_64) * 8 - 1), maxLenOfTraces);
        return "see_the_error";
    }

    // Copying the length of traces into the constant memory
    checkCuda(cudaMemcpyToSymbol(d_traceLen, traceLen, numOfTraces * sizeof(char)));

    const int alphabetSize = static_cast<int>(alphabet.size());

    auto* LTLcache = new UINT_64[alphabetSize * numOfTraces];

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
            for (const auto& token : trace) {
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
        }
        for (const auto& trace : neg) {
            UINT_64 binTrace{};
            UINT_64 idx = 1;
            for (const auto& token : trace) {
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
        }
        allLTLs++; lastIdx++;
        if (found) return ch;
    }

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    // cost function
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

    const int LTLcacheCapacity = maxAllocationSize / (numOfTraces * sizeof(UINT_64)) * 1.5;
    const int temp_LTLcacheCapacity = LTLcacheCapacity / 2;

    // const int LTLcacheCapacity = 2000000;
    // const int temp_LTLcacheCapacity = 100000000;

    // 7 for ~, &, |, X, F, G, U
    int* startPoints = new int[(maxCost + 2) * 7]();
    startPoints[c1 * 7 + 6] = lastIdx;
    startPoints[(c1 + 1) * 7] = lastIdx;

    int* d_FinalLTLIdx;
    auto* FinalLTLIdx = new int[1]; *FinalLTLIdx = -1;
    checkCuda(cudaMalloc(&d_FinalLTLIdx, sizeof(int)));
    checkCuda(cudaMemcpy(d_FinalLTLIdx, FinalLTLIdx, sizeof(int), cudaMemcpyHostToDevice));

    UINT_64* d_LTLcache, * d_temp_LTLcache;
    int* d_leftIdx, * d_rightIdx, * d_temp_leftIdx, * d_temp_rightIdx;
    checkCuda(cudaMalloc(&d_leftIdx, LTLcacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_rightIdx, LTLcacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_leftIdx, temp_LTLcacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_rightIdx, temp_LTLcacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_LTLcache, LTLcacheCapacity * numOfTraces * sizeof(UINT_64)));
    checkCuda(cudaMalloc(&d_temp_LTLcache, temp_LTLcacheCapacity * numOfTraces * sizeof(UINT_64)));

    using hash_set_t = warpcore::HashSet<
        UINT_64,         // key type
        UINT_64(0) - 1,  // empty key
        UINT_64(0) - 2,  // tombstone key
        warpcore::probing_schemes::QuadraticProbing<warpcore::hashers::MurmurHash <UINT_64>>>;

    hash_set_t cHashSet(2 * LTLcacheCapacity);
    hash_set_t iHashSet(2 * LTLcacheCapacity);

    checkCuda(cudaMemcpy(d_LTLcache, LTLcache, alphabetSize * numOfTraces * sizeof(UINT_64), cudaMemcpyHostToDevice));
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


        // negation (~)
        // NegType = 1 is for negation of phi
        // NegType = 2 is for negation of char only
        if ((NegType == 1 && LTLcost - c2 >= c1) || (NegType == 2 && LTLcost - c2 == c1))
            if (LTLcost - c2 >= c1) {

                int Idx1 = startPoints[(LTLcost - c2) * 7];
                int Idx2 = startPoints[(LTLcost - c2 + 1) * 7] - 1;
                int N = Idx2 - Idx1 + 1;

                if (N) {
                    int x = Idx1, y;
                    do {
                        y = x + std::min(temp_LTLcacheCapacity - 1, Idx2 - x);
                        N = (y - x + 1);
#ifndef MEASUREMENT_MODE
                        printf("Cost %-2d | (~) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                            LTLcost, allLTLs, lastIdx, N);
#endif
                        int Blc = (N + 1023) / 1024;
                        processOperator<hash_set_t> << <Blc, 1024 >> > (
                            '~', x, y, 0, 0, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
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

        // intersection (&)
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
                    processOperator<hash_set_t> << <Blc, 1024 >> > (
                        '&', Idx1, Idx2, x, y, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
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

        // union (|)
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
                    processOperator<hash_set_t> << <Blc, 1024 >> > (
                        '|', Idx1, Idx2, x, y, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
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

        // next (X)
        if (LTLcost - c5 >= c1) {

            int Idx1 = startPoints[(LTLcost - c5) * 7];
            int Idx2 = startPoints[(LTLcost - c5 + 1) * 7] - 1;
            int N = Idx2 - Idx1 + 1;

            if (N) {
                int x = Idx1, y;
                do {
                    y = x + std::min(temp_LTLcacheCapacity - 1, Idx2 - x);
                    N = (y - x + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (X) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLcost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<hash_set_t> << <Blc, 1024 >> > (
                        'X', x, y, 0, 0, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
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

        // finally (F)
        if (LTLcost - c6 >= c1) {

            int Idx1 = startPoints[(LTLcost - c6) * 7];
            int Idx2 = startPoints[(LTLcost - c6 + 1) * 7] - 1;
            int N = Idx2 - Idx1 + 1;

            if (N) {
                int x = Idx1, y;
                do {
                    y = x + std::min(temp_LTLcacheCapacity - 1, Idx2 - x);
                    N = (y - x + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (F) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLcost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<hash_set_t> << <Blc, 1024 >> > (
                        'F', x, y, 0, 0, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
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

        // globally (G)
        if (LTLcost - c7 >= c1) {

            int Idx1 = startPoints[(LTLcost - c7) * 7];
            int Idx2 = startPoints[(LTLcost - c7 + 1) * 7] - 1;
            int N = Idx2 - Idx1 + 1;

            if (N) {
                int x = Idx1, y;
                do {
                    y = x + std::min(temp_LTLcacheCapacity - 1, Idx2 - x);
                    N = (y - x + 1);
#ifndef MEASUREMENT_MODE
                    printf("Cost %-2d | (G) | AllLTLs: %-11lu | StoredLTLs: %-10d | ToBeChecked: %-10d \n",
                        LTLcost, allLTLs, lastIdx, N);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<hash_set_t> << <Blc, 1024 >> > (
                        'G', x, y, 0, 0, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
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

        // until (U)
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
                    processOperator<hash_set_t> << <Blc, 1024 >> > (
                        'U', Idx1, Idx2, x, y, numOfP, numOfN, RlxUnqChkTyp, lenSum, onTheFly,
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

    }
    else {

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

    // cleanup
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
    std::vector<std::vector<std::string>>& pos,
    std::vector<std::vector<std::string>>& neg)
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
            std::vector<std::string> trace;
            if (line != "---") {
                std::string token;
                int j{};
                for (auto& c : line) {
                    if (c == ';') {
                        trace.push_back(token);
                        token = "";
                        j = 0;
                    }
                    else if (c == ',') continue;
                    else {
                        if (c == '1') token += alpha[j];
                        j++;
                    }
                }
                trace.push_back(token);
                pos.push_back(trace);
            }
            else break;
        }
        while (std::getline(file, line)) {
            std::vector<std::string> trace;
            if (line != "---") {
                std::string token;
                int j{};
                for (auto& c : line) {
                    if (c == ';') {
                        trace.push_back(token);
                        token = "";
                        j = 0;
                    }
                    else if (c == ',') continue;
                    else {
                        if (c == '1') token += alpha[j];
                        j++;
                    }
                }
                trace.push_back(token);
                neg.push_back(trace);
            }
            else break;
        }
        file.close();
        return true;
    }
    else printf("Failed to open the input file.\n");

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
    std::vector<std::vector<std::string>> pos, neg;
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
        for (const auto& t : trace) {
            std::string s;
            for (const auto& c : t) {
                s += c; s += ", ";
            }
            printf("{%s}\t", s.substr(0, s.length() - 2).c_str());
        }
        printf("\n");
    }

    printf("\nNegative: \n");
    for (const auto& trace : neg) {
        printf("\t");
        for (const auto& t : trace) {
            std::string s;
            for (const auto& c : t) {
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