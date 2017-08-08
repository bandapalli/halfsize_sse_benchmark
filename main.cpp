#include <tmmintrin.h>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <chrono>
#include <algorithm>
#include <iostream>

#if defined(_MSC_VER)
#define __restrict__ __restrict
#elif !defined(__GNUC__)
#define __restrict__ /* no-op */
#endif

namespace reference
{
    /**
     * Calculates the average of two rows of gray8 pixels by averaging four pixels.
     */
    void average2Rows(const uint8_t* __restrict__ row1, const uint8_t* __restrict__ row2, uint8_t* __restrict__ dst, size_t size)
    {
        for (int i = 0; i < size - 1; i += 2)
            *(dst++) = ((row1[i] + row1[i + 1] + row2[i] + row2[i + 1]) / 4) & 0xFF;
    }
}

// The original optimization attempt presented here:
// https://stackoverflow.com/questions/45541530/fastest-downscaling-of-8bit-gray-image-with-sse
namespace original 
{
    /*
     * Input: 16 8-bit values A-P
     * Output 8 8-bit values (A+B)/2, (C+D)/2, ..., (O+P)/2
     */
    inline __m128i avg16Bytes(const __m128i& ABCDEFGHIJKLMNOP)
    {
        static const __m128i  zero = _mm_setzero_si128(); 

        __m128i ABCDEFGH  = _mm_unpacklo_epi8(ABCDEFGHIJKLMNOP, zero);
        __m128i IJKLMNOP  = _mm_unpackhi_epi8(ABCDEFGHIJKLMNOP, zero);

        __m128i AIBJCKDL = _mm_unpacklo_epi16( ABCDEFGH, IJKLMNOP );
        __m128i EMFNGOHP = _mm_unpackhi_epi16( ABCDEFGH, IJKLMNOP );

        __m128i AEIMBFJN = _mm_unpacklo_epi16( AIBJCKDL, EMFNGOHP );
        __m128i CGKODHLP = _mm_unpackhi_epi16( AIBJCKDL, EMFNGOHP );

        __m128i ACEGIKMO = _mm_unpacklo_epi16( AEIMBFJN, CGKODHLP );
        __m128i BDFHJLNP = _mm_unpackhi_epi16( AEIMBFJN, CGKODHLP );

        return _mm_avg_epu8(ACEGIKMO, BDFHJLNP);
    }

    /*
     * Calculates the average of two rows of gray8 pixels by averaging four pixels.
     */
    void average2Rows(const uint8_t* __restrict__ src1, const uint8_t* __restrict__ src2, uint8_t* __restrict__ dst, size_t size)
    {
        size /= 2;
        for (size_t i = 0; i < size - 15; i += 16)
        {
            __m128i tl = _mm_load_si128((__m128i *)&src1[i * 2]);
            __m128i tr = _mm_load_si128((__m128i *)&src1[i * 2 + 16]);
            __m128i bl = _mm_load_si128((__m128i *)&src2[i * 2]);
            __m128i br = _mm_load_si128((__m128i *)&src2[i * 2 + 16]);

            // Average the first 16 values of src1 and src2:
            __m128i left = _mm_avg_epu8(tl, bl);

            // Average the following 16 values of src1 and src2:
            __m128i right = _mm_avg_epu8(tr, br);
            
            // Now pairwise average the 32 values in left and right:
            __m128i avg = _mm_packus_epi16(avg16Bytes(left), avg16Bytes(right));
            _mm_store_si128((__m128i *)(dst + i), avg);
        }
    }
}

// The solution given by user Paul:
// https://stackoverflow.com/a/45542669/396803
namespace paul
{
    void average2Rows(const uint8_t* __restrict__ src1, const uint8_t* __restrict__ src2, uint8_t* __restrict__ dst, size_t size)
    {
        const __m128i vk1 = _mm_set1_epi8(1);
        
        size /= 2;
        for (size_t i = 0; i < size - 15; i += 16)
        {
            __m128i tl = _mm_load_si128((__m128i *)&src1[i * 2]);
            __m128i tr = _mm_load_si128((__m128i *)&src1[i * 2 + 16]);
            __m128i bl = _mm_load_si128((__m128i *)&src2[i * 2]);
            __m128i br = _mm_load_si128((__m128i *)&src2[i * 2 + 16]);

            __m128i w0 = _mm_maddubs_epi16(tl, vk1);        // unpack and horizontal add
            __m128i w1 = _mm_maddubs_epi16(tr, vk1);
            __m128i w2 = _mm_maddubs_epi16(bl, vk1);
            __m128i w3 = _mm_maddubs_epi16(br, vk1);

            w0 = _mm_add_epi16(w0, w2);                     // vertical add
            w1 = _mm_add_epi16(w1, w3);

            w0 = _mm_srli_epi16(w0, 2);                     // divide by 4
            w1 = _mm_srli_epi16(w1, 2);

            w0 = _mm_packus_epi16(w0, w1);                  // pack

            _mm_store_si128((__m128i *)(dst + i), w0);
        }
    }
}

// The solution given by user Peter Cordes:
// https://stackoverflow.com/a/45564565/396803
namespace peterCordes_V1
{
    void average2Rows(const uint8_t* __restrict__ src1, const uint8_t* __restrict__ src2, uint8_t* __restrict__ dst, size_t size)
    {
        size /= 2;
        for (size_t i = 0; i < size - 15; i += 16)
        {
            __m128i tl = _mm_load_si128((__m128i *)&src1[i * 2]);
            __m128i tr = _mm_load_si128((__m128i *)&src1[i * 2 + 16]);
            __m128i bl = _mm_load_si128((__m128i *)&src2[i * 2]);
            __m128i br = _mm_load_si128((__m128i *)&src2[i * 2 + 16]);

            __m128i left = _mm_avg_epu8(tl, bl);
            __m128i right = _mm_avg_epu8(tr, br);

            __m128i l_odd = _mm_srli_epi16(left, 8);   // line up horizontal pairs
            __m128i r_odd = _mm_srli_epi16(right, 8);

            __m128i l_avg = _mm_avg_epu8(left, l_odd);  // leaves garbage in the high halves
            __m128i r_avg = _mm_avg_epu8(right, r_odd);

            l_avg = _mm_and_si128(l_avg, _mm_set1_epi16(0x00FF));
            r_avg = _mm_and_si128(r_avg, _mm_set1_epi16(0x00FF));

            __m128i avg = _mm_packus_epi16(l_avg, r_avg);          // pack
            _mm_store_si128((__m128i *)(dst + i), avg);
        }
    }
}

namespace peterCordes_V2
{
    void average2Rows(const uint8_t* __restrict__ src1, const uint8_t* __restrict__ src2, uint8_t* __restrict__ dst, size_t size)
    {
    size /= 2;
    for (size_t i = 0; i < size - 15; i += 16)
    {
        __m128i v0 = _mm_load_si128((__m128i *)&src1[i*2]);
        __m128i v1 = _mm_load_si128((__m128i *)&src1[i*2 + 16]);
        __m128i v2 = _mm_load_si128((__m128i *)&src2[i*2]);
        __m128i v3 = _mm_load_si128((__m128i *)&src2[i*2 + 16]);

        __m128i left  = _mm_avg_epu8(v0, v2);
        __m128i right = _mm_avg_epu8(v1, v3);

        __m128i l_odd  = _mm_srli_epi16(left, 8);   // line up horizontal pairs
        __m128i r_odd  = _mm_srli_epi16(right, 8);

        __m128i l_avg = _mm_avg_epu8(left, l_odd);  // leaves garbage in the high halves
        __m128i r_avg = _mm_avg_epu8(right, r_odd);

        l_avg = _mm_and_si128(l_avg, _mm_set1_epi16(0x00FF));
        r_avg = _mm_and_si128(r_avg, _mm_set1_epi16(0x00FF));

        __m128i avg   = _mm_packus_epi16(l_avg, r_avg);          // pack
        _mm_storeu_si128((__m128i *)&dst[i], avg);
    }
}
}
// One implementation suggested by user https://stackoverflow.com/users/1196549/yves-daoust:
namespace yves_exact
{
    /*
     * row1: 16 8-bit values A-P
     * row2: 16 8-bit values a-p
     * Output 16 8-bit values (A+B+a+b)/4, (C+D+c+d)/4, ..., (O+P+o+p)/4
     */
    inline __m128i avg16BytesX2(const __m128i& row1, const __m128i& row2)
    {
        static const __m128i Mask_01010101 = _mm_set_epi32(0xFF00FF, 0xFF00FF, 0xFF00FF, 0xFF00FF);

        // Right shift 8 bits and add each row to itself:
        __m128i sum_row1 = _mm_add_epi16(_mm_and_si128(row1, Mask_01010101), _mm_and_si128(_mm_srli_epi64 (row1, 8), Mask_01010101));
        __m128i sum_row2 = _mm_add_epi16(_mm_and_si128(row2, Mask_01010101), _mm_and_si128(_mm_srli_epi64 (row2, 8), Mask_01010101));

        // Add the two rows:
        __m128i sum = _mm_add_epi16(sum_row1, sum_row2);
                
        // Divide by 4:
        sum = _mm_srli_epi16(sum, 2);

        // Mask out every other byte:
        sum =  _mm_and_si128(sum, Mask_01010101);

        return sum;
    };
    void average2Rows(const uint8_t* __restrict__ src1, const uint8_t* __restrict__ src2, uint8_t* __restrict__ dst, size_t size)
    {
        size /= 2;
        for (size_t i = 0; i < size - 15; i += 16)
        {
            __m128i tl = _mm_load_si128((__m128i *)&src1[i * 2]);
            __m128i tr = _mm_load_si128((__m128i *)&src1[i * 2 + 16]);
            __m128i bl = _mm_load_si128((__m128i *)&src2[i * 2]);
            __m128i br = _mm_load_si128((__m128i *)&src2[i * 2 + 16]);

            __m128i avg = _mm_packus_epi16(avg16BytesX2(tl, bl), avg16BytesX2(tr, br));
            _mm_store_si128((__m128i *)(dst + i), avg);
        }
    }
}

// Another version of Yves Daousts suggestion:
namespace yves_inexact
{
    /*
     * row1: 16 8-bit values A-P
     * row2: 16 8-bit values a-p
     * Output 16 8-bit values (A+B+a+b)/4, (C+D+c+d)/4, ..., (O+P+o+p)/4
     */
    __m128i avg16BytesX2(const __m128i& row1, const __m128i& row2)
    {
        static const __m128i Mask_01010101 = _mm_set_epi32(0xFF00FF, 0xFF00FF, 0xFF00FF, 0xFF00FF);

        // Right shift 8 bits and add each row to itself:
        __m128i sum_row1 = _mm_avg_epu8(row1, _mm_srli_epi64(row1, 8));
        __m128i sum_row2 = _mm_avg_epu8(row2, _mm_srli_epi64(row2, 8));

        // Add the two rows:
        __m128i sum = _mm_avg_epu8(sum_row1, sum_row2);

        // Mask out every other byte:
        sum = _mm_and_si128(sum, Mask_01010101);

        return sum;
    };
    void average2Rows(const uint8_t* __restrict__ src1, const uint8_t* __restrict__ src2, uint8_t* __restrict__ dst, size_t size)
    {
        size /= 2;
        for (size_t i = 0; i < size - 15; i += 16)
        {
            __m128i tl = _mm_load_si128((__m128i *)&src1[i * 2]);
            __m128i tr = _mm_load_si128((__m128i *)&src1[i * 2 + 16]);
            __m128i bl = _mm_load_si128((__m128i *)&src2[i * 2]);
            __m128i br = _mm_load_si128((__m128i *)&src2[i * 2 + 16]);

            __m128i avg = _mm_packus_epi16(avg16BytesX2(tl, bl), avg16BytesX2(tr, br));
            _mm_store_si128((__m128i *)(dst + i), avg);
        }
    }
}

// This function just samples every fourth pixel:
namespace subsample
{
    /*
     * Downsamples two rows of gray8 pixels by sampling one out of every four pixels.
     */
    void average2Rows(const uint8_t* __restrict__ src1, const uint8_t* __restrict__ src2, uint8_t* __restrict__ dst, size_t size)
    {
        size /= 2;
        for (size_t i = 0; i < size - 15; i += 16)
        {
            __m128i tl = _mm_load_si128((__m128i *)&src1[i * 2]);
            __m128i br = _mm_load_si128((__m128i *)&src2[i * 2 + 16]);
            __m128i avg = _mm_packus_epi16(_mm_srli_epi16(tl, 8), _mm_srli_epi16(br, 8));
            _mm_store_si128((__m128i *)(dst + i), avg);
        }
    }
}

typedef void (*Function)(const uint8_t* __restrict__, const uint8_t* __restrict__, uint8_t* __restrict__, size_t); 

struct BenchMarkResult
{
    int maxError = 0;
    double time_us = 0;
};

BenchMarkResult testAndBenchmark(Function fun)
{
    BenchMarkResult result;
    const int n = 1024;

    alignas(32) uint8_t src1[n];
    alignas(32) uint8_t src2[n];
    alignas(32) uint8_t dest_ref[n / 2];
    alignas(32) uint8_t dest_test[n / 2];

    for (int i = 0; i < n; ++i)
    {
        src1[i] = rand()%256;
        src2[i] = rand()%256;
    }
    
    for (int i = 0; i < n / 2; ++i)
    {
        dest_ref[i] = 0xaa;
        dest_test[i] = 0x55;
    }

    reference::average2Rows(src1, src2, dest_ref, n);
    using namespace std::chrono;
    
#ifdef NDEBUG
    static const int REPS = 1000000;
#else
    static const int REPS = 100;
#endif

    auto begin = high_resolution_clock::now();
    for (int i = 0; i<REPS; i++)
        fun(src1, src2, dest_test, n);
    
    auto elapsed = high_resolution_clock::now()-begin;
    result.time_us = duration_cast<microseconds>(elapsed).count()*1.0/REPS;

    for (int i = 0; i < n / 2; ++i)
        if (int diff = std::abs(dest_test[i] - dest_ref[i]))
            result.maxError = std::max(diff, result.maxError);

    return result;
}

int main(int argc, char* argv[])
{   
    std::pair<Function, const char*> tests[] = {
        {reference::average2Rows, "Naive"}, 
        {original::average2Rows, "Bjorn"}, 
        {yves_exact::average2Rows,  "Yves Daoust v1"}, 
        {yves_inexact::average2Rows,  "Yves Daoust v2"}, 
        {peterCordes_V1::average2Rows,  "Peter v1"}, 
        {peterCordes_V2::average2Rows,  "Peter v2"}, 
        {paul::average2Rows,  "Paul"}, 
        {subsample::average2Rows, "Subsample"}
    };

    // Do one empty run:
    testAndBenchmark(reference::average2Rows);

    std::cout << "Name,Time(us),Status" << std::endl;
    for (auto test: tests)
    {
        std::cout << test.second;
        auto result = testAndBenchmark(test.first);
        std::cout << "," << result.time_us << "," << (result.maxError==0 ? "Exact" : result.maxError==1 ? "Off by one" : "Degraded")<<"\n";
    }

    return 0;
}
