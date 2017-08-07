#include <tmmintrin.h>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <chrono>
#include <algorithm>
#include <iostream>
  
namespace reference
{
    /**
     * Calculates the average of two rows of gray8 pixels by averaging four pixels.
     */
    void average2Rows(const uint8_t* row1, const uint8_t* row2, uint8_t* dst, int size)
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
    void average2Rows(const uint8_t* src1, const uint8_t* src2, uint8_t* dst, int size)
    {
        for(int i = 0;i<size-31; i+=32)
        {
            // Average the first 16 values of src1 and src2:
            __m128i left = _mm_avg_epu8(
                            _mm_loadu_si128((__m128i const*)(src1+i)), 
                            _mm_loadu_si128((__m128i const*)(src2+i)));

            // Average the following 16 values of src1 and src2:
            __m128i right = _mm_avg_epu8(
                            _mm_loadu_si128((__m128i const*)(src1+i+16)), 
                            _mm_loadu_si128((__m128i const*)(src2+i+16)));

            // Now pairwise average the 32 values in left and right:
            _mm_storeu_si128((__m128i *)(dst+(i/2)), 
                             _mm_packus_epi16(avg16Bytes(left), avg16Bytes(right)));
        }
    }
}

// The solution given by user Paul:
// https://stackoverflow.com/a/45542669/396803
namespace paul
{
    void average2Rows(const uint8_t* src1, const uint8_t* src2, uint8_t* dst, int size)
    {
        const __m128i vk1 = _mm_set1_epi8(1);

        for (int i = 0; i < size - 31; i += 32)
        {
            __m128i v0 = _mm_loadu_si128((__m128i *)&src1[i]);
            __m128i v1 = _mm_loadu_si128((__m128i *)&src1[i + 16]);
            __m128i v2 = _mm_loadu_si128((__m128i *)&src2[i]);
            __m128i v3 = _mm_loadu_si128((__m128i *)&src2[i + 16]);

            __m128i w0 = _mm_maddubs_epi16(v0, vk1);        // unpack and horizontal add
            __m128i w1 = _mm_maddubs_epi16(v1, vk1);
            __m128i w2 = _mm_maddubs_epi16(v2, vk1);
            __m128i w3 = _mm_maddubs_epi16(v3, vk1);

            w0 = _mm_add_epi16(w0, w2);                     // vertical add
            w1 = _mm_add_epi16(w1, w3);

            w0 = _mm_srli_epi16(w0, 2);                     // divide by 4
            w1 = _mm_srli_epi16(w1, 2);

            w0 = _mm_packus_epi16(w0, w1);                  // pack

            _mm_storeu_si128((__m128i *)&dst[i / 2], w0);
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
        static const __m128i Mask_01010101 = _mm_set_epi8(0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255);

        // Right shift 8 bits and add each row to itself:
        __m128i sum_row1 = _mm_add_epi16(_mm_and_si128(row1, Mask_01010101), _mm_and_si128(_mm_srli_epi64 (row1, 8), Mask_01010101));
       __m128i sum_row2 = _mm_add_epi16(_mm_and_si128(row2, Mask_01010101), _mm_and_si128(_mm_srli_epi64 (row2, 8), Mask_01010101));

        //__m128i sum_row1 = _mm_add_epi16((row1), (_mm_srli_epi64 (row1, 8)));
         //__m128i sum_row2 = _mm_add_epi16((row2), (_mm_srli_epi64 (row2, 8)));

        // Add the two rows:
        __m128i sum = _mm_add_epi16(sum_row1, sum_row2);
                
        // Divide by 4:
        sum = _mm_srli_epi16(sum, 2);

        // Mask out every other byte:
        sum =  _mm_and_si128(sum, Mask_01010101);

        return sum;
    };

    void average2Rows(const uint8_t*const src1, const uint8_t*const src2, uint8_t*const dst, int size)
    {
        for(int i = 0;i<size-31; i+=32)
        {
            __m128i tl = _mm_loadu_si128((__m128i *)&src1[i]);
            __m128i tr = _mm_loadu_si128((__m128i *)&src1[i + 16]);
            __m128i bl = _mm_loadu_si128((__m128i *)&src2[i]);
            __m128i br = _mm_loadu_si128((__m128i *)&src2[i + 16]);

            _mm_storeu_si128((__m128i *)(dst+(i/2)), _mm_packus_epi16(avg16BytesX2(tl, bl), avg16BytesX2(tr, br)));
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
        static const __m128i Mask_01010101 = _mm_set_epi8(0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255);

        // Right shift 8 bits and add each row to itself:
        __m128i sum_row1 = _mm_avg_epu8(row1, _mm_srli_epi64(row1, 8));
        __m128i sum_row2 = _mm_avg_epu8(row2, _mm_srli_epi64(row2, 8));

        // Add the two rows:
        __m128i sum = _mm_avg_epu8(sum_row1, sum_row2);

        // Mask out every other byte:
        sum = _mm_and_si128(sum, Mask_01010101);

        return sum;
    };
    void average2Rows(const uint8_t*const src1, const uint8_t*const src2, uint8_t*const dst, int size)
    {
        for (int i = 0; i < size - 31; i += 32)
        {
            __m128i tl = _mm_loadu_si128((__m128i *)&src1[i]);
            __m128i tr = _mm_loadu_si128((__m128i *)&src1[i + 16]);
            __m128i bl = _mm_loadu_si128((__m128i *)&src2[i]);
            __m128i br = _mm_loadu_si128((__m128i *)&src2[i + 16]);

            _mm_storeu_si128((__m128i *)(dst + (i / 2)), _mm_packus_epi16(avg16BytesX2(tl, bl), avg16BytesX2(tr, br)));
        }
    }
}

namespace subsample
{
    /*
     * Downsamplels two rows of gray8 pixels by sampling one out of every four pixels.
     */
    void average2Rows(const uint8_t*const src1, const uint8_t*const src2, uint8_t*const dst, int size)
    {
        for (int i = 0; i < size - 31; i += 32)
        {
            __m128i left = _mm_loadu_si128((__m128i *)&src1[i]);
            __m128i right = _mm_loadu_si128((__m128i *)&src1[i + 16]);
            _mm_storeu_si128((__m128i *)&dst[i / 2], _mm_packus_epi16(_mm_srli_epi16(left, 8), _mm_srli_epi16(right, 8)));
        }
    }
}

typedef void (*Function)(const uint8_t *,const uint8_t *,uint8_t *,int) ; 

struct BenchMarkResult
{
    int maxError = 0;
    double time_us = 0;
};

BenchMarkResult testAndBenchmark(Function fun)
{
    BenchMarkResult result;
    const int n = 1024;

    uint8_t src1[n];
    uint8_t src2[n];
    uint8_t dest_ref[n / 2];
    uint8_t dest_test[n / 2];

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
    typedef high_resolution_clock Time;
    typedef microseconds us;
    
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

/*
    Name            Time(us)  Status
    Naive           0.619641  Identical
    Original        0.188113  Off by one
    Yves Daoust v1  0.194501  Identical
    Yves Daoust v2  0.151681  Off by one
    Paul's          0.110176  Identical
    Subsample       0.055163  Degraded
*/
int main(int argc, char* argv[])
{   
    std::pair<Function, const char*> tests[] = {
        {reference::average2Rows, "Naive"}, 
        {original::average2Rows, "Original"}, 
        {yves_exact::average2Rows,  "Yves Daoust v1"}, 
        {yves_inexact::average2Rows,  "Yves Daoust v2"}, 
        {paul::average2Rows,  "Paul's"}, 
        {subsample::average2Rows, "Subsample"}
    };

    // Do one empty run:
    testAndBenchmark(reference::average2Rows);

    std::cout << "Name,Time(us),Status" << std::endl;
    for (auto test: tests)
    {
        auto result = testAndBenchmark(test.first);
        std::cout << test.second << "," << result.time_us << "," << (result.maxError==0 ? "Exact" : result.maxError==1 ? "Off by one" : "Degraded")<<"\n";
    }

    return 0;
}