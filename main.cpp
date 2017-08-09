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

enum Method{Reference, Subsample, Bjorn, Yves_v1, Yves_v2, Peter_v1, Peter_v2, Paul};

template <Method method>
__m128i average(const __m128i& row1_u8, const __m128i& row2_u8);

template <Method method>
void average2Rows(const uint8_t* __restrict__ src1, const uint8_t* __restrict__ src2, uint8_t* __restrict__ dst, size_t size)
{
    size /= 2;
    for (size_t i = 0; i < size - 15; i += 16)
    {
        __m128i tl = _mm_load_si128((__m128i *)&src1[i * 2]);
        __m128i tr = _mm_load_si128((__m128i *)&src1[i * 2 + 16]);
        __m128i bl = _mm_load_si128((__m128i *)&src2[i * 2]);
        __m128i br = _mm_load_si128((__m128i *)&src2[i * 2 + 16]);

        __m128i l_avg = average<method>(tl, bl);
        __m128i r_avg = average<method>(tr, br);
            
        // Pack 16 16-bit values:
        _mm_store_si128((__m128i *)(dst + i), _mm_packus_epi16(l_avg, r_avg));
    }
}

/**
 * Calculates the average of two rows of gray8 pixels by averaging four pixels.
 */
template <>
void average2Rows<Reference>(const uint8_t* __restrict__ src1, const uint8_t* __restrict__ src2, uint8_t* __restrict__ dst, size_t size)
{
    for (int i = 0; i < size - 1; i += 2)
        *(dst++) = ((src1[i] + src1[i + 1] + src2[i] + src2[i + 1]) / 4) & 0xFF;
}
/*
* Downsamples two rows of gray8 pixels by sampling one out of every four pixels.
*/
template <>
void average2Rows<Subsample>(const uint8_t* __restrict__ src1, const uint8_t* __restrict__ src2, uint8_t* __restrict__ dst, size_t size)
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


static const __m128i Zero_u8 = _mm_setzero_si128(); 
static const __m128i One_u8 = _mm_set1_epi8(1);
static const __m128i Mask_01010101_u8 = _mm_set1_epi16(0x00FF);

// The original optimization attempt presented here:
// https://stackoverflow.com/questions/45541530/fastest-downscaling-of-8bit-gray-image-with-sse
template <Method method>
inline __m128i average(const __m128i& row1_u8, const __m128i& row2_u8)
{
    __m128i ABCDEFGHIJKLMNOP = _mm_avg_epu8(row1_u8, row2_u8);

    __m128i ABCDEFGH = _mm_unpacklo_epi8(ABCDEFGHIJKLMNOP, Zero_u8);
    __m128i IJKLMNOP = _mm_unpackhi_epi8(ABCDEFGHIJKLMNOP, Zero_u8);

    __m128i AIBJCKDL = _mm_unpacklo_epi16( ABCDEFGH, IJKLMNOP );
    __m128i EMFNGOHP = _mm_unpackhi_epi16( ABCDEFGH, IJKLMNOP );

    __m128i AEIMBFJN = _mm_unpacklo_epi16( AIBJCKDL, EMFNGOHP );
    __m128i CGKODHLP = _mm_unpackhi_epi16( AIBJCKDL, EMFNGOHP );

    __m128i ACEGIKMO = _mm_unpacklo_epi16( AEIMBFJN, CGKODHLP );
    __m128i BDFHJLNP = _mm_unpackhi_epi16( AEIMBFJN, CGKODHLP );

    return _mm_avg_epu8(ACEGIKMO, BDFHJLNP);
}


// The solution given by user Paul:
// https://stackoverflow.com/a/45542669/396803
template <>
inline __m128i average<Paul>(const __m128i& row1_u8, const __m128i& row2_u8)
{
    // Unpack and horizontal add:
    __m128i row1_avg = _mm_maddubs_epi16(row1_u8, One_u8);
    __m128i row2_avg = _mm_maddubs_epi16(row2_u8, One_u8);

    // vertical add:
    __m128i avg = _mm_add_epi16(row1_avg, row2_avg);              

    // divide by 4:
    return _mm_srli_epi16(avg, 2);                     
}

// The solution given by user Peter Cordes:
// https://stackoverflow.com/a/45564565/396803
template <>
inline __m128i average<Peter_v1>(const __m128i& row1_u8, const __m128i& row2_u8)
{
    // Average the first 16 values of src1 and src2:
    __m128i avg_u8 = _mm_avg_epu8(row1_u8, row2_u8);

    // Unpack and horizontal add:
    __m128i avg_u16 = _mm_maddubs_epi16(avg_u8, One_u8);

    // Divide by 2:
    return  _mm_srli_epi16(avg_u16, 1);
}

template <>
inline __m128i average<Peter_v2>(const __m128i& row1_u8, const __m128i& row2_u8)
{
    // Average the first 16 values of src1 and src2:
    __m128i avg_u8 = _mm_avg_epu8(row1_u8, row2_u8);

        // Line up horizontal pairs:
    __m128i odd_u8 = _mm_srli_epi16(avg_u8, 8);

        // Leaves garbage in the high halves:
    __m128i avg_u16 = _mm_avg_epu8(avg_u8, odd_u8);

        // Mask out high halves: :
    return _mm_and_si128(avg_u16, Mask_01010101_u8);
}

/*
* input: 16 8-bit values A-P
* returns: 8 16-bit values (A+B), (C+D), ..., (O+P)
*/
static inline __m128i horizonalAdd(const __m128i& ABCDEFGHIJKLMNOP)
{
    // Right shift 8 bits and add each row to itself:
    __m128i _ABCDEFGHIJKLMNO = _mm_srli_epi64 (ABCDEFGHIJKLMNOP, 8);
    __m128i _A_C_E_G_I_K_M_O = _mm_and_si128(_ABCDEFGHIJKLMNO, Mask_01010101_u8);
    __m128i _B_D_F_H_J_L_N_P = _mm_and_si128(ABCDEFGHIJKLMNOP, Mask_01010101_u8);
    return _mm_add_epi16(_A_C_E_G_I_K_M_O, _B_D_F_H_J_L_N_P);
}

// One implementation suggested by user https://stackoverflow.com/users/1196549/yves-daoust:
template <>
inline __m128i average<Yves_v1>(const __m128i& row1_u8, const __m128i& row2_u8)
{
    // Horizontal add:
    __m128i sum_row1 = horizonalAdd(row1_u8);
    __m128i sum_row2 = horizonalAdd(row2_u8);

    // Vertical add:
    __m128i sum = _mm_add_epi16(sum_row1, sum_row2);
                
    // Divide by 4:
    sum = _mm_srli_epi16(sum, 2);

    // Mask out every other byte:
    sum =  _mm_and_si128(sum, Mask_01010101_u8);

    return sum;
}


// Another version of Yves Daousts suggestion:
template <>
inline __m128i average<Yves_v2>(const __m128i& row1_u8, const __m128i& row2_u8)
{
    // Right shift 8 bits and add and average each row to itself:
    __m128i avg_row1_u8 = _mm_avg_epu8(row1_u8, _mm_srli_epi64(row1_u8, 8));
    __m128i avg_row2_u8 = _mm_avg_epu8(row2_u8, _mm_srli_epi64(row2_u8, 8));

    // Average the two rows:
    __m128i avg_u8 = _mm_avg_epu8(avg_row1_u8, avg_row2_u8);

    // Mask out every other byte:
    return _mm_and_si128(avg_u8, Mask_01010101_u8);
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

    average2Rows<Reference>(src1, src2, dest_ref, n);
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
        {average2Rows<Reference>, "Naive"}, 
        {average2Rows<Bjorn>, "Bjorn"}, 
        {average2Rows<Yves_v1>,  "Yves Daoust v1"}, 
        {average2Rows<Yves_v2>,  "Yves Daoust v2"}, 
        {average2Rows<Peter_v1>,  "Peter v1"}, 
        {average2Rows<Peter_v2>,  "Peter v2"}, 
        {average2Rows<Paul>,  "Paul"}, 
        {average2Rows<Subsample>, "Subsample"}
    };

    // Do one empty run:
    testAndBenchmark(average2Rows<Reference>);

    std::cout << "Name,Time(us),Status" << std::endl;
    for (auto test: tests)
    {
        std::cout << test.second;
        auto result = testAndBenchmark(test.first);
        std::cout << "," << result.time_us << "," << (result.maxError==0 ? "Exact" : result.maxError==1 ? "Off by one" : "Degraded")<<"\n";
    }

    return 0;
}
