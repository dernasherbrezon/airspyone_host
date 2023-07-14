/*
Copyright (C) 2014, Youssef Touil <youssef@airspy.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "iqconverter_float.h"
#include <stdlib.h>
#include <string.h>

#include <stdio.h>

#if defined(__MINGW32__) && !defined(__MINGW64_VERSION_MAJOR)
#include <malloc.h>
#define _aligned_malloc __mingw_aligned_malloc
#define _aligned_free  __mingw_aligned_free
#define _inline inline
#define FIR_STANDARD
#elif defined(__APPLE__)

#include <malloc/malloc.h>

#define _aligned_malloc(size, alignment) malloc(size)
#define _aligned_free(mem) free(mem)
#define _inline inline
#define FIR_STANDARD
#elif defined(__FreeBSD__)
#define USE_SSE2
#include <immintrin.h>
#define _inline inline
#define _aligned_free(mem) free(mem)
void *_aligned_malloc(size_t size, size_t alignment)
{
    void *result;
    if (posix_memalign(&result, alignment, size) == 0)
        return result;
    return 0;
}
#elif defined(__GNUC__) && !defined(__MINGW64_VERSION_MAJOR)
#include <malloc.h>
#define _aligned_malloc(size, alignment) memalign(alignment, size)
#define _aligned_free(mem) free(mem)
#define _inline inline
#else
#if (_MSC_VER >= 1800)
    //#define USE_SSE2
    //#include <immintrin.h>
#endif
#endif

#define SIZE_FACTOR 32
#define DEFAULT_ALIGNMENT 16
#define HPF_COEFF 0.01f

#if defined(_MSC_VER)
#define ALIGNED __declspec(align(DEFAULT_ALIGNMENT))
#else
#define ALIGNED
#endif

iqconverter_float_t *iqconverter_float_create(const float *hb_kernel, int len) {
  int i, j;
  size_t buffer_size;
  iqconverter_float_t *cnv = (iqconverter_float_t *) _aligned_malloc(sizeof(iqconverter_float_t), DEFAULT_ALIGNMENT);

  cnv->len = len / 2 + 1;
  cnv->hbc = hb_kernel[len / 2];

  buffer_size = cnv->len * sizeof(float);

  cnv->fir_kernel = (float *) _aligned_malloc(buffer_size, DEFAULT_ALIGNMENT);
  cnv->fir_queue = (float *) _aligned_malloc((262144 + len - 1) * sizeof(float), DEFAULT_ALIGNMENT);
  cnv->delay_line = (float *) _aligned_malloc((262144 + 24 + 4) * sizeof(float), DEFAULT_ALIGNMENT);

  iqconverter_float_reset(cnv);
  cnv->fir_index = len;

  for (i = 0, j = 0; i < cnv->len; i++, j += 2) {
    cnv->fir_kernel[i] = hb_kernel[j];
  }

  return cnv;
}

void iqconverter_float_free(iqconverter_float_t *cnv) {
  _aligned_free(cnv->fir_kernel);
  _aligned_free(cnv->fir_queue);
  _aligned_free(cnv->delay_line);
  _aligned_free(cnv);
}

void iqconverter_float_reset(iqconverter_float_t *cnv) {
  cnv->avg = 0.0f;
  cnv->fir_index = 0;
  cnv->delay_index = 0;
  memset(cnv->delay_line, 0, (262144 + 24 + 4) * sizeof(float));
  memset(cnv->fir_queue, 0, (262144 + 47 - 1) * sizeof(float));
}

#define SCALE (0.01f)

void iqconverter_float_process(iqconverter_float_t *cnv, float *samples, int len) {
  float avg = cnv->avg;
  float hbc = cnv->hbc;
  float *bPtr = cnv->fir_kernel;
  //FIXME 46 is the history
  memcpy(cnv->fir_queue + 46, samples, len * sizeof(float));

  for (int i = 0; i < (len); i+=4) {
    cnv->fir_queue[i + 46] -= avg;
    avg += SCALE * cnv->fir_queue[i + 46];
    cnv->fir_queue[i + 46] = -(cnv->fir_queue[i + 46]);
    samples[i] = bPtr[0] * (cnv->fir_queue[i] + cnv->fir_queue[i+46])
                 + bPtr[1] * (cnv->fir_queue[i+2] + cnv->fir_queue[i+44])
                 + bPtr[2] * (cnv->fir_queue[i+4] + cnv->fir_queue[i+42])
                 + bPtr[3] * (cnv->fir_queue[i+6] + cnv->fir_queue[i+40])
                 + bPtr[4] * (cnv->fir_queue[i+8] + cnv->fir_queue[i+38])
                 + bPtr[5] * (cnv->fir_queue[i+10] + cnv->fir_queue[i+36])
                 + bPtr[6] * (cnv->fir_queue[i+12] + cnv->fir_queue[i+34])
                 + bPtr[7] * (cnv->fir_queue[i+14] + cnv->fir_queue[i+32])
                 + bPtr[8] * (cnv->fir_queue[i+16] + cnv->fir_queue[i+30])
                 + bPtr[9] * (cnv->fir_queue[i+18] + cnv->fir_queue[i+28])
                 + bPtr[10] * (cnv->fir_queue[i+20] + cnv->fir_queue[i+26])
                 + bPtr[11] * (cnv->fir_queue[i+22] + cnv->fir_queue[i+24]);

    samples[i+1] = cnv->delay_line[i + 1];
    cnv->fir_queue[i + 46 + 1] -= avg;
    avg += SCALE *  cnv->fir_queue[i + 46 + 1];
    cnv->delay_line[i + 1 + 24] = -(cnv->fir_queue[i + 46 + 1] * hbc);

    cnv->fir_queue[i + 46 + 2] -= avg;
    avg += SCALE *  cnv->fir_queue[i + 46 + 2];
    samples[i+2] = bPtr[0] * (cnv->fir_queue[i+2+0] + cnv->fir_queue[i+2+46])
                 + bPtr[1] * (cnv->fir_queue[i+2+2] + cnv->fir_queue[i+2+44])
                 + bPtr[2] * (cnv->fir_queue[i+2+4] + cnv->fir_queue[i+2+42])
                 + bPtr[3] * (cnv->fir_queue[i+2+6] + cnv->fir_queue[i+2+40])
                 + bPtr[4] * (cnv->fir_queue[i+2+8] + cnv->fir_queue[i+2+38])
                 + bPtr[5] * (cnv->fir_queue[i+2+10] + cnv->fir_queue[i+2+36])
                 + bPtr[6] * (cnv->fir_queue[i+2+12] + cnv->fir_queue[i+2+34])
                 + bPtr[7] * (cnv->fir_queue[i+2+14] + cnv->fir_queue[i+2+32])
                 + bPtr[8] * (cnv->fir_queue[i+2+16] + cnv->fir_queue[i+2+30])
                 + bPtr[9] * (cnv->fir_queue[i+2+18] + cnv->fir_queue[i+2+28])
                 + bPtr[10] * (cnv->fir_queue[i+2+20] + cnv->fir_queue[i+2+26])
                 + bPtr[11] * (cnv->fir_queue[i+2+22] + cnv->fir_queue[i+2+24]);

    samples[i + 3] = cnv->delay_line[i + 3];
    cnv->fir_queue[i + 46 + 3] -= avg;
    avg += SCALE *  cnv->fir_queue[i + 46 + 3];
    cnv->delay_line[i + 3 + 24] =  cnv->fir_queue[i + 46 + 3] * hbc;

  }

  memcpy(cnv->fir_queue, cnv->fir_queue + (len), 46 * sizeof(float));
  memcpy(cnv->delay_line, cnv->delay_line + (len), (24 + 4) * sizeof(float));

  cnv->avg = avg;

}
