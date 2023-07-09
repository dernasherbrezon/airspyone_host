#include <stdlib.h>
#include <time.h>
#include "iqconverter_float.h"
#include "filters.h"
#include <stdio.h>

#define FIRST_50_EXPECTED_LEN 50

const float FIRST_50_EXPECTED[] = {
    0.000000, 0.000000, -0.000016, 0.000000, 0.000057, 0.000000, -0.000145, 0.000000, 0.000310, 0.000000, -0.000595, 0.000000, 0.001059, 0.000000, -0.001784, 0.000000, 0.002888, 0.000000, -0.004548, 0.000000, 0.007071, 0.000000, -0.011133, 0.000000, 0.020044, -0.003906, -0.023849, 0.011602,
    0.025987, -0.019145, -0.027188, 0.026537, 0.027787, -0.033782, -0.027980, 0.040883, 0.027899, -0.047843, -0.027634, 0.054665, 0.027252, -0.061350, -0.026799, 0.067903, 0.026308, -0.074325, -0.025800, 0.080619, 0.025286, -0.086789
};

int main(void) {
  int total_executions = 1;
  int input_size = 6000000;
  float **input = malloc(sizeof(float *) * total_executions);
  if (input == NULL) {
    return EXIT_FAILURE;
  }
  for (int i = 0; i < total_executions; i++) {
    input[i] = malloc(sizeof(float *) * input_size);
    if (input[i] == NULL) {
      return EXIT_FAILURE;
    }
    for (size_t j = 0; j < input_size; j++) {
      // don't care about the loss of data
      input[i][j] = ((float) (j)) / 128.0f;
    }
  }

  iqconverter_float_t *cnv_f = iqconverter_float_create(HB_KERNEL_FLOAT, HB_KERNEL_FLOAT_LEN);
  int batch_size = 262144;
  int number_of_batches = input_size / batch_size;
  clock_t begin = clock();
  for (int i = 0; i < total_executions; i++) {
    for( int j=0;j<number_of_batches;j++ ) {
      iqconverter_float_process(cnv_f, input[i] + j * batch_size, batch_size);
    }
  }
  clock_t end = clock();
  double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
  printf("%f\n", time_spent / total_executions);
  // validation is here to make sure assembly implementation is correct + -O2 optimization won't throw away calculations completely
  for (int i = 0; i < FIRST_50_EXPECTED_LEN; i++) {
    if (((int) (input[0][i] * 1000)) != ((int) (FIRST_50_EXPECTED[i] * 1000))) {
      printf("invalid output at index %d. expected: %f got %f\n", i, FIRST_50_EXPECTED[i], input[0][i]);
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}