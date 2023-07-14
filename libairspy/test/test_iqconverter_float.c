#include <stdlib.h>
#include <time.h>
#include "iqconverter_float.h"
#include "filters.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
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
    for (int j = 0; j < number_of_batches; j++) {
      iqconverter_float_process(cnv_f, input[i] + j * batch_size, batch_size);
    }
  }
  clock_t end = clock();
  double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
  printf("%f\n", time_spent / total_executions);
  if (argc > 1 && strcmp(argv[1], "generate") == 0) {
    FILE *fp = fopen("expected.cf32", "wb");
    if (fp == NULL) {
      printf("cannot open file to write: expected.cf32\n");
      return EXIT_FAILURE;
    }
    fwrite(input[0], sizeof(float), input_size, fp);
    fclose(fp);
  } else {
    // validation is here to make sure assembly implementation is correct + -O2 optimization won't throw away calculations completely
    FILE *fp = fopen("expected.cf32", "rb");
    if (fp == NULL) {
      printf("cannot find file: expected.cf32\n");
      return EXIT_FAILURE;
    }
    float *expected = malloc(sizeof(float) * number_of_batches * batch_size);
    fread(expected, sizeof(float), number_of_batches * batch_size, fp);
    fclose(fp);
    for (int i = 0; i < number_of_batches * batch_size; i++) {
      if (((int) (input[0][i] * 1000)) != ((int) (expected[i] * 1000))) {
        printf("invalid output at index %d. expected: %f got %f\n", i, expected[i], input[0][i]);
        return EXIT_FAILURE;
      }
    }
  }

  return EXIT_SUCCESS;
}