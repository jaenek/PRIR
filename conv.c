#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void conv(uint8_t *src, uint8_t *dst, int width, int height, float *kernel,
          int ks) {
  if (src == NULL || dst == NULL || kernel == NULL) {
    puts("Allocate memory for params");
    exit(1);
  }

  double time = omp_get_wtime();
  int x, y, i, j, k, l;
  float val = 0.f;

  for (i = 0; i < width; i++) {
    for (j = 0; j < height; j++) {
      val = 0.f;
      for (k = 0; k < ks; k++) {
        for (l = 0; l < ks; l++) {
          x = i + k - ks / 2;
          y = j + l - ks / 2;

          if (x >= 0 && x < width && y >= 0 && y < height)
            val += src[width * x + y] * kernel[ks * k + l];
        }
      }
      dst[width * i + j] = fmaxf(0.f, fminf(255.f, val));
    }
  }
  printf("%f\n", omp_get_wtime() - time);
}

// Funkcja z paraleizacją na zewnątrz
void conv_p2(uint8_t *src, uint8_t *dst, int width, int height, float *kernel,
             int ks) {
  if (src == NULL || dst == NULL || kernel == NULL) {
    puts("Allocate memory for params");
    exit(1);
  }

  omp_set_num_threads(24);

  double time = omp_get_wtime();
  int i, j, k, l, x, y;
  float val = 0.f;

  #pragma omp parallel for schedule(static, 24) private(j, k, l, x, y) reduction(+:val)
  for (i = 0; i < width; i++) {
    for (j = 0; j < height; j++) {
      val = 0.f;
      for (k = 0; k < ks; k++) {
        for (l = 0; l < ks; l++) {
          x = i + k - ks / 2;
          y = j + l - ks / 2;

          if (x >= 0 && x < width && y >= 0 && y < height)
            val += src[width * x + y] * kernel[ks * k + l];
        }
      }
      dst[width * i + j] = fmaxf(0.f, fminf(255.f, val));
    }
  }
  printf("%f\n", omp_get_wtime() - time);
}

// Funkcja sprawdzająca
void test(uint8_t *out, uint8_t *src, int width, int height, float *kernel,
          int ks) {
  if (out == NULL || src == NULL || kernel == NULL) {
    puts("Allocate memory for params");
    exit(1);
  }
  uint8_t *test_buf = calloc(width * height, sizeof(uint8_t));
  conv(src, test_buf, width, height, kernel, ks);
  for (int32_t i = 0; i < width * height; i++) {
    if (test_buf[i] != out[i]) {
      printf("%d: %d %d\n", i, test_buf[i], out[i]);
      puts("TEST FAILED");
      exit(1);
    }
  }
  free(test_buf);
}

int main(void) {
  // Rozmycie box 2x2
  float k1[] = {
      // clang-format off
	   0.25f, 0.25f,
	   0.25f, 0.25f,
      // clang-format on
  };
  // Wykrywanie krawędzi
  float k2[] = {
      // clang-format off
	   0.f, -1.f,  0.f,
	  -1.f,  4.f, -1.f,
	   0.f, -1.f,  0.f,
      // clang-format on
  };
  // Rozmycie gaussowskie 5x5
  float k3[] = {
      // clang-format off
	  1.f / 256,  4.f / 256,  6.f / 256,  4.f / 256, 1.f / 256,
	  4.f / 256, 16.f / 256, 24.f / 256, 16.f / 256, 4.f / 256,
	  6.f / 256, 24.f / 256, 36.f / 256, 24.f / 256, 6.f / 256,
	  4.f / 256, 16.f / 256, 24.f / 256, 16.f / 256, 4.f / 256,
	  1.f / 256,  4.f / 256,  6.f / 256,  4.f / 256, 1.f / 256,
      // clang-format on
  };

  int w = 0, h = 0, n = 0;
  unsigned char *in_buf, *out_buf;

  void (*convfunc)(uint8_t *src, uint8_t *dst, int width, int height,
                   float *kernel, int ks);
  // Tutaj można zmienić badaną fukcję
  convfunc = conv;
  //convfunc = conv_p2;

  in_buf = stbi_load("sander-crombach-6b3r1WAjPBI-unsplash.jpg", &w, &h, &n, 1);
  printf("ptr: %x, w: %d, h: %d\n", in_buf, w, h);
  out_buf = calloc(w * h, sizeof(uint8_t));

  convfunc(in_buf, out_buf, w, h, k1, 2);
  test(out_buf, in_buf, w, h, k1, 2);

  convfunc(in_buf, out_buf, w, h, k2, 3);
  test(out_buf, in_buf, w, h, k2, 3);

  convfunc(in_buf, out_buf, w, h, k3, 5);
  test(out_buf, in_buf, w, h, k3, 5);

  stbi_image_free(in_buf);
  free(out_buf);
}
