#ifndef _HEAD_MPI_H_
#define _HEAD_MPI_H_
#include "mpi.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>

#define MIN(a, b) ((a)<(b)?(a):(b))

struct BestDim {
  double norma;
  int dim;
};

struct BlockNorm {
  int i;
  int j;
  double norma = -1.;

  BlockNorm() = default;
};

double get_full_time();
int get_start(const int n, const int m,
            const int id, const int p);
int get_len(const int n, const int m,
            const int id, const int p);
double Norma(double *matrix, double *buf,
             const int m, const int n,
             const int id, const int p);
double ResultNorm(double *a, double *b, double *buf,
                  const int m, const int n, const int id, const int p);
int InputMatrix(double *matrix, double *inverse, double *buf,
                const int n, const int m, const int s, char *filename,
                const int id, const int p);
bool CheckVars(const int n, const int m, const int p, const int r, const int s, const char* filename) noexcept;
void PrintMatrix(const double *matrix, double *buf,
              const int n, const int m, const int min,
              const int id, const int p_);
int Solve(double *matrix, double *inverse, double *buf, int *ind, int *index,
          const int m, const int n, const double tempNorm,
          const int id, const int p);

#endif
