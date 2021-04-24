#include "head_MPI.h"

////// INPUT_OUTPUT_TOOLS //////

double get_full_time() {
  struct timeval buf;
  gettimeofday(&buf,0);
  return (double)buf.tv_sec+(double)buf.tv_usec/1000000.;
}

void PrintBlock(const double *matrix, const int n, const int l,  const int r) {
  int cols = MIN(l, r);
  int rows = MIN(n, r);
  for (int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      printf(" %10.3e", matrix[i * l + j]);
    }
    printf("\n");
  }
}

bool CheckVars(const int n, const int m, const int p, 
               const int r, const int s, const char* filename) noexcept {
  int num = (n % m != 0 ? n / m + 1 : n / m);
  return (num >= p) && (p > 0) && (n > 0) && (m > 0) && (r > 0) && (n >= m) && (s > -1)
         && (s < 5) && (!(s > 0) || (filename == 0)) && (m % 3 == 0);
}

double FillFunc(const int i, const int j,
                const int n, const int s) noexcept {
  switch (s) {
    case -1:
      return 1.0 * (i != j ? 0 : 1);
    case 1:
      return 1.0 * (n - (i < j ? j : i));
    case 2:
      return 1.0 * (i < j ? j : i) + 1;
    case 3:
      return 1.0 * fabs(i - j);
    case 4:
      return 1.0 / (i + j + 1);
    default:
      return 0.0;
  }
}

int get_len(const int n, const int m,
            const int id, const int p) {
  if (p == 1)
    return n;
  int k = n / m, l = n - m * k;
  int dim = (l != 0 ? k + 1 : k);
  int len = 0, rest = dim % p;
  len = (dim / p) * m + 
          (id < rest && p != dim ? m : 0) + 
          ((rest == (id + 1) || k % p == id) && l != 0 ? l - m : 0);
  return len;
}

double Norma(double *matrix, double *buf,
             const int m, const int n,
             const int id, const int p) {
  int len = get_len(n, m, id, p);
  int dim = len / m;
  int k = n / m, l = n - k * m;
  int dimRow, dimCol;
  double *temp = buf + m * n;
  double *block;
  double max = -1.0;
  for (int i = 0; i < len * n; i += m * n) {
    dimCol = (i != dim * m * n ? m : l);
    for (int u = 0; u < dimCol; u++) {
      temp[u] = 0.;
    }
    for (int j = 0; j < dimCol * n; j += dimCol * m) {
      dimRow = (j != k * dimCol * m ? m : l);
      block = matrix + i + j;
      for (int u = 0; u < dimRow; u++) {
        for (int v = 0; v < dimCol; v++) {
	  temp[v] += fabs(block[u * dimCol + v]);
	}
      } 
    }
    for (int u = 0; u < dimCol; u++) {
      if (temp[u] > max) {
        max = temp[u];
      }
    }
  }
  return max;
}

void InputMatrixWithAlg(double *matrix,
                        const int n, const int m,
                        const int id, const int p_, 
                        const int s) noexcept {
  int k = n / m, l = n - k * m;
  int j_loc, dim_block_row, dim_block_col;
  int i_true = 0, j_true = 0;
  for (int j_glob = id * m * n; j_glob < n * n; j_glob += m * n * p_) { // по всем столбцам
    j_loc = (j_glob - id * m * n) / p_;
    i_true = 0;
    j_true = j_glob / n;
    dim_block_col = (j_glob != k * n * m ? m : l);
    for (int t = 0; t < n * dim_block_col; t += m * dim_block_col) { // по блокАМ столбца
      dim_block_row = ((t != k * m * dim_block_col) ? m : l);
      for (int i = 0; i < dim_block_row; i++) { // по блокУ столбца
        for (int j = 0; j < dim_block_col; j++) {
          matrix[j_loc + t + i * dim_block_col + j] = FillFunc(i_true, j_true, n, s);
          j_true++;
	}
	j_true -= dim_block_col;
	i_true++;
      }
    }
  }
}

int FileInput(double *matrix, double *buf,
              int n, int m, char *filename,
              int id, int p_) {
  int loc_err = 0;
  int block_row_counter = 0, j_loc;
  int k = n / m, l = n - k * m;
  MPI_Comm G = MPI_COMM_WORLD;
  int counter = 0, all_counter = 0, i_counter = 0;
  int dim_cols, dim_rows;
  int len = get_len(n, m, id, p_);
  int num = (l != 0 ? k + 1: k);
  FILE *IN = 0;
  double *buf_new = buf + (num / p_ + 1) * m * m;
  int glob_err;
  for (int i = 0; i < n * m; i++) {
    buf_new[i] = 0.;
  }
  if ((!id) && (!(IN = fopen(filename, "r"))))
    loc_err = 1;
  MPI_Bcast(&loc_err, 1, MPI_INT, 0, G);
  if (loc_err) {
    return -1;
  }
  for (int r = 0; r < n * n; r += n * m) {
    dim_cols = (r != k * n * m ? m : l);
    if (!id) {
      all_counter = 0;
      for (int i = 0; i < dim_cols  && (!loc_err); i++) {
        i_counter = 0;
        for (int t = 0; i_counter < n  && (!loc_err); t++) {
          counter = 0;
          for (int p = 0; p < p_ && i_counter < n && (!loc_err); p++) {
            dim_rows = (i_counter != n - l ? m : l);
            for (int j = 0; j < dim_rows && (!loc_err); j++) {
              if (fscanf(IN, "%lf",
                  &buf[counter + t * dim_cols * m + i * dim_rows + j]) != 1) {
                loc_err = 2;
                break;
	      }
              all_counter++;
              i_counter++;
            }
            counter += dim_cols * (num / p_ + 1) * m;
          }
        }
      }
    }
    MPI_Allreduce(&loc_err, &glob_err, 1, MPI_INT, MPI_SUM, G);
    if (!glob_err) {
      MPI_Scatter(buf, (num / p_ + 1) * m * dim_cols, MPI_DOUBLE,
               buf_new, (num / p_ + 1) * m * dim_cols, MPI_DOUBLE, 0, G);
    }
    for (int i = 0; i < n * len; i += n * m) {
      dim_rows = (i + n * m > n * len && l != 0 ? l : m);
      j_loc = block_row_counter * dim_rows * m;
      memcpy(matrix + j_loc + i, buf_new + i / n * dim_cols, dim_rows * dim_cols * sizeof(double));
    }
    block_row_counter++;
  }
  if (!id)
    fclose(IN);
  MPI_Bcast(&loc_err, 1, MPI_INT, 0, G);
  if (loc_err)
    return -1;
  return 0;
}

int InputMatrix(double *matrix, double *inverse, double *buf,
                const int n, const int m, const int s, char *filename,
                const int id, const int p) {
  InputMatrixWithAlg(inverse, n, m, id, p, -1);
  InputMatrixWithAlg(matrix, n, m, id, p, s);
  if (filename) {
    return FileInput(matrix, buf, n, m, filename, id, p);
  }
  return 0;
}

void PrintMatrix(const double *matrix, double *buf,
              const int n, const int m, const int min,
              const int id, const int p_) {
  int k = n / m, l = n - k * m;
  MPI_Comm G = MPI_COMM_WORLD;
  int counter = 0, all_counter = 0, i_counter = 0, i_glob_counter = 0;
  int j_loc;
  int dim_cols, dim_rows;
  int len = get_len(n, m, id, p_);
  int dim = MIN(n, min);
  int block_row_counter = 0;
  int num = (l != 0 ? k + 1 : k);
  int div = len / m;
  double *buf_new = buf + (num / p_ + 1) * m * m;
  for (int i = 0; i < 2 * p_ * (num / p_ + 1) * m * m + 3 * m * m; i++) {
    buf[i] = 0.;
  }
  for (int r = 0; r < dim * n; r += n * m) {
    dim_cols = (r != k * n * m ? m : l);
    for (int i = 0; i < n * len; i += n * m) {
      dim_rows = (i != div * n * m ? m : l);
      j_loc = block_row_counter * dim_rows * m;
      memcpy(buf_new + i / n * dim_cols, matrix + j_loc + i, dim_rows * dim_cols * sizeof(double));
    } 
    MPI_Gather(buf_new, (num / p_ + 1) * m * dim_cols, MPI_DOUBLE, 
               buf, (num / p_ + 1) * m * dim_cols, MPI_DOUBLE, 0, G);
    if (!id) {
      all_counter = 0;
      for (int i = 0; i < dim_cols && i_glob_counter < dim; i++) {
        i_counter = 0;
        for (int t = 0; i_counter < dim && i_glob_counter < dim; t++) {
          counter = 0;
          for (int p = 0; p < p_ && i_counter < dim; p++) {
            dim_rows = (i_counter != n - l ? m : l);
            for (int j = 0; j < dim_rows && i_counter < dim; j++) {
              printf("%10.3e ", buf[counter + t * dim_cols * m + i * dim_rows + j]);
	      all_counter++;
	      i_counter++;
	    }
	    counter += dim_cols * (num / p_ + 1) * m;
          }
	}
	i_glob_counter++;
	printf("\n");
      }
    }
    block_row_counter++;
  }
  if (id == 0) {
    printf("\n");
  }
}
///////////////////////////////////////////////////////

static inline double Norma(const double* matrix, const int n, const int m) {
  double sum = 0;
  double max = 0; 
  for (int i = 0; i < n; i++) {
    sum = 0;
    for (int j = 0; j < m; j++)
      sum += fabs(matrix[i * n + j]);
    if (sum > max)
      max = sum;
  }
  return max;
}

static inline void BlockMultiplication(const double* block1, const double* block2, double* result,
                         const int n, const int l, const int m) {
  int r, t, q;
  double sum = 0;
  double c[9];
  if (n == l &&  l == m && n % 3 == 0) {
    for (r = 0; r < n; r += 3) {
      for (t = 0; t < m;  t += 3) {
        c[0] = 0.;
        c[1] = 0.;
        c[2] = 0.;
        c[3] = 0.;
        c[4] = 0.;
        c[5] = 0.;
        c[6] = 0.;
        c[7] = 0.;
        c[8] = 0.;
        for (q = 0; q < l; ++q) {
          c[0] += block1[r * l + q] * block2[q * m + t];
          c[1] += block1[r * l + q] * block2[q * m + t + 1];
          c[2] += block1[r * l + q] * block2[q * m + t + 2];
          c[3] += block1[(r + 1) * l + q] * block2[q * m + t];
          c[4] += block1[(r + 1) * l + q] * block2[q * m + t + 1];
          c[5] += block1[(r + 1) * l + q] * block2[q * m + t + 2];
          c[6] += block1[(r + 2) * l + q] * block2[q * m + t];
          c[7] += block1[(r + 2) * l + q] * block2[q * m + t + 1];
          c[8] += block1[(r + 2) * l + q] * block2[q * m + t + 2];
        }
        result[r * n + t]             = c[0];
        result[r * n + t + 1]         = c[1];
        result[r * n + t + 2]         = c[2];
        result[(r + 1) * n + t]       = c[3];
        result[(r + 1) * n + (t + 1)] = c[4];
        result[(r + 1) * n + (t + 2)] = c[5];
        result[(r + 2) * n + t]       = c[6];
        result[(r + 2) * n + (t + 1)] = c[7];
        result[(r + 2) * n + (t + 2)] = c[8];
      }
    }
  }
  else {
    for(int i = 0; i < n; i++)
      for(int j = 0; j < m; j++) {
        sum = 0;
        for(int k = 0; k < l; k++)
          sum += block1[i * l + k] * block2[j + k * m];
        result[i * m + j] = sum;
      }
   }
}

static inline void SumBlocks(double* matrix1, const double* matrix2,
               const int d, const int l) {
 if (d == l && d % 3 == 0) {
  for (int i = 0; i < d; i += 3)
    for (int j = 0; j < l; j += 3) {
      matrix1[i * l + j] += matrix2[i * l + j];
      matrix1[i * l + j + 1] += matrix2[i * l + j + 1];
      matrix1[i * l + j + 2] += matrix2[i * l + j + 2];
      matrix1[(i + 1) * l + j] += matrix2[(i + 1) * l + j];
      matrix1[(i + 1) * l + j + 1] += matrix2[(i + 1) * l + j + 1];
      matrix1[(i + 1) * l + j + 2] += matrix2[(i + 1) * l + j + 2];
      matrix1[(i + 2) * l + j ] += matrix2[(i + 2) * l + j];
      matrix1[(i + 2) * l + j + 1] += matrix2[(i + 2) * l + j + 1];
      matrix1[(i + 2) * l + j + 2] += matrix2[(i + 2) * l + j + 2];
    }
 } else {
   for (int i = 0; i < d; i++)
    for (int j = 0; j < l; j++) {
      matrix1[i * l + j] -= matrix2[i * l + j];
    }
 }
}

double ResultNorm(double *a, double *b, double *buf, 
                  const int m, const int n, const int id, const int p) {
  int k = n / m, l = n % m;
  int num = (l != 0 ? k + 1 : k);
  int j_loc;
  int block_row_counter = 0;
  int len = get_len(n, m, id, p), div = len / m;
  int temp_len, dim_cols, dim_rows, temp_div, dim_col, dim_row;
  double max = -1.0;
  double *buf_new = buf + p * (num / p + 1) * m * m;
  double *buffer, *temp, *result, *total_result, *output_result;
  MPI_Comm G = MPI_COMM_WORLD;
  for (int r = 0; r < n * n; r += n * m) {
    for (int i = 0; i < 2 * p * (num / p + 1) * m * m + 3 * m * m; i++) {
      buf[i] = 0.;
    }
    dim_cols = (r != k * n * m ? m : l);
    div = len / m;
    for (int i = 0; i < n * len; i += n * m) {
      dim_rows = (i != div * n * m ? m : l);
      j_loc = block_row_counter * dim_rows * m;
      memcpy(buf_new + i / n * dim_cols, a + j_loc + i, dim_rows * dim_cols * sizeof(double));
    }
    MPI_Allgather(buf_new, (num / p + 1) * m * dim_cols, MPI_DOUBLE,
               buf, (num / p + 1) * m * dim_cols, MPI_DOUBLE, G);
    buffer = buf_new;
    int i_count = 0;
    for (int i = 0; i < p * (num / p + 1) * m * dim_cols; i += (num / p + 1) * m * dim_cols) {
      temp_len = get_len(n, m, i_count, p);
      temp_div = temp_len / m;
      j_loc = i_count * m * dim_cols;
      for (int j = 0; j < temp_len * dim_cols; j += dim_cols * m) {
	dim_rows = (j != temp_div * dim_cols * m ? m : l);
        memcpy(buffer + j_loc, buf + i + j, dim_rows * dim_cols * sizeof(double));
	j_loc += p * m * dim_cols;
      }
      i_count++;
    }
    temp = buf;
    result = buf + m * m;
    total_result = buf + 2 * m * m;
    for (int i = 0; i < dim_cols; i++) {
      total_result[i] = 0;
    }
    for (int i = 0; i < len * n; i += n * m) {
      dim_row = (i != div * n * m ? m : l);
      for (int j = 0; j < m * m; j++) {
        result[j] = 0;
      }
      int j_row = 0;
      for (int j = 0; j < dim_row * n; j += dim_row * m) {
        dim_col = (j != k * m * dim_row ? m : l);
        BlockMultiplication(buffer + j_row, b + i + j, temp, dim_cols, dim_col, dim_row);
        SumBlocks(result, temp, dim_cols, dim_row);
        j_row += dim_cols * m;
      }
      for (int u = 0; u < dim_cols; u++) {
        for (int v = 0; v < dim_row; v++) {
	   total_result[u] += fabs(result[u * dim_row + v]);
	}
      }
    }
    output_result = buf;
    for (int i = 0; i < dim_cols; i++) {
      output_result[i] = 0;
    }
    MPI_Allreduce(total_result, output_result, dim_cols, MPI_DOUBLE, MPI_SUM, G);
    for (int i = 0; i < dim_cols; i++) {
      output_result[i] -= 1;
      if (output_result[i] > max) {
        max = output_result[i];
      }
    }
    block_row_counter++;
  }
  return max;
}

static inline int InverseBlock(double *a, double *x, const int n, const double tempNorm, int* index) {
  int indMax1;
  int indMax2;
  int i, j, k;
  double tmp;
  int tmp1;
  double max;
  for (i = 0; i < n; i++)
    index[i] = i;
  for (i = 0; i < n; ++i)
    for (j = 0; j < n; ++j)
      x[i * n + j] = (double)(i == j);
  for (i = 0; i < n; ++i) {
    max = -1;
    indMax1 = 0;
    indMax2 = 0;
    tmp1 = (n > 40 ? i + 1 : n);
    for (j = i; j < tmp1; ++j)
      for (k = i; k < n; ++k)
        if (max < fabs(a[j * n + k])) {
          max = fabs(a[j * n + k]);
          indMax1 = j;
          indMax2 = k;
        }
    if (max <= fabs(1e-16 * tempNorm))
      return -1;
    if (indMax1 != i) {
      for (j = 0; j < n; ++j) {
          tmp = a[i * n + j];
          a[i * n + j] = a[indMax1 * n + j];
          a[indMax1 * n + j] = tmp;
      }
      for (j = 0; j < n; ++j) {
        tmp = x[i * n + j];
        x[i * n + j] = x[indMax1 * n + j];
        x[indMax1 * n + j] = tmp;
      }
    }
    if (indMax2 != i) {
      k = index[i];
      index[i] = index[indMax2];
      index[indMax2] = k;
      for (j = 0; j < n; ++j) {
        tmp = a[j * n + i];
        a[j * n + i] = a[j * n + indMax2];
        a[j * n + indMax2] = tmp;
      }
    }
    tmp = 1.0 / a[i * n + i];
    for (j = i; j < n; ++j)
      a[i * n + j] *= tmp;
    for (j = 0; j < n; ++j)
      x[i * n + j] *= tmp;
    for (j = i + 1; j < n; ++j) {
      tmp = a[j * n + i];
      for (k = i; k < n; ++k)
        a[j * n + k] -= a[i * n + k] * tmp;
      for (k = 0; k < n; ++k)
        x[j * n + k] -= x[i * n + k] * tmp;
    }
  }
  for (k = 0; k < n; ++k)
    for (i = n - 1; i >= 0; --i) {
      tmp = x[i * n + k];
      for (j = i + 1; j < n; ++j)
        tmp -= a[i * n + j] * x[j * n + k];
      x[i * n + k] = tmp;
    }
  for (i = 0; i < n; ++i) {
    k = index[i];
    for (j = 0; j < n; ++j)
      a[k * n + j] = x[i * n + j];
  }
   return 0;
}

void ChangeBlocks(double *block_1, double *block_2,
                  const int dim1, const int dim2,
		  double *buf) {
  memcpy(buf, block_1, dim1 * dim2 * sizeof(double));
  memcpy(block_1, block_2, dim1 * dim2 * sizeof(double));
  memcpy(block_2, buf, dim1 * dim2 * sizeof(double));
}

void ChangeRow(double *matrix, double *buf,
               const int m, const int n,
               const int /*p*/, const int len,
	       const int i_1, const int i_2) {
  int dim_row;
  int l = n - (n / m) * m, k = len / m;
  for (int i = 0; i < len * n; i += n * m) {
    dim_row = (i != k * m * n ? m : l);
    ChangeBlocks(matrix + i + i_1 * m * dim_row, matrix + i + i_2 * m * dim_row, m, dim_row, buf);
  }
}

void ChangeCol(double *matrix, double *buf,
               const int m, const int n,
               const int /*p*/, const int /*len*/,
               const int j_1, const int j_2) {
  int dim_col;
  int l = n - (n / m) * m, k = n / m;
  for (int i = 0; i < m * n; i += m * m) {
    dim_col = (i != k * m * m ? m : l);
    ChangeBlocks(matrix + j_1 * m * n + i, matrix + j_2 * m * n + i, dim_col, m, buf);
  }
}

static inline void DiffBlocks(double* matrix1, const double* matrix2,
               const int d, const int l) {
 if (d == l && d % 3 == 0) {
  for (int i = 0; i < d; i += 3)
    for (int j = 0; j < l; j += 3) {
      matrix1[i * l + j] -= matrix2[i * l + j];
      matrix1[i * l + j + 1] -= matrix2[i * l + j + 1];
      matrix1[i * l + j + 2] -= matrix2[i * l + j + 2];
      matrix1[(i + 1) * l + j] -= matrix2[(i + 1) * l + j];
      matrix1[(i + 1) * l + j + 1] -= matrix2[(i + 1) * l + j + 1];
      matrix1[(i + 1) * l + j + 2] -= matrix2[(i + 1) * l + j + 2];
      matrix1[(i + 2) * l + j ] -= matrix2[(i + 2) * l + j];
      matrix1[(i + 2) * l + j + 1] -= matrix2[(i + 2) * l + j + 1];
      matrix1[(i + 2) * l + j + 2] -= matrix2[(i + 2) * l + j + 2];
    }
 } else {
   for (int i = 0; i < d; i++)
    for (int j = 0; j < l; j++) {
      matrix1[i * l + j] -= matrix2[i * l + j];
    }
 }
}

void PutBlock(double *block_1, const double *block_2,
              const int dim1, const int dim2) {
  memcpy(block_1, block_2, dim1 * dim2 * sizeof(double));
}

void GetBlock(const double *block_1, double *block_2,
              const int dim1, const int dim2) {
  memcpy(block_2, block_1, dim1 * dim2 * sizeof(double));
}

static inline void RowMultiplication(double* matrix, const double* x,
                                     const int t, const int p, const int d,
                                     const int n, const int m, const int len,
                                     const int /*id*/, const int ptr,
                                     double* temp, double* result) {
  int div = len / m;
  int rowDim = d, colDim;
  int l = n % m;
  int jStart = p / ptr;
  int iPos;
  for (int j = jStart * n * m; j < len * n; j += n * m) { // умножаем обращенный блок на остальные в строке
    colDim = (j != div * m * n ? m : l);
    iPos = t * colDim * m;
    GetBlock(matrix + j + iPos, temp, rowDim, colDim);
    result = matrix + j + iPos;
    BlockMultiplication(x, temp, result, rowDim, rowDim, colDim);
  }
}

static inline void ReverseGauss(double* matrix, double* buffer,
                                const int epoch, const int m, const int n, const int len, const int /*id*/, const int /*p*/,
                                double* values, double* temp1, double* result, double *temp2) {
  int div = len / m;
  int k = n / m;
  int l = n % m;
  int iStart, jStart = 0;
  int dimCol, dimValues = (epoch != k ? m : l);
  //dimRow = (epoch != k ? m : l);
  for (int j = jStart; j < len * n; j += n * m) {
    dimCol = (j != div * m * n ? m : l);
    iStart = epoch * m * dimCol;
    temp1 = matrix + j + iStart;
    iStart -= dimCol * m;
    for (int i = iStart; i > -1; i -= dimCol * m) {
      values = buffer + (i / dimCol) * dimValues;
      result = matrix + j + i; // RESULT
      BlockMultiplication(values, temp1, temp2, m, dimValues, dimCol);
      DiffBlocks(result, temp2, m, dimCol);
    }
  }
}

static inline void ForwardGauss(double* matrix,
                                const int epoch, const int m, const int n, const int len, const int id, const int p,
                                double* values, double* temp1, double* result, double *temp2, double* buffer,
                                const bool notInverse) {
  int div = len / m;
  int k = n / m;
  int l = n % m;
  int iStart, jStart = (epoch / p) * m * n + (epoch % p != id ? 0 : 1) * m * n;
  jStart = (notInverse ? jStart : 0);
  int dimCol, dimRow;
  for (int j = jStart; j < len * n; j += n * m) {
    dimCol = (j != div * m * n ? m : l);
    iStart = epoch * m * dimCol;
    temp1 = matrix + j + iStart; 
    iStart += dimCol * m;
    for (int i = iStart; i < dimCol * n; i += dimCol * m) {
      values = buffer + i / dimCol * m;
      dimRow = (i != k * dimCol * m ? m : l);
      result = matrix + j + i; // RESULT
      BlockMultiplication(values, temp1, temp2, dimRow, m, dimCol);
      DiffBlocks(result, temp2, dimRow, dimCol);
    }
  }
}

BlockNorm FindMinBlock(const double *matrix, double *buf, int *ind,
                   const int m, const int n, const int p,
		   double tempNorm,
                   const int len, const int epoch, const int id) {
  int dim_row, dim_col;
  BlockNorm result = {epoch, epoch, -1.};
  double newMin = -1.;
  int l = n - (n / m) * m, k_glob = n / m, k = len / m;
  if (epoch == k_glob && epoch % p == id) {
    memcpy(buf, matrix + n * m * k + l * m * k_glob, l * l * sizeof(double));
    if (InverseBlock(buf, buf + m * m, l, tempNorm, ind) == 0) {
      result.norma = Norma(buf, l, l);
      result.i = n / m;
      result.j = n / m;
    }
    return result;
  }
  int iStart = (id + p * (epoch / p) >= epoch ? epoch / p : epoch / p + 1) * m * n;
  for (int i = iStart; i < k * n * m; i += n * m) {
    dim_row = m;
    for (int j = epoch * m * m; j < dim_row * n; j += dim_row * m) {
      dim_col = (j != k_glob * m * m ? m : l);
      memcpy(buf, matrix + i + j, dim_row * dim_col * sizeof(double));
      if (dim_col == dim_row && InverseBlock(buf, buf + m * m, dim_col, tempNorm, ind) == 0) {
        newMin = Norma(buf, dim_row, dim_row);
        if (newMin < result.norma || result.norma < 0) {
          result.norma = newMin;
          result.i = j / m / m;  // position in local memory
          result.j = id + i / n / m * p;
        }
      }
    }
  }
  return result;
}

int Solve(double *matrix, double *inverse, double *buf, int *ind, int *index,
          const int m, const int n, const double tempNorm,
          const int id, const int p) {
  int len = get_len(n, m, id, p);
  int k = n / m, l = n - k * m;
  int blockNum = (l != 0 ? k + 1 : k);
  int loc_err = 0;
  (void) loc_err;
  MPI_Comm G = MPI_COMM_WORLD;
  MPI_Status stat;
  BlockNorm topBlock;
  int tmp;
  for (int i = 0; i < n; i++) {
    index[i] = i;
  }
  BlockNorm bestIdIn[1];
  BlockNorm bestIdOut[1] = {-1, -1, -1};
  BestDim bestI[1];
  BestDim bestJ[1];
  BestDim resI[1];
  BestDim resJ[1];
  for (int epoch = 0; epoch < blockNum; epoch++) {
    bestIdIn[0] = FindMinBlock(matrix, buf, ind, m, n, p, tempNorm, len, epoch, id);
    bestIdIn[0].norma = 1.0 / bestIdIn[0].norma;
    bestI[0].norma = bestJ[0].norma = bestIdIn[0].norma;
    bestI[0].dim = bestIdIn[0].i;
    bestJ[0].dim = bestIdIn[0].j;
    MPI_Allreduce(&bestI, &resI, 1, MPI_DOUBLE_INT, MPI_MAXLOC, G); // to do 
    MPI_Allreduce(&bestJ, &resJ, 1, MPI_DOUBLE_INT, MPI_MAXLOC, G);
    bestIdOut[0].norma = resI[0].norma;
    if (bestIdOut[0].norma < 0) {
      return -2;
    }
    bestIdOut[0].i = resI[0].dim;
    bestIdOut[0].j = resJ[0].dim;
    if (bestIdOut[0].i != epoch) {
      ChangeRow(matrix, buf, m, n, p, len, bestIdOut[0].i, epoch);
      ChangeRow(inverse, buf, m, n, p, len, bestIdOut[0].i, epoch);
    }
    int dimRow = (epoch != k ? m : l);
    if (id != bestIdOut[0].j % p) {
      MPI_Bcast(buf, m * n,  MPI_DOUBLE, bestIdOut[0].j % p, G);
    } else {
      MPI_Bcast(matrix + (bestIdOut[0].j / p) * m * n, m * n,  MPI_DOUBLE, bestIdOut[0].j % p, G);
    }
    bool flag = false;
    if (bestIdOut[0].j != epoch) {
      tmp = index[epoch];
      index[epoch] = index[bestIdOut[0].j];
      index[bestIdOut[0].j] = tmp;
      if (!(epoch % p == bestIdOut[0].j % p) && (epoch % p == id || bestIdOut[0].j % p == id)) {
        tmp = (epoch % p == id ? epoch / p : bestIdOut[0].j / p);  // меняем столбцы в разных процессах
        flag = true;
	if (epoch % p == id) {
          memcpy(buf + n * m, buf + epoch * m * m, dimRow * dimRow * sizeof(double));
	  MPI_Send(matrix + (epoch / p) * n * m , m * n,
                   MPI_DOUBLE, bestIdOut[0].j % p, 0, G);
	  memcpy(matrix + (epoch / p) * n * m, buf, m * n * sizeof(double)); //may be removed
	} else {
          memcpy(buf, matrix + (bestIdOut[0].j / p) * n * m, m * n * sizeof(double));
	  memcpy(buf + n * m, matrix + (epoch / p) * m * n + epoch * m * m, dimRow * dimRow * sizeof(double));
	  MPI_Recv(matrix + (bestIdOut[0].j / p) * n * m, m * n,
                   MPI_DOUBLE, epoch % p, 0, G, &stat);
	}
      } else if (id == epoch % p) {
        ChangeCol(matrix, buf + n * m + 2 * m * m, m, n, p, len, bestIdOut[0].j / p, epoch / p); // меняем столбцы в одном процессе
      }
    }
    //
    if (!flag && id == epoch % p) {
      memcpy(buf + n * m, matrix + (epoch / p) * m * n + epoch * m * dimRow, dimRow * dimRow * sizeof(double));
    } else {
      memcpy(buf + n * m, buf + epoch * dimRow * m, dimRow * dimRow * sizeof(double));
    }
    double *temp = buf + m * n;
    double *x = buf + m * n + m * m;
    double *result = buf + m * n + 2 * m * m;
    //MPI_Barrier(G);
    /*
    if (id == 1) {
      printf("======= proc = %d\n", id);
      PrintBlock(temp, dimRow, dimRow, dimRow);
      printf("\n");
    }
    */
    InverseBlock(temp, x, dimRow, tempNorm, ind);
    if (id == -1) {
      printf("======= proc = %d\n", id);
      PrintBlock(temp, dimRow, dimRow, dimRow);
      printf("\n");
    }
    RowMultiplication(matrix, temp, epoch, epoch, dimRow, n, m, len, id, p, x, result);
    RowMultiplication(inverse, temp, epoch, 0, dimRow, n, m, len, id, p, x, result);
    /*
    MPI_Barrier(G);
    if (id == 0) {
      printf("AFTER ROW MULTIPLICATION\n");
    }
    if (id == 0 || p > 1) {
      PrintMatrix(matrix, buf, n, m, 10, id, p);
      MPI_Barrier(G);
      PrintMatrix(inverse, buf, n, m, 10, id, p);
    }
    */
    double *temp1 = buf + m * n;
    double *buffer = buf;
    if (!flag && id == epoch % p) {
      buffer = matrix + (epoch / p) * m * n;
    }
    ForwardGauss(matrix, epoch, m, n, len, id, p, x, temp, result, temp1, buffer, true);
    ForwardGauss(inverse, epoch, m, n, len, id, p, x, temp, result, temp1, buffer, false);
    /*
    MPI_Barrier(G);
    if (id == 0) {
      printf("AFTER FORWARD GAUSS step= %d\n", epoch);
    }
    if (id == 0 || p > 1) {
      PrintMatrix(matrix, buf, n, m, 10, id, p);
      MPI_Barrier(G);
      PrintMatrix(inverse, buf, n, m, 10, id, p);
    }
    */
  }
  //
  /*
  MPI_Barrier(G);
  if (id == 0) {
      printf("AFTER FORWARD PASS\n");
  }
  if (id == 0 || p > 1) {
    PrintMatrix(matrix, buf, n, m, 10, id, p);
    MPI_Barrier(G);
    PrintMatrix(inverse, buf, n, m, 10, id, p);
  }
  */
  for (int epoch = blockNum - 1; epoch > 0; epoch--) {
    int dimCol = (epoch != k ? m : l);
    double *temp = buf + m * n;
    double *temp1 = buf + m * n + 3 * m * m;
    double *x = buf + m * n + m * m;
    double *result = buf + m * n + 2 * m * m;
    double *buffer = buf;
    if (id != epoch % p) {
      MPI_Bcast(buf, dimCol * (n - (blockNum - 1 - epoch) * m), MPI_DOUBLE, epoch % p, G);
    } else {
      MPI_Bcast(matrix + (epoch / p) * m * n,
                dimCol * (n - (blockNum - 1 - epoch) * m),  MPI_DOUBLE, epoch % p, G);
      buffer = matrix + (epoch / p) * m * n;
    }
    ReverseGauss(inverse, buffer, epoch, m, n, len, id, p, x, temp, temp1, result);
    /*
    if (id == 0) {
      printf("AFTER REVERSE GAUSS step= %d\n", epoch);
    }
    if (id == 0 || p > 1) {
      PrintMatrix(matrix, buf, n, m, 10, id, p);
      MPI_Barrier(G);
      PrintMatrix(inverse, buf, n, m, 10, id, p);
    }
    */
  }
  int tk, dk, tmp1, div = len / m;
  for (int i = 0; i < blockNum; i++) {
    tk = index[i];
    dk = (tk != k ? m : l);
    for (int j = 0; j < len * n; j += m * n) {
      tmp1 = (j != div * m * n ? m : l);
      double *temp = inverse + j + i * m * tmp1;
      PutBlock(matrix + j + tk * m * tmp1, temp, dk, tmp1);
    }
  }
  return 0;
}

////// INPUT_OUTPUT_TOOLS //////
