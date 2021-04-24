#include "head_MPI.h"

int main(int argc, char *argv[]){
  int    n;             // размерность матрицы
  int    m;             // размер блока
  int    r;             // кол-во выводимых значений в матрице
  int    s;             // номер применяемого алгоритма
  int    p;             // число процессов
  char   *filename;     // имя файла
  int    id;            // номер текущего процесса
  int    error = 0;
  double *matrix = 0;
  double *buf = 0;
  double *inverse = 0;
  int    *ind = 0;
  int    *index = 0;
  int    loc_err = 0, glob_err = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm G = MPI_COMM_WORLD;
  if (!(argc == 5 || argc == 6)) {
    if (!id) {
      printf("Invalid number of parametrs, usage: n, m, r, s, filename \n");
      printf("\n");
    }
    MPI_Finalize();
    return 0;
  }
  sscanf(argv[1], "%d", &n);
  sscanf(argv[2], "%d", &m);
  sscanf(argv[3], "%d", &r);
  sscanf(argv[4], "%d", &s);
  if (argv[5])
    filename = argv[5];
  else
    filename = 0;
  if (!CheckVars(n, m, p, r, s, filename)) {
    if (!id) {
      printf("Variables error, usage: n, m, r, s, filename \n Actual : n = %d, m = %d, p = %d, r = %d, s = %s ", n, m, p, r, filename);
      printf("\n");
    }
    MPI_Finalize();
    return 0;
  }
  if (!(matrix = new double[get_len(n, m, id, p) * n]))
    loc_err = 1;
  for (int i = 0; i < get_len(n, m, id, p) * n; i++) {
    matrix[i] = 0.;
  }
  if (!(inverse = new double[get_len(n, m, id, p) * n]))
    loc_err = 1;
  for (int i = 0; i < get_len(n, m, id, p) * n; i++) {
    inverse[i] = 0.;
  }
  int num = (n % m != 0 ? n / m + 1 : n / m); 
  if (!(buf = new double[2 * p * (num / p + 1) * m * m + 3 * m * m]))
    loc_err = 1;
  for (int i = 0; i < 2 * p * (num / p + 1) * m * m + 3 * m * m; i++) {
    buf[i] = 0.;
  }
  if (!(ind = new int[m]))
    loc_err = 1;
  for (int i = 0; i < m; i++) {
    ind[i] = 0;
  }
  if (!(index = new int[n]))
    loc_err = 1;
  for (int i = 0; i < n; i++) {
    index[i] = 0;
  }
  MPI_Allreduce(&loc_err, &glob_err, 1, MPI_INT, MPI_SUM, G);
  if (glob_err) {
    if (!id)
      printf("Memory error!\n");
    delete [] matrix;
    delete [] inverse;
    delete [] buf;
    delete [] ind;
    delete [] index;
    MPI_Finalize();
    return 0;
  }
  loc_err = InputMatrix(matrix, inverse, buf, n, m, s, filename, id, p);
  MPI_Allreduce(&loc_err, &glob_err, 1, MPI_INT, MPI_SUM, G);
  if (glob_err) {
    if (!id)
      printf("Error in init\n");
    delete [] matrix;
    delete [] inverse;
    delete [] buf;
    delete [] ind;
    delete [] index;
    MPI_Finalize();
    return 0;
  }
  if (!id) {
    printf("\n");
    printf("------ Original matrix : ------\n");
  }
  PrintMatrix(matrix, buf, n, m, r, id, p);
  if (!id) {
    printf("-------------------------------\n");
  }
  double tempNorm = Norma(matrix, buf, m, n, id, p);
  double resultNorm;
  MPI_Allreduce(&tempNorm, &resultNorm, 1, MPI_DOUBLE, MPI_MAX, G);
  double time = get_full_time();
  loc_err = Solve(matrix, inverse, buf, ind, index, m, n, resultNorm, id, p);
  time = get_full_time() - time;
  MPI_Allreduce(&loc_err, &glob_err, 1, MPI_INT, MPI_SUM, G);
  if (glob_err) {
    if (!id)
      printf("Matrix is not invertible !!!\n");
    delete [] matrix;
    delete [] inverse;
    delete [] buf;
    delete [] ind;
    delete [] index;
    MPI_Finalize();
    return 0;
  }
  if (!id) {
    printf("\n");
    printf("------ Solution : ------\n");
  }
  PrintMatrix(matrix, buf, n, m, r, id, p);
  if (!id) {
    printf("-------------------------------\n");
    printf("\n");
  }
  loc_err = InputMatrix(inverse, inverse, buf, n, m, s, filename, id, p);
  MPI_Allreduce(&loc_err, &glob_err, 1, MPI_INT, MPI_SUM, G);
  if (glob_err) {
    if (!id)
      printf("Error in init\n");
    delete [] matrix;
    delete [] inverse;
    delete [] buf;
    delete [] ind;
    delete [] index;
    MPI_Finalize();
    return 0;
  }
  resultNorm = ResultNorm(inverse, matrix, buf, m, n, id, p);
  if (!id) {
    printf("%s : residual = %e elapsed = %.2f for s = %d n = %d m = %d p = %d\n", argv[0], resultNorm, time, s, n, m, p);
  }
  delete [] matrix;
  delete [] inverse;
  delete [] buf;
  delete [] ind;
  delete [] index;
  MPI_Finalize();
  return 0;
}

