
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>


#define GET(x, y, size_x) ((x) + (y) * (size_x))
#define STENCIL_SIZE 30

int blockDim;
int nbBlock;
int me, N;
MPI_Status status;

typedef float stencil_t;

/** conduction coeff used in computation */
static const stencil_t alpha = 0.02;

/** threshold for convergence */
static const stencil_t epsilon = 0.0001;

/** max number of steps */
static const int stencil_max_steps = 100000;

static stencil_t*values = NULL;
static stencil_t*prev_values = NULL;

static int size_x = STENCIL_SIZE;
static int size_y = STENCIL_SIZE;

FILE * output;

/** init stencil values to 0, borders to non-zero */
static void stencil_init(void)
{
  values = malloc(size_x * size_y * sizeof(stencil_t));
  prev_values = malloc(size_x * size_y * sizeof(stencil_t));
  int x, y;
  for(x = 0; x < size_x; x++)
    {
      for(y = 0; y < size_y; y++)
        {
          values[x + size_x * y] = 0.0;
        }
    }
  for(x = 0; x < size_x; x++)
    {
      values[x + size_x * 0] = x;
      values[x + size_x * (size_y - 1)] = size_x - x;
    }
  for(y = 0; y < size_y; y++)
    {
      values[0 + size_x * y] = y;
      values[size_x - 1 + size_x * y] = size_y - y;
    }
  memcpy(prev_values, values, size_x * size_y * sizeof(stencil_t));
}

static void stencil_free(void)
{
  free(values);
  free(prev_values);
}

/** display a (part of) the stencil values */
static void stencil_display(int x0, int x1, int y0, int y1)
{
  int x, y;
  for(y = y0; y <= y1; y++)
    {
      for(x = x0; x <= x1; x++)
        {
          fprintf(output,"%8.5g ", values[x + size_x * y]);
        }
      fprintf(output,"\n");
    }
}

/** compute the next stencil step, return 1 if computation has converged */
static int stencil_step(void)
{
  int convergence = 1;
  /* switch buffers */
  stencil_t*tmp = prev_values;
  prev_values = values;
  values = tmp;
  int x, y;
  for(y = 1; y < size_y - 1; y++)
    {
      for(x = 1; x < size_x - 1; x++)
        {
          values[x + size_x * y] =
            alpha * prev_values[x - 1 + size_x * y] +
            alpha * prev_values[x + 1 + size_x * y] +
            alpha * prev_values[x + size_x * (y - 1)] +
            alpha * prev_values[x + size_x * (y + 1)] +
            (1.0 - 4.0 * alpha) * prev_values[x + size_x * y];
          if(convergence && (fabs(prev_values[x + size_x * y] - values[x + size_x * y]) > epsilon))
            {
              convergence = 0;
            }
        }
    }
  return convergence;
}

int main(int argc, char**argv)
{
  if (argc <3){
    fprintf(stderr,"Usage : stencil <file_name>\n");
    exit(-1);
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  // On considère que blockDim divise STENCIL_SIZE
  nbBlock = N;
  blockDim = STENCIL_SIZE/nbBlock;
  if(STENCIL_SIZE % blockDim != 0){
    fprintf(stderr,"La matrice ne se divise pas correctement et ne pourra pas être reconstituée correctement.\n");
    exit(-1);
  }
  FILE *file = NULL;
  // A la fin de l'execution on va rappatrier les données de tous les processus 
  if (me == 0){
    file = fopen(argv[1], "w");
    if (file == NULL) {
      perror("Error opening file");
      return 1; 
    }
    output = stdout;
  }
  stencil_init();

  //stencil_display(0, size_x - 1, 0, size_y - 1);

  struct timespec t1, t2;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  int s;
  for(s = 0; s < stencil_max_steps; s++)
  {
    int convergence = stencil_step();
    if(convergence)
    {
        break;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &t2);
  const double t_usec = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_nsec - t1.tv_nsec) / 1000.0;
  printf("# steps = %d\n", s);
  printf("# time = %g usecs.\n", t_usec);
  printf("# gflops = %g\n", (6 * size_x * size_y * s) / (t_usec * 1000));
  // TODO : récupération de la matrice complète
  printf("I'm %d/%d\n", me,N);
  if(me==0 && file != NULL){
    output = file;
    stencil_display(0, size_x - 1, 0, size_y - 1);
    fclose(file);
  }
  stencil_free(); 
  MPI_Finalize();
  return 0;
}
