
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <math.h>
#include <unistd.h>

#define GET(x, y, size_x) ((x) + (y) * (size_x))

int me, me_cart, N, dessous, dessus, gauche, droite;
MPI_Status status;
MPI_Comm comm_cart;
int coords[2];
int dims[2], blockDim[2], periods[2];

unsigned int offset_x, offset_y;
unsigned int dim_x, dim_y;
MPI_Datatype top_border_type, side_border_type, top_border_type_remain, side_border_type_remain;

typedef float stencil_t;

/** conduction coeff used in computation */
static const stencil_t alpha = 0.02;

/** threshold for convergence */
static const stencil_t epsilon = 0.0001;

/** max number of steps */
static const int stencil_max_steps = 10000;

static stencil_t *values = NULL;
static stencil_t *prev_values = NULL;
static stencil_t *global_matrix = NULL;

static int size_x = STENCIL_SIZE;
static int size_y = STENCIL_SIZE;

void calculateDimensions(int N, int dims[2])
{
  int minDiff = N;
  int sqrtN = sqrt(N);

  for (int i = 1; i <= sqrtN; i++)
  {
    if (N % i == 0)
    {
      int currentDiff = abs(i - N / i);
      if (currentDiff < minDiff)
      {
        minDiff = currentDiff;
        dims[0] = i;
        dims[1] = N / i;
      }
    }
  }
}

/** init stencil values to 0, borders to non-zero */
static void stencil_init(void)
{
  values = malloc(size_x * size_y * sizeof(stencil_t));
  prev_values = malloc(size_x * size_y * sizeof(stencil_t));
  int x, y;
  for (x = 0; x < size_x; x++)
  {
    for (y = 0; y < size_y; y++)
    {
      values[x + size_x * y] = 0.0;
    }
  }
  for (x = 0; x < size_x; x++)
  {
    values[x + size_x * 0] = x;
    values[x + size_x * (size_y - 1)] = size_x - x;
  }
  for (y = 0; y < size_y; y++)
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

/** Display a (part of) the stencil values */
static void stencil_display(int x0, int x1, int y0, int y1, const stencil_t *matrix)
{
  int x, y;
  for (y = y0; y <= y1; y++)
  {
    for (x = x0; x <= x1; x++)
    {
      fprintf(stdout, "%8.5g ", matrix[x + STENCIL_SIZE * y]);
    }
    fprintf(stdout, "\n");
  }
}

/** Récupère la matrice globale à partir des sous-matrices des processus MPI */
static void gather_global_matrix(stencil_t *values)
{
  MPI_Datatype custom_type;
  MPI_Type_create_hvector(blockDim[1] + 1, blockDim[0] + 1, STENCIL_SIZE * sizeof(stencil_t), MPI_FLOAT, &custom_type);
  MPI_Type_commit(&custom_type);

  if (me_cart == 0 && global_matrix == NULL)
  {
    // Allocation de l'espace pour la matrice globale
    global_matrix = malloc(size_x * size_y * sizeof(stencil_t));
    int x, y;
    for (x = 0; x < size_x; x++)
    {
      global_matrix[x + size_x * 0] = x;
      global_matrix[x + size_x * (size_y - 1)] = size_x - x;
    }
    for (y = 0; y < size_y; y++)
    {
      global_matrix[0 + size_x * y] = y;
      global_matrix[size_x - 1 + size_x * y] = size_y - y;
    }
  }
  if (me_cart != 0)
  {
    // Chaque processus MPI envoie sa sous-matrice au processus de rang 0
    MPI_Send(
        values + GET(offset_x, offset_y, size_x), // Adresse de la sous-matrice locale
        1,                                        // Nombre d'éléments à envoyer
        custom_type,                              // Type des éléments
        0,                                        // Rang du processus récepteur (processus de rang 0)
        me_cart,                                  // Tag
        comm_cart                                 // Communicateur
    );
  }
  else
  {
    int tmp[2];
    for (int i = 1; i < N; i++)
    {
      MPI_Cart_coords(comm_cart, i, 2, tmp);
      MPI_Recv(
          global_matrix + GET(tmp[0] * blockDim[0], tmp[1] * blockDim[1], STENCIL_SIZE), // Adresse où placer les données de chaque processus
          1,                                                                             // Nombre d'éléments à recevoir de chaque processus
          custom_type,                                                                   // Type des éléments
          i,                                                                             // Rang du processus émetteur
          i,                                                                             // Tag
          comm_cart,                                                                     // Communicateur
          MPI_STATUS_IGNORE);
    }
  }

  MPI_Type_free(&custom_type);

  // Affichage de la matrice globale par le processus de rang 0
  if (me_cart == 0)
  {
    int x, y;
    for (y = 1 + offset_y; y < (1 + offset_y + dim_y); y++)
    {
      for (x = 1 + offset_x; x < (1 + offset_x + dim_x); x++)
      {
        global_matrix[x + size_x * y] = values[x + size_x * y];
      }
    }
    stencil_display(0, STENCIL_SIZE - 1, 0, STENCIL_SIZE - 1, global_matrix);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

/** compute the next stencil step, return 1 if computation has converged */
static int stencil_step(void)
{
  int convergence = 1;
  int convergence_globale;
  /* switch buffers */
  stencil_t *tmp = prev_values;
  prev_values = values;
  values = tmp;
  int x, y;

  MPI_Request req[8];
  MPI_Status stat[8];

  if (dessus != MPI_PROC_NULL)
  {
    MPI_Isend(&prev_values[GET(1 + offset_x, 1 + offset_y, size_x)], 1, top_border_type, dessus, 0, comm_cart, &req[0]);
    MPI_Irecv(&prev_values[GET(1 + offset_x, offset_y, size_x)], 1, top_border_type, dessus, 1, comm_cart, &req[1]);
  }
  else
  {
    req[0] = MPI_REQUEST_NULL;
    req[1] = MPI_REQUEST_NULL;
  }
  if (dessous != MPI_PROC_NULL)
  {
    MPI_Isend(&prev_values[GET(1 + offset_x, offset_y + dim_y, size_x)], 1, top_border_type, dessous, 1, comm_cart, &req[2]);
    MPI_Irecv(&prev_values[GET(1 + offset_x, offset_y + 1 + dim_y, size_x)], 1, top_border_type, dessous, 0, comm_cart, &req[3]);
  }
  else
  {
    req[2] = MPI_REQUEST_NULL;
    req[3] = MPI_REQUEST_NULL;
  }
  if (gauche != MPI_PROC_NULL)
  {
    MPI_Isend(&prev_values[GET(1 + offset_x, 1 + offset_y, size_x)], 1, side_border_type, gauche, 2, comm_cart, &req[4]);
    MPI_Irecv(&prev_values[GET(offset_x, 1 + offset_y, size_x)], 1, side_border_type, gauche, 3, comm_cart, &req[5]);
  }
  else
  {
    req[4] = MPI_REQUEST_NULL;
    req[5] = MPI_REQUEST_NULL;
  }
  if (droite != MPI_PROC_NULL)
  {
    MPI_Isend(&prev_values[GET(offset_x + dim_x, 1 + offset_y, size_x)], 1, side_border_type, droite, 3, comm_cart, &req[6]);
    MPI_Irecv(&prev_values[GET(offset_x + dim_x + 1, 1 + offset_y, size_x)], 1, side_border_type, droite, 2, comm_cart, &req[7]);
  }
  else
  {
    req[6] = MPI_REQUEST_NULL;
    req[7] = MPI_REQUEST_NULL;
  }

  MPI_Waitall(8, req, stat);

  for (y = 1 + offset_y; y < (1 + offset_y + dim_y); y++)
  {
    for (x = 1 + offset_x; x < (1 + offset_x + dim_x); x++)
    {
      values[x + size_x * y] =
          alpha * prev_values[x - 1 + size_x * y] +
          alpha * prev_values[x + 1 + size_x * y] +
          alpha * prev_values[x + size_x * (y - 1)] +
          alpha * prev_values[x + size_x * (y + 1)] +
          (1.0 - 4.0 * alpha) * prev_values[x + size_x * y];
      if (convergence && (fabs(prev_values[x + size_x * y] - values[x + size_x * y]) > epsilon))
      {
        convergence = 0;
      }
    }
  }

  MPI_Allreduce(&convergence, &convergence_globale, 1, MPI_INT, MPI_LAND, comm_cart);
  return convergence_globale;
}

int main(int argc, char **argv)
{
  /* --- Initialisation of MPI --- */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  /* --- Initialisation de la grille cartesienne --- */
  calculateDimensions(N, dims);
  if (me == 0)
    printf("Cartesian grid dimensions = %d %d\n", dims[0], dims[1]);
  blockDim[0] = (STENCIL_SIZE - 2) / dims[0] + ((STENCIL_SIZE - 2) % dims[0] != 0 ? 1 : 0);
  blockDim[1] = (STENCIL_SIZE - 2) / dims[1] + ((STENCIL_SIZE - 2) % dims[1] != 0 ? 1 : 0);
  periods[0] = 0;
  periods[1] = 0;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_cart);
  MPI_Comm_rank(MPI_COMM_WORLD, &me_cart);
  // On récupère les coordonnées dans la grille cartésienne
  MPI_Cart_coords(comm_cart, me, 2, coords);
  // Je stocke le rang des mes voisins
  MPI_Cart_shift(comm_cart, 1, -1, &dessous, &dessus);
  MPI_Cart_shift(comm_cart, 0, 1, &gauche, &droite);

  // On assigne les valeur de l'offset et que l'on doit calculer
  offset_x = coords[0] * blockDim[0];
  offset_y = coords[1] * blockDim[1];
  // Calcul des dimensions réelles inutles car on suppose que N divise STENCIL_SIZE, mais je l'ai laissé pour des futures versions...
  dim_x = (coords[0] * blockDim[0] + blockDim[0] <= (STENCIL_SIZE - 2)) ? blockDim[0] : (STENCIL_SIZE - 2) - coords[0] * blockDim[0];
  dim_y = (coords[1] * blockDim[1] + blockDim[1] <= (STENCIL_SIZE - 2)) ? blockDim[1] : (STENCIL_SIZE - 2) - coords[1] * blockDim[1];

  /* --- Création du type des bordures du côté et du dessus --- */
  // Type générique
  MPI_Type_vector(1, blockDim[0], size_x, MPI_FLOAT, &top_border_type);
  MPI_Type_vector(blockDim[1], 1, size_x, MPI_FLOAT, &side_border_type);
  MPI_Type_commit(&top_border_type);
  MPI_Type_commit(&side_border_type);
  // Type des bordures de droite et du bas: celles-ci peuvent être plus courtes;
  // Si les dimensions du tableau de sont pas divisibles. 
  // Exemple : STENCIL_SIZE=200 avec 6 processus MPI
  // On obtiens un grille 2x3 avec des blocs de 100x67 (on voudrait 100x66 pour le dernier)
  /* Non implémenté pour le moment
  MPI_Type_vector(1, STENCIL_SIZE - (dims[0] - 1) * blockDim[0] - 2, size_x, MPI_FLOAT, &top_border_type_remain);
  MPI_Type_vector(STENCIL_SIZE - (dims[1] - 1) * blockDim[1] - 2, 1, size_x, MPI_FLOAT, &side_border_type_remain);
  MPI_Type_commit(&top_border_type_remain);
  MPI_Type_commit(&side_border_type_remain);
  */

  stencil_init();

  struct timespec t1, t2;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  int s;
  for (s = 0; s < stencil_max_steps; s++)
  {
    int convergence = stencil_step();
    if (convergence)
    {
      break;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &t2);
  const double t_usec = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_nsec - t1.tv_nsec) / 1000.0;
  if (me_cart == 0)
  {
    printf("# size = %d\n", STENCIL_SIZE);
    printf("# steps = %d\n", s);
    printf("# time = %g usecs.\n", t_usec);
    printf("# gflops = %g\n", (float)(6 * size_x * size_y * s) / (t_usec * 1000));
  }

  stencil_free();
  MPI_Finalize();
  return 0;
}
