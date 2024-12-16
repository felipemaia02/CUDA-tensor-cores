#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h> // Necessário para Tensor Cores

using namespace nvcuda;

/* Constantes */
#define CS2 (1.0 / 3.0)
#define t0 (4.0 / 9.0)
#define t1 (1.0 / 9.0)
#define t2 (1.0 / 36.0)

#define DEBUG

/* Estruturas */
typedef struct {
    int t_max;
    double density;
    double accel;
    double omega;
    double r_rey;
} s_properties;

typedef struct {
    unsigned int lx;
    unsigned int ly;
    unsigned int n;
    unsigned int n_obst;
    unsigned short int *obst;
    double *node;
    double *temp;
} s_lattice;

/* Alocação e liberação de memória */
void alloc_lattice(s_lattice *l) {
    l->obst = (unsigned short int *)calloc(l->lx * l->ly, sizeof(unsigned short int));
    l->node = (double *)calloc(l->lx * l->ly * l->n, sizeof(double));
    l->temp = (double *)calloc(l->lx * l->ly * l->n, sizeof(double));
}

void dealloc_lattice(s_lattice *l) {
    free(l->obst);
    free(l->node);
    free(l->temp);
}

void read_parameters(s_properties *p, char *file) {
    FILE *archive = fopen(file, "r");
    if (!archive) {
        fprintf(stderr, "Could not open parameters input file\nFile: %s\n\n", file);
        exit(EXIT_FAILURE);
    }
    fscanf(archive, "%d %lf %lf %lf %lf", &p->t_max, &p->density, &p->accel, &p->omega, &p->r_rey);
    fclose(archive);
}

void read_obstacles(s_lattice *l, char *file) {
    unsigned int i, j, max;
    int c = 0;
    FILE *archive = fopen(file, "r");
    if (!archive) {
        fprintf(stderr, "Could not read collision input file\nFile: %s\n\n", file);
        exit(EXIT_FAILURE);
    }
    fscanf(archive, "%u %u %u %u", &l->lx, &l->ly, &l->n, &max);
    alloc_lattice(l);
    while (c < max) {
        fscanf(archive, "%d %d", &i, &j);
        if (i <= l->lx && j <= l->ly && i >= 1 && j >= 1) {
            l->obst[(i - 1) * l->ly + (j - 1)] = 1;
        } else {
            fprintf(stderr, "Obstacle point[%d,%d] invalid!\n\n", i, j);
            exit(EXIT_FAILURE);
        }
        c++;
    }
    l->n_obst = max;
    fclose(archive);
}

/* Inicialização de densidades */
void init_density(s_lattice *l, double density) {
    unsigned int i, xy;
    double t_0 = density * t0;
    double t_1 = density * t1;
    double t_2 = density * t2;
    for (i = 0; i < l->lx * l->ly; ++i) {
        xy = l->n * i;
        l->node[xy] = t_0;
        l->node[xy + 1] = t_1;
        l->node[xy + 2] = t_1;
        l->node[xy + 3] = t_1;
        l->node[xy + 4] = t_1;
        l->node[xy + 5] = t_2;
        l->node[xy + 6] = t_2;
        l->node[xy + 7] = t_2;
        l->node[xy + 8] = t_2;
    }
}

/* Kernel usando Tensor Cores */
__global__ void relaxation_tensor_core(unsigned short int *obst, half *node, half *temp, float omega, unsigned int max) {
    __shared__ half shared_temp[16 * 16]; // Shared memory para `temp`
    __shared__ half shared_n_equ[16 * 16]; // Shared memory para `n_equ`
    __shared__ float shared_result[16 * 16]; // Shared memory para resultado acumulador

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> temp_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> n_equ_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < max) {
        if (!obst[idx]) {
            unsigned int xy = idx * 9;

            // Carregar dados da memória global para a shared memory
            for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
                shared_temp[i] = temp[xy + i];
                shared_n_equ[i] = temp[xy + i]; // Exemplo: dados para n_equ
            }
            __syncthreads();

            // Carregar shared memory nos fragmentos
            wmma::load_matrix_sync(temp_frag, shared_temp, 16);
            wmma::load_matrix_sync(n_equ_frag, shared_n_equ, 16);

            // Realiza o cálculo com Tensor Cores
            wmma::fill_fragment(c_frag, 0.0f);
            wmma::mma_sync(c_frag, temp_frag, n_equ_frag, c_frag);

            // Armazena o resultado na shared memory
            wmma::store_matrix_sync(shared_result, c_frag, 16, wmma::mem_row_major);
            __syncthreads();

            // Copiar resultados da shared memory para memória global
            for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
                node[xy + i] = __float2half(shared_result[i]);
            }
        }
        idx += blockDim.x * gridDim.x;
    }
}

double check_sum(s_lattice *l) {
    double sum = 0.0;
    for (int i = 0; i < l->lx * l->ly * l->n; ++i) {
        sum += l->node[i];
    }
    return sum;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s [file_properties] [file_collision]\n\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    s_properties properties;
    s_lattice lattice;

    read_parameters(&properties, argv[1]);
    read_obstacles(&lattice, argv[2]);
    init_density(&lattice, properties.density);

    half *d_temp_half, *d_node_half;
    unsigned short int *d_obst;
    cudaMalloc((void **)&d_obst, lattice.lx * lattice.ly * sizeof(unsigned short int));
    cudaMalloc((void **)&d_temp_half, lattice.lx * lattice.ly * lattice.n * sizeof(half));
    cudaMalloc((void **)&d_node_half, lattice.lx * lattice.ly * lattice.n * sizeof(half));

    for (int i = 0; i < lattice.lx * lattice.ly * lattice.n; i++) {
        lattice.temp[i] = __double2half(lattice.temp[i]);
        lattice.node[i] = __double2half(lattice.node[i]);
    }

    cudaMemcpy(d_obst, lattice.obst, lattice.lx * lattice.ly * sizeof(unsigned short int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_half, lattice.temp, lattice.lx * lattice.ly * lattice.n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_half, lattice.node, lattice.lx * lattice.ly * lattice.n * sizeof(half), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((lattice.lx * lattice.ly + threadsPerBlock.x - 1) / threadsPerBlock.x);

    #ifdef DEBUG
	printf("Initial sum: %.10lf\n", check_sum(&lattice));
	#endif

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    relaxation_tensor_core<<<blocksPerGrid, threadsPerBlock>>>(d_obst, d_node_half, d_temp_half, properties.omega, lattice.lx * lattice.ly);
    cudaEventRecord(stop);

    cudaMemcpy(lattice.node, d_node_half, lattice.lx * lattice.ly * lattice.n * sizeof(half), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_obst);
    cudaFree(d_node_half);
    cudaFree(d_temp_half);

	#ifdef DEBUG
	printf("End sum: %.10lf\n", check_sum(&lattice));
	#endif
	dealloc_lattice(&lattice);

    printf("lb_2d_cuda_tensor,%.4f,%d,%d,%d,%d,%f,%f,%f,%f\n",
           milliseconds / 1000,
           lattice.lx, lattice.ly,
           lattice.n,
           properties.t_max,
           properties.density,
           properties.accel,
           properties.omega,
           properties.r_rey);

    return 0;
}
