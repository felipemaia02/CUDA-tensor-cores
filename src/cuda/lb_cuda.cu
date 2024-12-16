/* Matheus S. Serpa
 * matheusserpa@gmail.com */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/* The sound velocity squared */
#define    CS2    (1.0/3.0)

/* The eq. coeff. in 2D */
#define    t0     (4.0/9.0)
#define    t1     (1.0/9.0)
#define    t2     (1.0/36.0)

#define DEBUG

/* struct contends macroscopic information */
typedef struct{
  int t_max;      /* Maximum number of iterations */
  double density; /* Fluid density per link in g/cm³ */
  double accel;   /* Macroscopic accelleration in cm²/s */
  double omega;   /* Relaxation parameter */
  double r_rey;   /* Linear dimension for Reynolds number in cm */
} s_properties;

/* lattice structure */
typedef struct {
  unsigned int lx; /* nodes number in axis x */
  unsigned int ly; /* nodes number in axis y */
  unsigned int n;  /* lattice dimension elements */
  unsigned int n_obst; /* nodes number is obstacle */
  unsigned short int *obst; /* Obstacle Array lx * ly */
  double *node; /* n-speed lattice  n * lx * ly */
  double *temp; /* temporarily storage of fluid densities */
} s_lattice;

/* Alloc memory space to the grid */
void alloc_lattice(s_lattice *l) {
	l->obst = (unsigned short int *) calloc(l->lx * l->ly, sizeof(unsigned short int));	
	l->node = (double *) calloc(l->lx * l->ly * l->n, sizeof(double));
	l->temp = (double *) calloc(l->lx * l->ly * l->n, sizeof(double));
}

/* Free memory space of the grid */
void dealloc_lattice(s_lattice *l){
	free(l->obst);
	free(l->node);
	free(l->temp);
	
}

/* Read parameters from file and save in the properties structure */
void read_parametrs(s_properties *p, char *file){
	FILE *archive = fopen(file, "r");

	if (!archive){
		fprintf(stderr, "Could not open parameters input file\nFile: %s\n\n", file);
		exit(EXIT_FAILURE);
	}

	fscanf(archive, "%d", &p->t_max);
	fscanf(archive, "%lf", &p->density);
	fscanf(archive, "%lf", &p->accel);
	fscanf(archive, "%lf", &p->omega);
	fscanf(archive, "%lf", &p->r_rey);

	fclose(archive);
	
}

/* Read obstacles from file and save in lattice structure */
void read_obstacles(s_lattice *l, char *file){
	unsigned int i, j, max = 0;
	int c = 0;
		
	FILE *archive = fopen(file, "r");
	if (!archive) {
		fprintf(stderr, "Could not read colision input file\nFile: %s\n\n", file);
        exit(EXIT_FAILURE);
	}

	fscanf(archive, "%u", &l->lx);	
	fscanf(archive, "%u", &l->ly);
	fscanf(archive, "%u", &l->n);
	fscanf(archive, "%d", &max);
	
	alloc_lattice(l);

	//Reading obstacle points
	while (c < max) {
		fscanf(archive, "%d %d", &i, &j);
		//Check if i and j are less than x_max and y_max
		if(i <= l->lx && j <= l->ly && i >= 1 && j >= 1)
			l->obst[(i - 1) * l->ly + j - 1] = 1;
		else{
			fprintf(stderr, "Obstacle point[%d,%d] invalid!\n\n", i, j);
		  	exit(EXIT_FAILURE);
		}
		c++;
	}
	l->n_obst = max;
	fclose(archive);

}

/* Initializes lattice node with three density levels */
void init_density(s_lattice *l, double density) {
	unsigned int i, xy;
	double t_0 = density * t0;
	double t_1 = density * t1;
	double t_2 = density * t2;
	
	for (i = 0; i < l->lx * l->ly; ++i) {
		xy = l->n*i;
		/* Zero velocity density */
		l->node[xy] = t_0; 
		/* Equilibrium densities for axis speeds */
		l->node[xy + 1] = t_1; /* East -> */
		l->node[xy + 2] = t_1; /* North ^ */
		l->node[xy + 3] = t_1; /* West <- */
		l->node[xy + 4] = t_1; /* South v */
		/* Equilibrium densities for diagonal speeds */
		l->node[xy + 5] = t_2; /* North-east />  */
		l->node[xy + 6] = t_2; /* North-west <\  */
		l->node[xy + 7] = t_2; /* South-west </  */
		l->node[xy + 8] = t_2; /* South-east \>  */
	}

}

/* Return time in a specific moment */
double crono(){
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

 static void GPUHandleError( cudaError_t err, const char *file, const int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define GPU_HANDLE_ERROR( err ) (GPUHandleError( err, __FILE__, __LINE__ ))

 __global__ void redistribute(unsigned short int *obst, double *node, double t_1, double t_2, unsigned int max) {
	unsigned int y = blockIdx.x * blockDim.x + threadIdx.x; // only in lattice.x = 0

	while(y < max){
		unsigned int xy = y*9;
		//Check to avoid negative densities
		if (!obst[y] && node[xy + 3] - t_1 > 0 && node[xy + 6] - t_2 > 0 && node[xy + 7] - t_2 > 0) {
			/* Increase east -> */
			node[xy + 1] += t_1;
			/* Decrease west <- */
			node[xy + 3] -= t_1;
			/* Increase north-east /> */
			node[xy + 5] += t_2;
			/* Decrease north-west <\ */
			node[xy + 6] -= t_2;
			/* Decrease south-west </ */
			node[xy + 7] -= t_2;
			/* Increase south-east \> */
			node[xy + 8] += t_2;
		}
		y += blockDim.x * gridDim.x;
	}
}

__global__ void propagate(double *node, double *temp, unsigned int lx, unsigned int ly) {	
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while(idx < lx * ly){
		unsigned int x = idx / ly;
		unsigned int y = idx % ly;
		unsigned int xy = x*ly*9 + y*9;
		unsigned int x_e, x_w, y_n, y_s;

		//Compute upper and right next neighbour nodes
		x_e = (x + 1) % lx;
		y_n = (y + 1) % ly;
		//Compute lower and left next neighbour nodes
		x_w = (x - 1 + lx) % lx;
		y_s = (y - 1 + ly) % ly;
		//Density propagation
		/* Zero */
		temp[xy] = node[xy];
		/* East -> */
		temp[x_e*ly*9 + y*9 + 1] = node[xy + 1];
		/* North ^ */
		temp[x*ly*9 + y_n*9 + 2] = node[xy + 2];
		/* West <- */
		temp[x_w*ly*9 + y*9 + 3] = node[xy + 3];
		/* South v */
		temp[x*ly*9 + y_s*9 + 4] = node[xy + 4];
		/* North-east /> */
		temp[x_e*ly*9 + y_n*9 + 5] = node[xy + 5];
		/* North-west <\ */
		temp[x_w*ly*9 + y_n*9 + 6] = node[xy + 6];
		/* South-west </ */
		temp[x_w*ly*9 + y_s*9 + 7] = node[xy + 7];
		/* South-east \> */
		temp[x_e*ly*9 + y_s*9 + 8] = node[xy + 8];

		idx += blockDim.x * gridDim.x;
	}
}

__global__ void bounceback(unsigned short int *obst, double *node, double *temp, unsigned int max) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while(idx < max){
		if(obst[idx]){
			unsigned int xy = idx * 9;
			/* East -> */
			node[xy + 1] = temp[xy + 3];
			/* North ^ */
			node[xy + 2] = temp[xy + 4];
			/* West <- */
			node[xy + 3] = temp[xy + 1];
			/* South v */
			node[xy + 4] = temp[xy + 2];
			/* North-east /> */
			node[xy + 5] = temp[xy + 7];
			/* North-west <\ */
			node[xy + 6] = temp[xy + 8];
			/* South-west </ */
			node[xy + 7] = temp[xy + 5];
			/* South-east \> */
			node[xy + 8] = temp[xy + 6];
		}
		idx += blockDim.x * gridDim.x;
	}
}

//Relaxation
__global__ void relaxation(unsigned short int *obst, double *node, double *temp, double omega, unsigned int max) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	while(idx < max){
		if(!obst[idx]){
			double u_x, u_y;
			double u_n[9], n_equ[9], u_squ, d_loc;
			unsigned int j, xy = idx * 9;

			d_loc = 0.0;
			for (j = 0; j < 9; ++j) {
				d_loc += temp[xy + j];
			}

			//x-, and y- velocity components
			/* ->  +  />  +  \>  -  <-  -  <\  -  </ */
			u_x = (temp[xy + 1] + temp[xy + 5] + temp[xy + 8] - (temp[xy + 3] + temp[xy + 6] + temp[xy + 7])) / d_loc;
			/*  ^  +  />  +  <\  -  v  -  </  -  \> */
			u_y = (temp[xy + 2] + temp[xy + 5] + temp[xy + 6] - (temp[xy + 4] + temp[xy + 7] + temp[xy + 8])) / d_loc;

			//Square velocity
			u_squ = u_x * u_x + u_y * u_y;

			//n- velocity compnents
			//Only 3 speeds would be necessary
			u_n[1] = u_x;
			u_n[2] = u_y;
			u_n[3] = -u_x;
			u_n[4] = -u_y;
			u_n[5] = u_x + u_y;
			u_n[6] = -u_x + u_y;
			u_n[7] = -u_x - u_y;
			u_n[8] = u_x - u_y;
			
			//Zero velocity density
			n_equ[0] = t0 * d_loc * (1.0 - u_squ / (2.0 * CS2));
			//Axis speeds: factor: t1
			n_equ[1] = t1 * d_loc * (1.0 + u_n[1] / CS2 + u_n[1] * u_n[1] / (2.0 * CS2 * CS2) - u_squ / (2.0 * CS2));
			n_equ[2] = t1 * d_loc * (1.0 + u_n[2] / CS2 + u_n[2] * u_n[2] / (2.0 * CS2 * CS2) - u_squ / (2.0 * CS2));
			n_equ[3] = t1 * d_loc * (1.0 + u_n[3] / CS2 + u_n[3] * u_n[3] / (2.0 * CS2 * CS2) - u_squ / (2.0 * CS2));
			n_equ[4] = t1 * d_loc * (1.0 + u_n[4] / CS2 + u_n[4] * u_n[4] / (2.0 * CS2 * CS2) - u_squ / (2.0 * CS2));

			//Diagonal speeds: factor t2
			n_equ[5] = t2 * d_loc * (1.0 + u_n[5] / CS2 + u_n[5] * u_n[5] / (2.0 * CS2 * CS2) - u_squ / (2.0 * CS2));
			n_equ[6] = t2 * d_loc * (1.0 + u_n[6] / CS2 + u_n[6] * u_n[6] / (2.0 * CS2 * CS2) - u_squ / (2.0 * CS2));
			n_equ[7] = t2 * d_loc * (1.0 + u_n[7] / CS2 + u_n[7] * u_n[7] / (2.0 * CS2 * CS2) - u_squ / (2.0 * CS2));
			n_equ[8] = t2 * d_loc * (1.0 + u_n[8] / CS2 + u_n[8] * u_n[8] / (2.0 * CS2 * CS2) - u_squ / (2.0 * CS2));

			//Relaxation step
			for (j = 0; j < 9; ++j) 
				node[xy + j] = temp[xy + j] + omega * (n_equ[j] - temp[xy + j]);
		}
		idx += blockDim.x * gridDim.x;
	}
}

double check_sum(s_lattice *l){
	unsigned int i;
	double sum = 0;
	for (i = 0; i < l->lx * l->ly * 9; ++i)
		sum += l->node[i];

	return sum;
}


int main(int argc, char **argv) {
	//Iteration counter
	int time;

	//Execution Time
	float execution_time = 0.0;

	//Input structure
	s_properties properties;

	//Lattice structure
	s_lattice lattice;
	
	/*
		Flags
		-d    debug execution
		-o    output save
		-m    matlab plot
	*/

	//Checking arguments
	if (argc < 3) {
		fprintf(stderr, "Usage: %s [file_properties] [file_colision]\n\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	
	read_parametrs(&properties, argv[1]);

	read_obstacles(&lattice, argv[2]);
	
	init_density(&lattice, properties.density);
	

	double t_1 = properties.density * properties.accel * t1;
	double t_2 = properties.density * properties.accel * t2;

	/*************** BEGINNING OF CUDA CODE ***************/
	// Workstation UNIPAMPA: 0 to Tesla C2075, 1 to Quadro 5000
	GPU_HANDLE_ERROR(cudaSetDevice(0));

	// Device Memory
	unsigned short int *d_obst;
	double *d_node, *d_temp;

#if 1
	int blockSize;
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, redistribute, 0, 0);
//	fprintf(stdout, "redistribute grid size %d block size %d \n", minGridSize, blockSize);
	int BLOCK_REDISTRIBUTE = blockSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, propagate, 0, 0);
//	fprintf(stdout, "propagate grid size %d block size %d \n", minGridSize, blockSize);
	int  BLOCK_PROPAGATE = blockSize;
	
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, bounceback, 0, 0);
//	fprintf(stdout, "bounceback grid size %d block size %d \n", minGridSize, blockSize);
	int BLOCK_BOUNCEBACK = blockSize;
	
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, relaxation, 0, 0);
//	fprintf(stdout, "relaxation grid size %d block size %d \n", minGridSize, blockSize);
	int BLOCK_RELAXATION = blockSize;
#else
	// Blocks and Grids.
	int BLOCK_REDISTRIBUTE = atoi(argv[3]), BLOCK_PROPAGATE = atoi(argv[3]);
	int BLOCK_BOUNCEBACK = atoi(argv[3]), BLOCK_RELAXATION = atoi(argv[3]);
#endif

	unsigned int const GRID_REDISTRIBUTE = (lattice.ly + BLOCK_REDISTRIBUTE - 1) / BLOCK_REDISTRIBUTE;
	unsigned int const GRID_PROPAGATE = (lattice.lx * lattice.ly + BLOCK_PROPAGATE - 1) / BLOCK_PROPAGATE;
	unsigned int const GRID_BOUNCEBACK = (lattice.lx * lattice.ly + BLOCK_BOUNCEBACK - 1) / BLOCK_BOUNCEBACK;
	unsigned int const GRID_RELAXATION = (lattice.lx * lattice.ly + BLOCK_RELAXATION - 1) / BLOCK_RELAXATION;

	#ifdef DEBUG
	printf("Initial sum: %.10lf\n", check_sum(&lattice));
	#endif

	// Timer
	cudaEvent_t start, stop;

	GPU_HANDLE_ERROR(cudaEventCreate(&start));
	GPU_HANDLE_ERROR(cudaEventCreate(&stop));

	// Memory alloc
	GPU_HANDLE_ERROR(cudaMalloc((void **) &d_obst, lattice.lx * lattice.ly * sizeof(unsigned short int)));
	GPU_HANDLE_ERROR(cudaMalloc((void **) &d_node, lattice.lx * lattice.ly * lattice.n * sizeof(double)));
	GPU_HANDLE_ERROR(cudaMalloc((void **) &d_temp, lattice.lx * lattice.ly * lattice.n * sizeof(double)));

	// Synchronize
	GPU_HANDLE_ERROR(cudaDeviceSynchronize());

	GPU_HANDLE_ERROR(cudaEventRecord(start, 0));
	//GPU_HANDLE_ERROR(cudaEventSynchronize(stop));
	// Memory Copy
	GPU_HANDLE_ERROR(cudaMemcpy(d_obst, lattice.obst, lattice.lx * lattice.ly * sizeof(unsigned short int), cudaMemcpyHostToDevice));
	GPU_HANDLE_ERROR(cudaMemcpy(d_node, lattice.node, lattice.lx * lattice.ly * lattice.n * sizeof(double), cudaMemcpyHostToDevice));
	GPU_HANDLE_ERROR(cudaMemcpy(d_temp, lattice.temp, lattice.lx * lattice.ly * lattice.n * sizeof(double), cudaMemcpyHostToDevice));

	for (time = 1; time <= properties.t_max; time++) {
	    redistribute<<<GRID_REDISTRIBUTE, BLOCK_REDISTRIBUTE>>>(d_obst, d_node, t_1, t_2, lattice.ly);
//	    GPU_HANDLE_ERROR(cudaDeviceSynchronize()); GPU_HANDLE_ERROR(cudaGetLastError());
 
	    propagate<<<GRID_PROPAGATE, BLOCK_PROPAGATE>>>(d_node, d_temp, lattice.lx, lattice.ly);
//	    GPU_HANDLE_ERROR(cudaDeviceSynchronize()); GPU_HANDLE_ERROR(cudaGetLastError());
	   
	    bounceback<<<GRID_BOUNCEBACK, BLOCK_BOUNCEBACK>>>(d_obst, d_node, d_temp, lattice.lx * lattice.ly);
//	    GPU_HANDLE_ERROR(cudaDeviceSynchronize()); GPU_HANDLE_ERROR(cudaGetLastError());
	   
	    relaxation<<<GRID_RELAXATION, BLOCK_RELAXATION>>>(d_obst, d_node, d_temp, properties.omega, lattice.lx * lattice.ly);
//	    GPU_HANDLE_ERROR(cudaDeviceSynchronize()); GPU_HANDLE_ERROR(cudaGetLastError());
	}
	// Copy device to host
	GPU_HANDLE_ERROR(cudaMemcpy(lattice.node, d_node, lattice.lx * lattice.ly * lattice.n * sizeof(double), cudaMemcpyDeviceToHost));
	// Synchronize
//	GPU_HANDLE_ERROR(cudaDeviceSynchronize());
	GPU_HANDLE_ERROR(cudaEventRecord(stop, 0));
	GPU_HANDLE_ERROR(cudaEventSynchronize(stop));
  //	GPU_HANDLE_ERROR(cudaDeviceSynchronize());
//	GPU_HANDLE_ERROR(cudaGetLastError());
    GPU_HANDLE_ERROR(cudaEventElapsedTime(&execution_time, start, stop));

    execution_time /= 1000;

    // Stop events
  	GPU_HANDLE_ERROR(cudaEventDestroy(start));

	GPU_HANDLE_ERROR(cudaEventDestroy(stop));

	// Free device memory
	GPU_HANDLE_ERROR(cudaFree(d_obst));
	GPU_HANDLE_ERROR(cudaFree(d_node));
	GPU_HANDLE_ERROR(cudaFree(d_temp));

	// Device Reset
	GPU_HANDLE_ERROR(cudaDeviceReset());
	/*************** END OF CUDA CODE ***************/
	
	#ifdef DEBUG
	printf("End sum: %.10lf\n", check_sum(&lattice));
	#endif
	dealloc_lattice(&lattice);

	//fprintf(stderr, "%.10f\n", execution_time);
	fprintf(stdout, "lb_2d_cuda,%.4f,%d,%d,%d,%d,%f,%f,%f,%f\n",
		execution_time,
		lattice.lx, lattice.ly,
		lattice.n,
		properties.t_max,
		properties.density,
		properties.accel,
		properties.omega,
		properties.r_rey);

	return 0;
}
