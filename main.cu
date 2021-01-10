#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define DIM 3
#define CENTROID_COUNT 2
#define POINTS_COUNT 2000000
#define POINTS_RANGE 50
#define ITERS 2
#define NORMAL_DIST false

#define DEBUG false
#define PRINT false

///// UTILS
#define checkCuda(ans) { checkCudaError((ans), __LINE__); }

void checkCudaError(cudaError_t cudaStatus, int line) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Line %d CUDA Error %d: %s\n", line, cudaStatus, cudaGetErrorString(cudaStatus));
    }
}

void print_timediff(const char* prefix, const struct timespec& start, const
struct timespec& end)
{
    double milliseconds = end.tv_nsec >= start.tv_nsec
                          ? (end.tv_nsec - start.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec) * 1e3
                          : (start.tv_nsec - end.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec - 1) * 1e3;
    printf("%s: %lf milliseconds\n", prefix, milliseconds);
}

//////////////////////////////////////// CPU ////////////////////////////////////////

double distanceBetweenTwoPoints(int *points, double *centroids, int point, int centroid) {

    //printf("Distance between [%d %d %d] and [%d %d %d] ",
    // points[point * DIM ],points[point * DIM +1 ],points[point * DIM +2 ],
    // centroids[centroid * DIM ],centroids[centroid * DIM +1 ],centroids[centroid * DIM +2 ]
    // );

    int sum = 0;
    for (int i = 0; i < DIM; i++) {
        sum += std::pow((double) points[point * DIM + i] - centroids[centroid * DIM + i], 2);
    }
    //printf(" = %f\n", sqrt(sum));
    return std::sqrt(sum);
}

void randomCentroids(const int *points, double *centroids, int size) {
    std::vector<double> copy(size);
    for (int i = 0; i < size; i++) {
        copy.at(i) = points[i];
    }

    for (int i = 0; i < CENTROID_COUNT; i++) {
        int index = INT32_MAX;
        while (index + DIM - 1 > copy.size()) {
            index = (random() % copy.size()) * DIM;
        }
        std::vector<double>::iterator it1, it2;
        it1 = (copy.begin() + index);
        it2 = (copy.begin() + index + DIM);
        for (int j = 0; j < DIM; j++) {
            centroids[i * DIM + j] = copy.at(index + j);
        }
        copy.erase(it1, it2);
    }
}

void kMeansCPU(int *points, int size) {

    // step 0: choose n random points
    double centroids[DIM * CENTROID_COUNT];
    randomCentroids(points, centroids, size);
    int pointToCentroid[size / DIM];
    int iters = 0;

    while (iters < ITERS) {
        // step 1: assign each point to the closest centroid
        for (int i = 0; i < size / DIM; i++) {
            double minDist = MAXFLOAT;
            int currentCentroid;
            for (int j = 0; j < CENTROID_COUNT; j++) {
                double dist = distanceBetweenTwoPoints(points, centroids, i, j);
                if (minDist > dist) {
                    minDist = dist;
                    currentCentroid = j;
                }
            }
            pointToCentroid[i] = currentCentroid;
        }


        // step 2: recompute centroids
        int countsPerCluster[CENTROID_COUNT] = {};
        int sumPerCluster[CENTROID_COUNT * DIM] = {};

        for (int i = 0; i < POINTS_COUNT; i++) { //point
            int c = pointToCentroid[i];
            countsPerCluster[c] += 1;
            for (int cDim = 0; cDim < DIM; cDim++) {
                sumPerCluster[c * DIM + cDim] += points[i * DIM + cDim];
            }
        }
        // recompute
        for (int i = 0; i < CENTROID_COUNT; i++) {
            for (int j = 0; j < DIM; j++) {
                centroids[i * DIM + j] = (double) sumPerCluster[i * DIM + j] / (double) countsPerCluster[i];
            }
        }

        // repeat step 1 and 2 until convergence (no point changed its cluster)
        iters++;
    }
#if PRINT
    for (int i = 0; i < size / DIM; i++) {
        std::cout << pointToCentroid[i] << ",";
    }
    std::cout << std::endl;
#endif
}

void randomArray(int size, int *array, int range = POINTS_RANGE) {
    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution<> d{POINTS_RANGE / 2, 2};
    std::normal_distribution<> d2{POINTS_RANGE / 5, 2};


    for (int i = 0; i < size; i++) {
#if NORMAL_DIST
        if (i < size / 2)
            array[i] = (int) d(gen) % range;
        else
            array[i] = (int) d2(gen) % range;
#else
        array[i] = random() % range;
#endif
    }
}

void printPoints(int size, const int *points) {
    for (size_t i = 0; i < size; i++) {
        std::cout << points[i] << ", ";
    }
    printf("\n");
}

//////////////////////////////////////// CUDA sol 1 (NOT OPTIMAL)////////////////////////////////////////

__global__ void distance(const int *points, double *dists, const double *centroids) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    //bounds check
    if (idx >= POINTS_COUNT) return;

    for (int currCentroid = 0; currCentroid < CENTROID_COUNT; currCentroid++) {
        double sum = 0;
        for (int currDim = 0; currDim < DIM; currDim++) {
            sum += std::pow((double) points[idx * DIM + currDim] - centroids[currCentroid * DIM + currDim], 2);
        }
        dists[idx * CENTROID_COUNT + currCentroid] = std::sqrt(sum);
    }
}

__global__ void assign(const double *dists, int *p2c) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= POINTS_COUNT) return;
    double minDist = MAXFLOAT;
    int minCentroid = -1;
    for (int dist = 0; dist < CENTROID_COUNT; dist++) {
        if (dists[idx * CENTROID_COUNT + dist] < minDist) {
            minDist = dists[idx * CENTROID_COUNT + dist];
            minCentroid = dist;
        }
    }
    p2c[idx] = minCentroid;
}

__global__ void newCentroids(const int *points, const int *p2c, double *centroids) {
    const uint centroidIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (centroidIdx >= CENTROID_COUNT) return;
    int sum[DIM] = {0};
    int count = 0;
    for (int i = 0; i < POINTS_COUNT; i++) {
        if (p2c[i] == centroidIdx) {
            for (int curDim = 0; curDim < DIM; curDim++) {
                sum[curDim] += points[i * DIM + curDim];
            }
            count++;
        }
    }
    //compute new centroid

    for (int curDim = 0; curDim < DIM; curDim++) {
        centroids[centroidIdx * DIM + curDim] = (double) sum[curDim] / (double) count;
    }
}

void kMeansCUDA(int *points) {
    dim3 block(32, 4);
    dim3 grid(block.x * block.y, ceil((double) POINTS_COUNT / (block.x * block.y)));

    double *dists;
    double *centroids;
    int *pointToCentroid;
    int *clusterSizes;
    checkCuda(cudaMallocManaged(&dists, CENTROID_COUNT * POINTS_COUNT * sizeof(double)));
    checkCuda(cudaMallocManaged(&centroids, CENTROID_COUNT * DIM * sizeof(double)));
    checkCuda(cudaMallocManaged(&pointToCentroid, POINTS_COUNT * sizeof(int)));
    checkCuda(cudaMallocManaged(&clusterSizes, CENTROID_COUNT * sizeof(int)));

    randomCentroids(points, centroids, POINTS_COUNT * DIM);
    int iter = 0;
    while (iter < ITERS) {
#if DEBUG
        std::cout <<"CENTROIDS:"<< std::endl;
        for (int i = 0; i < CENTROID_COUNT * DIM; i++) {
            std::cout << centroids[i] << ",";
        }
        std::cout << std::endl;
#endif
        //for each point calculate distance
        cudaMemset(dists, 0.0, CENTROID_COUNT * POINTS_COUNT * sizeof(double));
        distance<<<grid, block>>>(points, dists, centroids);
#if DEBUG
        checkCuda(cudaDeviceSynchronize());
        std::cout <<"DISTS:"<< std::endl;
        for (int i = 0; i < CENTROID_COUNT * POINTS_COUNT; i++) {
            std::cout << dists[i] << ",";
        }
        std::cout << std::endl;
#endif
        //assign centroid to each point
        cudaMemset(dists, 0, POINTS_COUNT * sizeof(int));
        assign<<<grid, block>>>(dists, pointToCentroid);
#if DEBUG
        checkCuda(cudaDeviceSynchronize());
        std::cout <<"POINT-TO-CENTROID:"<< std::endl;
        for (int i = 0; i < POINTS_COUNT; i++) {
            std::cout << pointToCentroid[i] << ",";
        }
        std::cout << std::endl;
#endif
        //recalculate each centroid
        cudaMemset(centroids, 0.0, CENTROID_COUNT * DIM * sizeof(double));
        newCentroids<<<1, CENTROID_COUNT>>>(points, pointToCentroid, centroids);
#if DEBUG
        checkCuda(cudaDeviceSynchronize());
        std::cout <<"NEW CENTROIDS:"<< std::endl;
        for (int i = 0; i < CENTROID_COUNT * DIM; i++) {
            std::cout << centroids[i] << ",";
        }
        std::cout << std::endl;
#endif
        iter++;
    }
#if PRINT
    checkCuda(cudaDeviceSynchronize());
    std::cout <<"POINT-TO-CENTROID:"<< std::endl;
    for (int i = 0; i < POINTS_COUNT; i++) {
        std::cout << pointToCentroid[i] << ",";
    }
    std::cout << std::endl;
#endif

    // cleanup
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(centroids));
    checkCuda(cudaFree(dists));
}
//////////////////////////////////////// CUDA sol 2 ////////////////////////////////////////

__device__ float distance_squared(float x1, float x2, float y1, float y2, float z1, float z2) {
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
}

__global__ void move_centroids(float* d_centroids_x, float* d_centroids_y, float* d_centroids_z, float* d_new_centroids_x, float* d_new_centroids_y, float* d_new_centroids_z, float* d_counters, int number_of_clusters)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int local_tid = threadIdx.x;
    extern __shared__ float this_centroid_x[];
    float* this_centroid_y = (float*)this_centroid_x + blockDim.x; //our current block dim is our previous gridDim
    float* this_centroid_z = (float*)this_centroid_x + 2 * + blockDim.x;
    float* this_centroid_counters = (float*)this_centroid_x + 3 * + blockDim.x;

    this_centroid_x[local_tid] = d_new_centroids_x[tid];
    this_centroid_y[local_tid] = d_new_centroids_y[tid];
    this_centroid_z[local_tid] = d_new_centroids_z[tid];
    this_centroid_counters[local_tid] = d_counters[tid];
    __syncthreads();

    //TODO reduce on values -> works only when number of blocks is some power of 2
    for(int d = blockDim.x/2; d > 0; d>>=1) {
        if(local_tid < d) {
            this_centroid_x[local_tid] += this_centroid_x[local_tid + d];
            this_centroid_y[local_tid] += this_centroid_y[local_tid + d];
            this_centroid_z[local_tid] += this_centroid_z[local_tid + d];
            this_centroid_counters[local_tid] += this_centroid_counters[local_tid + d];
        }
        __syncthreads();
    }

    //assignment of new values
    if(local_tid == 0) {
        const float count = this_centroid_counters[local_tid];
        d_centroids_x[blockIdx.x] = this_centroid_x[local_tid]/count;
        d_centroids_y[blockIdx.x] = this_centroid_y[local_tid]/count;
        d_centroids_z[blockIdx.x] = this_centroid_z[local_tid]/count;
    }
    __syncthreads();

    d_new_centroids_x[tid] = 0;
    d_new_centroids_y[tid] = 0;
    d_new_centroids_z[tid] = 0;
    d_counters[tid] = 0;
}

__global__ void distances_calculation(float* d_points_x, float* d_points_y, float* d_points_z, float* d_centroids_x, float* d_centroids_y, float* d_centroids_z, float* d_new_centroids_x, float* d_new_centroids_y, float* d_new_centroids_z, float* d_counters)
{
    //this version works on atomics with 7x speedup
    extern __shared__ float local_centroids[];
    float* s_array = (float*)local_centroids + DIM * CENTROID_COUNT;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    bool has_element = tid < POINTS_COUNT;
    if(tid >= POINTS_COUNT) return;
    int currentCentroid = 0;
    float _x; float _y; float _z; float currentDistance;

    if(has_element) {
        currentDistance = INFINITY;
        _x = d_points_x[tid];
        _y = d_points_y[tid];
        _z = d_points_z[tid];
    }

    if(local_tid < CENTROID_COUNT) {
        local_centroids[local_tid]= d_centroids_x[local_tid];
        local_centroids[local_tid + CENTROID_COUNT]= d_centroids_y[local_tid];
        local_centroids[local_tid + CENTROID_COUNT + CENTROID_COUNT]= d_centroids_z[local_tid];
    }
    __syncthreads();
    if(has_element) {
        for(int i = 0; i < CENTROID_COUNT; ++i) {
            const float _distance = distance_squared(_x, local_centroids[i], _y,local_centroids[i + CENTROID_COUNT] , _z, local_centroids[i + 2*CENTROID_COUNT]);
            if(_distance < currentDistance) {
                currentCentroid = i;
                currentDistance = _distance;
            }
        }
    }

    __syncthreads();

    int offset = blockDim.x; //
    int first = local_tid; // x
    int second = local_tid + offset; // y
    int third = local_tid + 2 * offset; //z
    int fourth = local_tid + 3 * offset; //counters

    for(int i = 0; i < number_of_clusters; ++i) {
        s_array[first] = (has_element && (currentCentroid == i)) ? _x : 0;
        s_array[second] = (has_element && (currentCentroid == i)) ? _y : 0;
        s_array[third] = (has_element && (currentCentroid == i)) ? _z : 0;
        s_array[fourth] = (has_element && (currentCentroid == i)) ? 1 : 0;
        __syncthreads();

        for(int d = blockDim.x/2; d > 0; d>>=1) {
            if(local_tid < d) {
                s_array[first] += s_array[first + d];
                s_array[second] += s_array[second + d];
                s_array[third] += s_array[third + d];
                s_array[fourth] += s_array[fourth + d];
            }
            __syncthreads();
        }

        if(local_tid == 0) {
            d_new_centroids_x[i * gridDim.x + blockIdx.x] = s_array[first];
            d_new_centroids_y[i * gridDim.x + blockIdx.x] = s_array[second];
            d_new_centroids_z[i * gridDim.x + blockIdx.x] = s_array[third];
            d_counters[i * gridDim.x + blockIdx.x] = s_array[fourth];
        }
        __syncthreads();
    }

}
void optimalKMeansCUDA(Points points, Points centroids, int number_of_examples, int iterations, int number_of_clusters){
    float* d_points_x;
    float* d_points_y;
    float* d_points_z;
    float* d_centroids_x;
    float* d_centroids_y;
    float* d_centroids_z;
    float* d_new_centroids_x;
    float* d_new_centroids_y;
    float* d_new_centroids_z;
    float* d_counters;
    int num_threads = 1024;
    int num_blocks = (number_of_examples + num_threads - 1) / num_threads;
    //we will be accessing memory structures concurrently -> AoS makes more sense than SoA
    cudaMallocManaged(&d_points_x, points.size()*sizeof(float));
    cudaMallocManaged(&d_points_y, points.size()*sizeof(float));
    cudaMallocManaged(&d_points_z, points.size()*sizeof(float));
    cudaMallocManaged(&d_centroids_x, centroids.size()*sizeof(float));
    cudaMallocManaged(&d_centroids_y, centroids.size()*sizeof(float));
    cudaMallocManaged(&d_centroids_z, centroids.size()*sizeof(float));
    cudaMallocManaged(&d_new_centroids_x, num_blocks*centroids.size()*sizeof(float));
    cudaMallocManaged(&d_new_centroids_y, num_blocks*centroids.size()*sizeof(float));
    cudaMallocManaged(&d_new_centroids_z, num_blocks*centroids.size()*sizeof(float));

    cudaMallocManaged(&d_counters, num_blocks*centroids.size()*sizeof(float));
    for(int i = 0; i < number_of_examples; ++i) {
        d_points_x[i] = points[i].x;
        d_points_y[i] = points[i].y;
        d_points_z[i] = points[i].z;
    }
    for(int i = 0; i < number_of_clusters; ++i) {
        d_centroids_x[i] = centroids[i].x;
        d_centroids_y[i] = centroids[i].y;
        d_centroids_z[i] = centroids[i].z;
        d_new_centroids_x[i] = 0;
        d_new_centroids_y[i] = 0;
        d_new_centroids_z[i] = 0;
    }


    int mem = 3*number_of_clusters*sizeof(float) + 4*num_threads*sizeof(float);
    int mem2 = 4*num_blocks*sizeof(float);
    printf("Starting parallel kmeans\n");
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < iterations; ++i) {
        distances_calculation<<<num_blocks, num_threads, mem>>>(d_points_x, d_points_y, d_points_z, d_centroids_x, d_centroids_y, d_centroids_z, d_new_centroids_x, d_new_centroids_y, d_new_centroids_z, d_counters, number_of_examples, number_of_clusters);
        checkCuda( cudaPeekAtLastError() );
        checkCuda( cudaDeviceSynchronize() );
        //for(int i = 0; i < number_of_clusters; ++i) printf("centroid sums: %f %f %f\n", d_new_centroids_x[i], d_new_centroids_y[i], d_new_centroids_z[i]);
        move_centroids<<<number_of_clusters, num_blocks, mem2>>>(d_centroids_x, d_centroids_y, d_centroids_z, d_new_centroids_x, d_new_centroids_y, d_new_centroids_z, d_counters, number_of_clusters);
        checkCuda( cudaPeekAtLastError() );
        checkCuda( cudaDeviceSynchronize() );

    }
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    printf("\nElapsed time in milliseconds : %f ms.\n\n", duration);

    for (int i = 0; i < number_of_clusters; i++){
        printf("%f  %f  %f", d_centroids_x[i], d_centroids_y[i], d_centroids_z[i]);     printf("\n");}


    cudaFree(d_points_x);
    cudaFree(d_points_y);
    cudaFree(d_points_z);
    cudaFree(d_centroids_x);
    cudaFree(d_centroids_y);
    cudaFree(d_centroids_z);
    cudaFree(d_new_centroids_x);
    cudaFree(d_new_centroids_y);
    cudaFree(d_new_centroids_z);
    cudaFree(d_counters);

}

int main() {
    struct timespec start, end;

    int size = POINTS_COUNT * DIM;
    int *points;
    checkCuda(cudaMallocManaged(&points, size * sizeof(int)));
    randomArray(size, points, POINTS_RANGE);

    printf("----CPU SOLUTION----\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    kMeansCPU(points, size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    print_timediff("CPU time: ", start, end);

    printf("----CUDA NOT OPTIMAL SOLUTION----\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    kMeansCUDA(points);
    clock_gettime(CLOCK_MONOTONIC, &end);
    print_timediff("CUDA time: ", start, end);

#if PRINT
    printPoints(size, points);
#endif

    checkCuda(cudaFree(points));
}

