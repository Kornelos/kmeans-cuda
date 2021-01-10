#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define DIM 3
#define CENTROID_COUNT 2
#define POINTS_COUNT 5000
#define POINTS_RANGE 50
#define ITERS 2
#define NORMAL_DIST false

#define DEBUG false
#define PRINT true

///// UTILS
#define checkCuda(ans) { checkCudaError((ans), __LINE__); }

void checkCudaError(cudaError_t cudaStatus, int line) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Line %d CUDA Error %d: %s\n", line, cudaStatus, cudaGetErrorString(cudaStatus));
    }
}

void print_timediff(const char *prefix, const struct timespec &start, const
struct timespec &end) {
    float milliseconds = end.tv_nsec >= start.tv_nsec
                          ? (end.tv_nsec - start.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec) * 1e3
                          : (start.tv_nsec - end.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec - 1) * 1e3;
    printf("%s: %lf milliseconds\n", prefix, milliseconds);
}

//////////////////////////////////////// CPU ////////////////////////////////////////

float distanceBetweenTwoPoints(int *points, float *centroids, int point, int centroid) {

    //printf("Distance between [%d %d %d] and [%d %d %d] ",
    // points[point * DIM ],points[point * DIM +1 ],points[point * DIM +2 ],
    // centroids[centroid * DIM ],centroids[centroid * DIM +1 ],centroids[centroid * DIM +2 ]
    // );

    int sum = 0;
    for (int i = 0; i < DIM; i++) {
        sum += std::pow((float) points[point * DIM + i] - centroids[centroid * DIM + i], 2);
    }
    //printf(" = %f\n", sqrt(sum));
    return std::sqrt(sum);
}

void randomCentroids(const int *points, float *centroids, int size) {
    std::vector<float> copy(size);
    for (int i = 0; i < size; i++) {
        copy.at(i) = points[i];
    }

    for (int i = 0; i < CENTROID_COUNT; i++) {
        int index = INT32_MAX;
        while (index + DIM - 1 > copy.size()) {
            index = (random() % copy.size()) * DIM;
        }
        std::vector<float>::iterator it1, it2;
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
    float centroids[DIM * CENTROID_COUNT];
    randomCentroids(points, centroids, size);
    int pointToCentroid[size / DIM];
    int iters = 0;

    while (iters < ITERS) {
        // step 1: assign each point to the closest centroid
        for (int i = 0; i < size / DIM; i++) {
            float minDist = MAXFLOAT;
            int currentCentroid;
            for (int j = 0; j < CENTROID_COUNT; j++) {
                float dist = distanceBetweenTwoPoints(points, centroids, i, j);
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
                centroids[i * DIM + j] = (float) sumPerCluster[i * DIM + j] / (float) countsPerCluster[i];
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

__global__ void distance(const int *points, float *dists, const float *centroids) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    //bounds check
    if (idx >= POINTS_COUNT) return;

    for (int currCentroid = 0; currCentroid < CENTROID_COUNT; currCentroid++) {
        float sum = 0;
        for (int currDim = 0; currDim < DIM; currDim++) {
            sum += std::pow((float) points[idx * DIM + currDim] - centroids[currCentroid * DIM + currDim], 2);
        }
        dists[idx * CENTROID_COUNT + currCentroid] = std::sqrt(sum);
    }
}

__global__ void assign(const float *dists, int *p2c) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= POINTS_COUNT) return;
    float minDist = MAXFLOAT;
    int minCentroid = -1;
    for (int dist = 0; dist < CENTROID_COUNT; dist++) {
        if (dists[idx * CENTROID_COUNT + dist] < minDist) {
            minDist = dists[idx * CENTROID_COUNT + dist];
            minCentroid = dist;
        }
    }
    p2c[idx] = minCentroid;
}

__global__ void newCentroids(const int *points, const int *p2c, float *centroids) {
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
        centroids[centroidIdx * DIM + curDim] = (float) sum[curDim] / (float) count;
    }
}

void kMeansCUDA(int *points) {
    dim3 block(32, 4);
    dim3 grid(block.x * block.y, ceil((float) POINTS_COUNT / (block.x * block.y)));

    float *dists;
    float *centroids;
    int *pointToCentroid;
    int *clusterSizes;
    checkCuda(cudaMallocManaged(&dists, CENTROID_COUNT * POINTS_COUNT * sizeof(float)));
    checkCuda(cudaMallocManaged(&centroids, CENTROID_COUNT * DIM * sizeof(float)));
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
        cudaMemset(dists, 0.0, CENTROID_COUNT * POINTS_COUNT * sizeof(float));
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
        cudaMemset(centroids, 0.0, CENTROID_COUNT * DIM * sizeof(float));
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

__device__ float distance_squared(const int *points, const float *centroid) {
    float sum = 0;
    for (int i = 0; i < DIM; i++) {
        sum +=(points[i] - centroid[i])*(points[i] - centroid[i]);
    }
    return sum;
}

__global__ void move_centroids(float *d_centroids, float *new_centroids, float *d_counters) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int local_tid = threadIdx.x;
    extern __shared__ float this_centroid[];
    float *this_centroid_counters = (float *) this_centroid + DIM * +blockDim.x;
    for (int cDim = 0; cDim < DIM; cDim++) {
        this_centroid[local_tid*DIM + cDim] = new_centroids[tid*DIM + cDim];
    }
    this_centroid_counters[local_tid] = d_counters[tid];
    __syncthreads();

    //TODO reduce on values -> works only when number of blocks is some power of 2
    for (int d = blockDim.x / 2; d > 0; d >>= 1) {
        if (local_tid < d) {
            for (int cDim = 0; cDim < DIM; cDim++) {
                this_centroid[local_tid*DIM + cDim] = new_centroids[(local_tid + d)*DIM + cDim];
            }
            this_centroid_counters[local_tid] += this_centroid_counters[local_tid + d];
        }
        __syncthreads();
    }

    //assignment of new values
    if (local_tid == 0) {
        const float count = this_centroid_counters[local_tid];
        for (int cDim = 0; cDim < DIM; cDim++) {
            d_centroids[local_tid*DIM + cDim] = this_centroid[local_tid*DIM + cDim] / count;
        }
    }
    __syncthreads();

    for (int cDim = 0; cDim < DIM; cDim++) {
        new_centroids[tid*DIM + cDim] = 0;
    }
    d_counters[tid] = 0;
}

__global__ void distances_calculation(const int *points, const float *centroids, float * new_centroids,float* counters) {
    //this version works on atomics with 7x speedup
    extern __shared__ float local_centroids[];
    float *s_array = (float *) local_centroids + DIM * CENTROID_COUNT;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    bool has_element = tid < POINTS_COUNT;
    if (tid >= POINTS_COUNT) return;
    int currentCentroid = 0;

    float currentDistance;
    int point[] = {0};

    if (has_element) {
        currentDistance = INFINITY;
        for (int cDim = 0; cDim < DIM; cDim++) {
            point[cDim] = points[tid * DIM + cDim];
        }
    }

    if (local_tid < CENTROID_COUNT) {
        for (int cDim = 0; cDim < DIM; cDim++) {
            local_centroids[local_tid * CENTROID_COUNT + cDim] = centroids[local_tid * CENTROID_COUNT + cDim];
        }
    }

    __syncthreads();

    if (has_element) {
        for (int i = 0; i < CENTROID_COUNT; ++i) {
            // take considered centroid
            float centroid[] = {0};
            for (int cDim = 0; cDim < DIM; cDim++) {
                centroid[cDim] = local_centroids[i * DIM + cDim];
            }
            const float _distance = distance_squared(point, centroid);
            if (_distance < currentDistance) {
                currentCentroid = i;
                currentDistance = _distance;
            }
        }
    }

    __syncthreads();

    int offset = blockDim.x;
    int n[DIM + 1] = {};
    for (int i = 0; i < DIM + 1; i++) {
        n[i] = local_tid + i * offset; // dims and counter
    }

    for (int i = 0; i < CENTROID_COUNT; ++i) {
        for (int cDim = 0; cDim < DIM; cDim++) {
            s_array[n[cDim]] = (has_element && (currentCentroid == i)) ? point[cDim] : 0; // dims
        }
        s_array[DIM + 1] = (has_element && (currentCentroid == i)) ? 1 : 0;

        __syncthreads();

        for (int d = blockDim.x / 2; d > 0; d >>= 1) {
            if (local_tid < d) {
                for(int cDim = 0; cDim < DIM; cDim++){
                    s_array[n[cDim]] +=s_array[n[cDim] + d];
                }
            }
            __syncthreads();
        }

        if (local_tid == 0) {
            for(int cDim = 0; cDim < DIM; cDim++){
                new_centroids[i * gridDim.x  + blockIdx.x * DIM] +=s_array[n[cDim]];
            }
            counters[i * gridDim.x + blockIdx.x] = s_array[DIM+1];

        }
        __syncthreads();
    }

}

void optimalKMeansCUDA(int *points) {
    float* centroids; float* new_centroids;
    float* counters;
    checkCuda(cudaMallocManaged(&centroids, CENTROID_COUNT * DIM * sizeof(float)));
    checkCuda(cudaMallocManaged(&new_centroids, CENTROID_COUNT * DIM * sizeof(float)));
    checkCuda(cudaMallocManaged(&counters, CENTROID_COUNT * sizeof(float)));

    randomCentroids(points, centroids, POINTS_COUNT * DIM);

    int num_threads = 1024;
    int num_blocks = (POINTS_COUNT + num_threads - 1) / num_threads;
    //we will be accessing memory structures concurrently -> AoS makes more sense than SoA

    int mem = DIM * CENTROID_COUNT * sizeof(float) + (DIM+1) * num_threads * sizeof(float);
    int mem2 = (DIM+1) * num_blocks * sizeof(float);
    for (int i = 0; i < ITERS; ++i) {
        distances_calculation<<<num_blocks, num_threads, mem>>>(points,centroids,new_centroids,counters);
        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());

        move_centroids<<<CENTROID_COUNT, num_blocks, mem2>>>(centroids,new_centroids,counters);
        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());
    }

#if PRINT
    checkCuda(cudaDeviceSynchronize());
    std::cout <<"POINT-TO-CENTROID:"<< std::endl;
    for (int i = 0; i < POINTS_COUNT; i++) {
        std::cout << centroids[i] << ",";
    }
    std::cout << std::endl;
#endif

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(centroids));
    checkCuda(cudaFree(new_centroids));
    checkCuda(cudaFree(counters));
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
    print_timediff("CPU time ", start, end);

    printf("----CUDA NOT OPTIMAL SOLUTION----\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    kMeansCUDA(points);
    clock_gettime(CLOCK_MONOTONIC, &end);
    print_timediff("CUDA time ", start, end);

    printf("----CUDA BETTER SOLUTION----\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    optimalKMeansCUDA(points);
    clock_gettime(CLOCK_MONOTONIC, &end);
    print_timediff("CUDA time ", start, end);

#if PRINT
    printPoints(size, points);
#endif

    checkCuda(cudaFree(points));
}

