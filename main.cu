#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define TPB 32 // tuning this parameter can improve CUDA perf

//control params
#define DIM 3
#define CENTROID_COUNT 8
#define POINTS_COUNT TPB * 40
#define POINTS_RANGE 256
#define ITERS 3
#define NORMAL_DIST false
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

    int sum = 0;
    for (int i = 0; i < DIM; i++) {
        sum += std::pow((float) points[point * DIM + i] - centroids[centroid * DIM + i], 2);
    }
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
    std::cout <<"POINT-TO-CENTROID:"<< std::endl;
    for (int i = 0; i < size / DIM; i++) {
        std::cout << pointToCentroid[i] << ",";
    }
    std::cout << std::endl;

    std::cout <<"CENTROIDS:"<< std::endl;
    for (int i = 0; i < CENTROID_COUNT * DIM; i++) {
        std::cout << centroids[i] << ",";
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
    dim3 grid(block.x * block.y, (int)ceil((float) POINTS_COUNT / (float)(block.x * block.y)));

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
        //for each point calculate distance
        cudaMemset(dists, 0.0, CENTROID_COUNT * POINTS_COUNT * sizeof(float));
        distance<<<grid, block>>>(points, dists, centroids);
        checkCuda(cudaDeviceSynchronize());
        //assign centroid to each point
        cudaMemset(dists, 0, POINTS_COUNT * sizeof(int));
        assign<<<grid, block>>>(dists, pointToCentroid);
        checkCuda(cudaDeviceSynchronize());
        //recalculate each centroid
        cudaMemset(centroids, 0.0, CENTROID_COUNT * DIM * sizeof(float));
        newCentroids<<<1, CENTROID_COUNT>>>(points, pointToCentroid, centroids);
        checkCuda(cudaDeviceSynchronize());
        iter++;
    }
#if PRINT
    checkCuda(cudaDeviceSynchronize());
    std::cout << "POINT-TO-CENTROID:" << std::endl;
    for (int i = 0; i < POINTS_COUNT; i++) {
        std::cout << pointToCentroid[i] << ",";
    }
    std::cout << std::endl;

    std::cout <<"CENTROIDS:"<< std::endl;
    for (int i = 0; i < CENTROID_COUNT * DIM; i++) {
        std::cout << centroids[i] << ",";
    }
    std::cout << std::endl;
#endif

    // cleanup
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(centroids));
    checkCuda(cudaFree(dists));
    checkCuda(cudaFree(pointToCentroid));
    checkCuda(cudaFree(clusterSizes));
}
//////////////////////////////////////// CUDA sol 2 ////////////////////////////////////////

__device__ float distance_squared(const int *points, const float *centroid) {
    float sum = 0;
    for (int i = 0; i < DIM; i++) {
        sum += (points[i] - centroid[i]) * (points[i] - centroid[i]);
    }
    return sum;
}

__global__ void distances_calculation(const int *points, int *p2c, const float *centroids) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= POINTS_COUNT) { return; }

    //find the closest centroid to this datapoint
    float min_dist = INFINITY;
    int closest_centroid = 0;

    int point[DIM] = {};
    for (int cDim = 0; cDim < DIM; cDim++) {
        point[cDim] = points[idx * DIM + cDim];
    }

    for (int c = 0; c < CENTROID_COUNT; ++c) {
        float centroid[DIM] = {};
        for (int cDim = 0; cDim < DIM; cDim++) {
            centroid[cDim] = centroids[c * DIM + cDim];
        }

        float dist = distance_squared(point, centroid);

        if (dist < min_dist) {
            min_dist = dist;
            closest_centroid = c;
        }
    }

    //assign closest cluster id for this datapoint/thread
    p2c[idx] = closest_centroid;
}



__global__ void move_centroids(const int *points, const int *p2c, float *centroids, int *counters) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= POINTS_COUNT) { return; }

    //get idx of thread at the block level
    const uint s_idx = threadIdx.x;

    //put the points and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
    __shared__ int sPoints[TPB * DIM];
    for (int cDim = 0; cDim < DIM; cDim++) {
        sPoints[s_idx * DIM + cDim] = points[idx * DIM + cDim];
    }

    __shared__ int sp2c[TPB];
    sp2c[s_idx] = p2c[idx];

    __syncthreads();

    //it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
    if (s_idx == 0) {
        float blockSums[CENTROID_COUNT * DIM] = {0};
        int blockCounts[CENTROID_COUNT] = {0};

        for (int j = 0; j < blockDim.x; ++j) {
            int cid = sp2c[j];
            for (int cDim = 0; cDim < DIM; cDim++) {
                blockSums[cid * DIM + cDim] += (float) sPoints[j * DIM + cDim];
            }
            blockCounts[cid] += 1;
        }

        //Now we add the sums to the global centroids and add the counts to the global counts.
        for (int z = 0; z < CENTROID_COUNT; ++z) {
            for (int cDim = 0; cDim < DIM; cDim++) {
                atomicAdd(&centroids[z * DIM + cDim], blockSums[z * DIM + cDim]);
            }
            atomicAdd(&counters[z], blockCounts[z]);
        }
    }

    __syncthreads();

    //get centroids
    if (idx < CENTROID_COUNT) {
        for (int cDim = 0; cDim < DIM; cDim++) {
            centroids[idx * DIM + cDim] = centroids[idx * DIM + cDim] / (float) counters[idx];

        }
    }
}

void optimalKMeansCUDA(int *points) {
    float *centroids;
    float *new_centroids;
    int *counters;
    int *pointToCentroid;
    checkCuda(cudaMallocManaged(&centroids, CENTROID_COUNT * DIM * sizeof(float)));
    checkCuda(cudaMallocManaged(&new_centroids, CENTROID_COUNT * DIM * sizeof(float)));
    checkCuda(cudaMallocManaged(&counters, CENTROID_COUNT * sizeof(int)));
    checkCuda(cudaMallocManaged(&pointToCentroid, POINTS_COUNT * sizeof(int)));

    randomCentroids(points, centroids, POINTS_COUNT * DIM);

    for (int i = 0; i < ITERS; ++i) {
        distances_calculation<<<(POINTS_COUNT + TPB - 1) / TPB, TPB>>>(points, pointToCentroid, centroids);
        checkCuda(cudaDeviceSynchronize());
        cudaMemset(centroids, 0.0, CENTROID_COUNT * sizeof(float));
        cudaMemset(counters, 0, CENTROID_COUNT * sizeof(int));
        move_centroids<<<(POINTS_COUNT + TPB - 1) / TPB, TPB>>>(points, pointToCentroid, centroids, counters);
        checkCuda(cudaDeviceSynchronize());
    }

#if PRINT
    checkCuda(cudaDeviceSynchronize());
    std::cout << "POINT-TO-CENTROID:" << std::endl;
    for (int i = 0; i < POINTS_COUNT; i++) {
        std::cout << pointToCentroid[i] << ",";
    }
    std::cout << std::endl;

    std::cout <<"CENTROIDS:"<< std::endl;
    for (int i = 0; i < CENTROID_COUNT * DIM; i++) {
        std::cout << centroids[i] << ",";
    }
    std::cout << std::endl;
#endif
    //cleanup
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(centroids));
    checkCuda(cudaFree(new_centroids));
    checkCuda(cudaFree(counters));
    checkCuda(cudaFree(pointToCentroid));

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

