#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define DIM 1
#define CENTROID_COUNT 2
#define POINTS_COUNT 8
#define POINTS_RANGE 50
#define ITERS 18
#define NORMAL_DIST true

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
    std::cout << "POINT-TO-CENTROID:" << std::endl;
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
        sum += (points[i] - centroid[i]) * (points[i] - centroid[i]);
    }
    return sum;
}

__global__ void distances_calculation(const int *points, int *d_clust_assn, const float *d_centroids) {
    //get idx for this datapoint
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //bounds check
    if (idx >= POINTS_COUNT) return;

    //find the closest centroid to this datapoint
    float min_dist = INFINITY;
    int closest_centroid = 0;

    int point[DIM] = {};
    for(int cDim = 0; cDim < DIM; cDim++){
        point[cDim] = points[idx*DIM + cDim];
    }

    for (int c = 0; c < CENTROID_COUNT; ++c) {
        float centroid[DIM] = {};
        for(int cDim = 0; cDim < DIM; cDim++){
            centroid[cDim] = d_centroids[c*DIM + cDim];
        }

        float dist = distance_squared(point, centroid);

        if (dist < min_dist) {
            min_dist = dist;
            closest_centroid = c;
        }
    }

    //assign closest cluster id for this datapoint/thread
    d_clust_assn[idx] = closest_centroid;
}

#define TPB 32

__global__ void move_centroids(const int *d_datapoints, const int *d_clust_assn, float *d_centroids, int *d_clust_sizes) {
    //get idx of thread at grid level
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //bounds check
    if (idx >= POINTS_COUNT) return;

    //get idx of thread at the block level
    const int s_idx = threadIdx.x;

    //put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
    __shared__ float s_datapoints[TPB];
    s_datapoints[s_idx] = d_datapoints[idx];

    __shared__ int s_clust_assn[TPB];
    s_clust_assn[s_idx] = d_clust_assn[idx];

    __syncthreads();

    //it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
    if (s_idx == 0) {
        float b_clust_datapoint_sums[CENTROID_COUNT] = {0};
        int b_clust_sizes[CENTROID_COUNT] = {0};

        for (int j = 0; j < blockDim.x; ++j) {
            int clust_id = s_clust_assn[j];
            b_clust_datapoint_sums[clust_id] += s_datapoints[j];
            b_clust_sizes[clust_id] += 1;
        }

        //Now we add the sums to the global centroids and add the counts to the global counts.
        for (int z = 0; z < CENTROID_COUNT; ++z) {
            atomicAdd(&d_centroids[z], b_clust_datapoint_sums[z]);
            atomicAdd(&d_clust_sizes[z], b_clust_sizes[z]);
        }
    }

    __syncthreads();

    //currently centroids are just sums, so divide by size to get actual centroids
    if (idx < CENTROID_COUNT) {
        d_centroids[idx] = d_centroids[idx] / d_clust_sizes[idx];
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

    int num_threads = 1024;
    int num_blocks = (POINTS_COUNT + num_threads - 1) / num_threads;
    //we will be accessing memory structures concurrently -> AoS makes more sense than SoA

    int mem = DIM * CENTROID_COUNT * sizeof(float) + (DIM + 1) * num_threads * sizeof(float);
    int mem2 = (DIM + 1) * num_blocks * sizeof(float);
    for (int i = 0; i < ITERS; ++i) {
        //__global__ void distances_calculation(const int *d_datapoints, int *d_clust_assn, const float *d_centroids) {
        distances_calculation<<<(POINTS_COUNT + TPB - 1) / TPB, TPB>>>(points, pointToCentroid, centroids);
        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());
        //__global__ void move_centroids(int *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clust_sizes) {
        cudaMemset(centroids, 0.0, CENTROID_COUNT * sizeof(float));
        cudaMemset(counters, 0, CENTROID_COUNT * sizeof(int));
        move_centroids<<<(POINTS_COUNT + TPB - 1) / TPB, TPB>>>(points, pointToCentroid, centroids, counters);
        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaDeviceSynchronize());
    }

#if PRINT
    checkCuda(cudaDeviceSynchronize());
    std::cout << "POINT-TO-CENTROID:" << std::endl;
    for (int i = 0; i < POINTS_COUNT; i++) {
        std::cout << pointToCentroid[i] << ",";
    }
    std::cout << std::endl;
#endif

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

