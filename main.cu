#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define DIM 3
#define CENTROID_COUNT 2
#define POINTS_COUNT 200
#define POINTS_RANGE 50
#define ITERS 10

#define checkCuda(ans) { checkCudaError((ans), __LINE__); }

void checkCudaError(cudaError_t cudaStatus, int line) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Line %d CUDA Error %d: %s\n", line, cudaStatus, cudaGetErrorString(cudaStatus));
    }
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
            for (int cDim = 0; cDim < DIM; cDim++){
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

    for (int i = 0; i < size / DIM; i++) {
        std::cout << pointToCentroid[i] << ",";
    }
    std::cout << std::endl;

}

void randomArray(int size, int *array, int range = POINTS_RANGE) {
    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution<> d{POINTS_RANGE / 2, 2};
    std::normal_distribution<> d2{POINTS_RANGE / 5, 2};


    for (int i = 0; i < size; i++) {
        if (i < size / 2)
            array[i] = (int) d(gen) % range;
        else
            array[i] = (int) d2(gen) % range;

    }
}

void printPoints(int size, const int *points) {
    for (size_t i = 0; i < size; i++) {
        std::cout << points[i] << ", ";
    }
    printf("\n");
}

//////////////////////////////////////// CUDA sol 1 ////////////////////////////////////////

__global__ void distance(const int *points, double *dists, const double *centroids) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    //bounds check
    if (idx >= POINTS_COUNT) return;

    for (int currCentroid = 0; currCentroid < CENTROID_COUNT; currCentroid++) {
        double sum = 0;
        for (int currDim = 0; currDim < DIM; currDim++) {
            sum += std::pow((double) points[idx * DIM + currDim] - centroids[currCentroid * DIM + currDim], 2);
        }
        dists[idx*CENTROID_COUNT + currCentroid] = std::sqrt(sum);
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

//__global__ void newCentroids(const int *points, const int *p2c, double *centroids, int *clusterSizes) {
//
//
//    //get idx of thread at grid level
//    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//    //bounds check
//    if (idx >= POINTS_COUNT) return;
//
//    //get idx of thread at the block level
//    const int s_idx = threadIdx.x;
//
//    //put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
//    __shared__ double s_datapoints[32];
//    s_datapoints[s_idx] = points[idx];
//
//    __shared__ int s_clust_assn[32];
//    s_clust_assn[s_idx] = p2c[idx];
//
//    __syncthreads();
//
//    //it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
//    if (s_idx == 0) {
//        double b_clust_datapoint_sums[CENTROID_COUNT] = {0};
//        int b_clust_sizes[CENTROID_COUNT] = {0};
//
//        for (int j = 0; j < blockDim.x; ++j) {
//            int clust_id = s_clust_assn[j];
//            b_clust_datapoint_sums[clust_id] += s_datapoints[j];
//            b_clust_sizes[clust_id] += 1;
//        }
//
//        //Now we add the sums to the global centroids and add the counts to the global counts.
//        for (int z = 0; z < CENTROID_COUNT; ++z) {
//            atomicAdd(&centroids[z], b_clust_datapoint_sums[z]);
//            atomicAdd(&clusterSizes[z], b_clust_sizes[z]);
//        }
//    }
//
//    __syncthreads();
//
//    //currently centroids are just sums, so divide by size to get actual centroids
//    if (idx < CENTROID_COUNT) {
//        centroids[idx] = centroids[idx] / clusterSizes[idx];
//    }
//}
__global__ void newCentroids(const int *points, const int *p2c, double *centroids) {
    const uint centroidIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (centroidIdx >= CENTROID_COUNT) return;
    int sum[DIM] = {0};
    int count = 0;
    for (int i = 0; i < POINTS_COUNT; i++) {
        if (p2c[i] == centroidIdx) {
            for (int curDim = 0; curDim < DIM; curDim++) {
                sum[curDim] += points[i*DIM + curDim];
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
#if true

        std::cout <<"CENTROIDS:"<< std::endl;
        for (int i = 0; i < CENTROID_COUNT * DIM; i++) {
            std::cout << centroids[i] << ",";
        }
        std::cout << std::endl;
#endif
        cudaMemset(dists, 0.0, CENTROID_COUNT * POINTS_COUNT * sizeof(double));
        distance<<<grid, block>>>(points, dists, centroids);
        checkCuda(cudaDeviceSynchronize());
        std::cout <<"DISTS:"<< std::endl;
        for (int i = 0; i < CENTROID_COUNT * POINTS_COUNT; i++) {
            std::cout << dists[i] << ",";
        }
        std::cout << std::endl;
        cudaMemset(dists, 0,  POINTS_COUNT * sizeof(int));
        assign<<<grid, block>>>(dists, pointToCentroid);
        checkCuda(cudaDeviceSynchronize());
        std::cout <<"POINT-TO-CENTROID:"<< std::endl;
        for (int i = 0; i < POINTS_COUNT; i++) {
            std::cout << pointToCentroid[i] << ",";
        }
        std::cout << std::endl;

        //reset before next step
        cudaMemset(centroids, 0.0, CENTROID_COUNT * DIM * sizeof(double));
        checkCuda(cudaDeviceSynchronize());
        newCentroids<<<1, CENTROID_COUNT>>>(points, pointToCentroid, centroids);
        checkCuda(cudaDeviceSynchronize());
        std::cout <<"NEW CENTROIDS:"<< std::endl;
        for (int i = 0; i < CENTROID_COUNT * DIM; i++) {
            std::cout << centroids[i] << ",";
        }
        std::cout << std::endl;

        checkCuda(cudaDeviceSynchronize());
        iter++;
    }

#if true
    checkCuda(cudaDeviceSynchronize());

//    for (int i = 0; i < CENTROID_COUNT * POINTS_COUNT; i++) {
//        std::cout << dists[i] << ",";
//    }
//    std::cout << std::endl;

#endif //DEBUG
    // cleanup
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(centroids));
    checkCuda(cudaFree(dists));
}


int main() {
    int size = POINTS_COUNT * DIM;
    int *points;
    checkCuda(cudaMallocManaged(&points, size * sizeof(int)));
    randomArray(size, points, POINTS_RANGE);

    printf("----CPU SOLUTION----\n");
    kMeansCPU(points, size);
    printf("----CUDA SOLUTION----\n");
    kMeansCUDA(points);

    printPoints(size, points);

    checkCuda(cudaFree(points));
}

