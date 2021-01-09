#include <iostream>
#include <vector>
#include <cmath>

#define DIM 3
#define CENTROID_COUNT 8
#define POINTS_COUNT 20
#define ITERS 10

#define checkCuda(ans) { checkCudaError((ans), __LINE__); }

void checkCudaError(cudaError_t cudaStatus, int line) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Line %d CUDA Error %d: %s\n", line, cudaStatus, cudaGetErrorString(cudaStatus));
    }
}

//////////////////////////////////////// CPU ////////////////////////////////////////

double distanceBetweenTwoPoints(int *points, float *centroids, int point, int centroid) {

    //printf("Distance between [%d %d %d] and [%d %d %d] ",
    // points[point * DIM ],points[point * DIM +1 ],points[point * DIM +2 ],
    // centroids[centroid * DIM ],centroids[centroid * DIM +1 ],centroids[centroid * DIM +2 ]
    // );

    int sum = 0;
    for (int i = 0; i < DIM; i++) {
        sum += std::pow((float)points[point * DIM + i] - centroids[centroid * DIM + i], 2);
    }
    //printf(" = %f\n", sqrt(sum));
    return std::sqrt(sum);
}

void randomCentroids(const int *points,float *centroids, int size) {
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
    randomCentroids(points,centroids, size);
    int pointToCentroid[size / DIM];
    int iters = 0;

    while (iters < ITERS) {
        // step 1: assign each point to the closest centroid
        for (int i = 0; i < size / DIM; i++) {
            double minDist = INT32_MAX;
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

        for (int i = 0; i < size / DIM; i++) { //point
//            for(int i2=0;i2<DIM;i2++){ //dimension
            int c = pointToCentroid[i];
            countsPerCluster[c] += 1;
            for (int i3 = 0; i3 < DIM; i3++) //dimension of  considered point
            {
                sumPerCluster[c * DIM + i3] += points[i * DIM + i3];
            }
//            }
        }
        // recompute
        for (int i = 0; i < CENTROID_COUNT; i++) {
            for (int j = 0; j < DIM; j++) {
                centroids[i * DIM + j] = (float)sumPerCluster[i * DIM + j] / (float)countsPerCluster[i];
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

void randomArray(int size,int* array, int range = 1000) {
    for (int i = 0; i < size; i++) {
        array[i] = random() % range;
    }
}

void printPoints(int size, const int *points) {
    for (size_t i = 0; i < size; i++) {
        std::cout << points[i] << ", ";
    }
    printf("\n");
}

//////////////////////////////////////// CUDA ////////////////////////////////////////
__global__ void distance(const int *points, float *dists, const float* centroids) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    //bounds check
    if (idx >= POINTS_COUNT) return;

    for (int currCentroid = 0; currCentroid < CENTROID_COUNT; currCentroid++){
        double sum = 0;
        for(int currDim = 0; currDim < DIM; currDim++){
            sum += std::pow((float)points[idx * DIM + currDim] - centroids[currCentroid * DIM + currDim], 2);
        }
        dists[idx*CENTROID_COUNT + currCentroid] = sum;
    }
}

void kMeansCUDA(int *points) {
    dim3 block(32, 4);
    dim3 grid(block.x * block.y, ceil((double) POINTS_COUNT / (block.x * block.y)));

    float *dists;
    float *centroids;
    checkCuda(cudaMallocManaged(&dists, CENTROID_COUNT * POINTS_COUNT * sizeof(float)));
    checkCuda(cudaMallocManaged(&centroids,CENTROID_COUNT * DIM * sizeof(float)));
    randomCentroids(points,centroids,POINTS_COUNT * DIM);
    distance<<<grid, block>>>(points, dists, centroids);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaFree(centroids));
    checkCuda(cudaFree(dists));
}


int main() {
    int size = POINTS_COUNT * DIM;
    int *points;
    checkCuda(cudaMallocManaged(&points,size * sizeof(int)));
    randomArray(size,points, 30);

    kMeansCPU(points, size);

    kMeansCUDA(points);

    printPoints(size, points);

    checkCuda(cudaFree(points));
}

