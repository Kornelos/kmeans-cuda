#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <fstream>

#define DIM 3
#define N 8
#define POINTS_COUNT 200
double distanceBetweenTwoPoints(int* points, float* centroids, int point, int centroid){

        //printf("Distance between [%d %d %d] and [%d %d %d] ",
        // points[point * DIM ],points[point * DIM +1 ],points[point * DIM +2 ],
        // centroids[centroid * DIM ],centroids[centroid * DIM +1 ],centroids[centroid * DIM +2 ]
        // );

        int sum = 0;
        for(int i = 0; i < DIM; i++){
            sum += pow(points[point * DIM + i] - centroids[centroid * DIM + i],2);
        }
        //printf(" = %f\n", sqrt(sum));
        return sqrt(sum);
}

float* randomCentroids(const int *points, int size){
    std::vector<float> copy(size);
    for (int i = 0; i < size; i++){
        copy.at(i) = points[i];
    }

    static float centroids[DIM*N];

    for(int i = 0; i<N;i++){
        int index = INT32_MAX;
        while(index + DIM-1 > copy.size()){
            index = (random() % copy.size()) * DIM;
        }
        std::vector<float>::iterator it1, it2;
        it1 = ( copy.begin() + index );
        it2 = ( copy.begin() + index + DIM );
        for (int j = 0; j < DIM; j++){
            centroids[i*DIM+j] = copy.at(index+j);
        }
        copy.erase(it1,it2);
    }
    return centroids;
}

void kmeans(int* points, int size){

    // step 0: choose n random points
    float* centroids = randomCentroids(points,size);
    int pointToCentroid[size/DIM];
    int iters = 0;

    while(iters < 10){
    // step 1: assign each point to the closest centroid
    for (int i = 0; i < size/DIM; i++){
        double minDist = INT32_MAX;
        int currentCentroid;
        for(int j = 0; j < N; j++){
            double dist = distanceBetweenTwoPoints(points,centroids,i,j);
            if(minDist > dist){
                minDist = dist;
                currentCentroid = j;
            }
        }
         pointToCentroid[i] = currentCentroid;
    }


    // step 2: recompute centroids
    int countsPerCluster[N];
    int sumPerCluster[N*DIM];

    for(int i = 0; i<size/DIM;i++){
        for(int j=0;j<DIM;j++){
            int c = pointToCentroid[i];
            countsPerCluster[c] += 1;
            for (int j = 0; j < DIM; j++)
            {
                sumPerCluster[c*DIM+j] += points[i*DIM + j];
            }
        }
    }
    // recompute
    for (int i = 0; i < N; i++){
        for (int j = 0; j < DIM; j++){
           centroids[i*DIM + j] = sumPerCluster[i*DIM + j] / countsPerCluster[i];
        }
    }
    
    // repeat step 1 and 2 until convergence (no point changed its cluster)
    iters++;
    }

    for (int i = 0; i < size/DIM; i++) {
        std::cout << pointToCentroid[i] << ",";
    }
    std::cout<<std::endl;

}

int* randomArray(int size, int range=1000){
    static int* array = new int[size];
    for (int i = 0; i < size; i++){
        array[i] = random() % 10;
    }
    return array;
}

int main(){

    int test[] = {
        1,2,3,
        4,5,6,
        7,8,9
    };
    int size = POINTS_COUNT * DIM;
    int* points = randomArray(size,30);
    kmeans(points,size);
    // //int *centroids = randomCentroids(test, size);

    // for (int i = 0; i < N; i++){
    //     for(int j = 0; j< DIM; j++){
    //             std::cout  << centroids[i*DIM+j] << " ";
    //     }
    //     std::cout<<std::endl;
    // }  
    for (size_t i = 0; i < size; i++){
        std::cout<<points[i]<<", ";
    }
    printf("\n");

    

    delete[] points;
}

