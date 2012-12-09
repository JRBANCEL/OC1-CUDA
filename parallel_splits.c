// $d: number of features
// $n: number of samples

##include <limits.h>

// This is the exhaustive greedy method
__global__ void parallel_splits(double samples[$n][$d],
                                double hyperplans[$d*$n][$d+2]) {

    int t_x, t_y;   // Thread id
    int i;          // Index variable
    double next = 1000000; //XXX Fix that

    t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the samples
    t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the features

    if(t_x < $n && t_y < $d) {
        // Find the next sample
        for(i = 0 ; i < $n ; i++)
            if(samples[i][t_y] > samples[t_x][t_y] && samples[i][t_y] < next)
                next = samples[i][t_y];
        // If a next point was found: the current point is not the last one
        if(next != 1000000) {
            hyperplans[t_x+t_y*$n][$d+1] = 1;
            hyperplans[t_x+t_y*$n][t_y] = 1;
            hyperplans[t_x+t_y*$n][$d] = -(samples[t_x][t_y] + next)/2.0;
        }
    }
}
