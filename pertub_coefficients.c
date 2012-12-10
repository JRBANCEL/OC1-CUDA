// $c: number of categories
// $d: number of features
// $n: number of samples
// $s: number of splits

// Compute the U_j
__global__ void perturb1(double samples[$n][$d],
                         double hyperplan[$d+2],
                         double U[$n],
                         unsigned int m) {

    int tid;        // Thread id
    int i;          // Index variable
    double V;

    tid = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the samples

    if(tid < $n) {
        V = hyperplan[$d];
        for(i = 0 ; i < $d ; i++) {
            V += hyperplan[i] * samples[tid][i];
            U[tid] = (hyperplan[m] * samples[tid][m] - V)/samples[tid][m];
        }
    }
}

// Compute the univariate splits of U_j
__global__ void perturb2(double U[$n],
                         double splits[$n][2]) {
    int tid;
    int i;
    double next = 1000000; //XXX Fix that

    tid = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the samples

    if(tid < $n) {
        for(i = 0 ; i < $n ; i++) {
            if(U[i] > U[tid] && U[i] < next)
                next = U[i];
        }
        if(next != 1000000) {
            splits[tid][1] = 1;
            splits[tid][0] = (U[tid] + next)/2.0;
        }
    }
}
