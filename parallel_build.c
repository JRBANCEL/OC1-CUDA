// $d: number of features
// $n: number of samples
// $t: number of trials (how many hyperplans are tested)
// $s: number of sets to split

// This computes the impurity of every parallel split in different sets at
// the same time. This equivalent of parallels_splits, impurity1, impurity2
// impurity3 in the version 1 on $s sets.
// Branching is the main problem of this kernel. However, branches don't
// diverge so it should not be a huge performance killer.
__global__ void parallel_splits(double samples[$n][$d],
                                unsigned int sieve[$s][$n],
                                unsigned int cat[$n],
                                double hyperplan[$s][$d*$n][$d+1],
                                double impurity[$s][$d*$n]) {

    int t_x, t_y, t_z;   // Thread id
    int i;          // Index variable
    double next = 1000000; //XXX Fix that
    double split;
    double GiniL = 1;
    double GiniR = 1;
    int countL[$c] = {};
    int countR[$c] = {};
    int Tl = 0;
    int Tr = 0;

    t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the samples
    t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the features
    t_z = blockIdx.z * blockDim.z + threadIdx.z; // Indexes the sets


    if(t_x < $n && t_y < $d && t_z < $s) {
        // Find the next sample
        for(i = 0 ; i < $n ; i++)
            if(sieve[t_z][i] > 0 &&
               samples[i][t_y] > samples[t_x][t_y] && samples[i][t_y] < next)
                next = samples[i][t_y];
        // If a next point was found, compute impurity of the hyperplan
        if(next != 1000000) {
            split = (samples[t_x][t_y] + next)/2.0;
            for(i = 0 ; i < $n ; i++) {
                if(sieve[t_z][i] && samples[i][t_y] > split) {
                    countL[cat[i]] += 1;
                    Tl += 1;
                }
                else {
                    countR[cat[i]] += 1;
                    Tr += 1;
                }
            }
            // Compute Gini score
            for(i = 0 ; i < $c ; i++) {
                GiniL -= (countL[i] * countL[i])/(double)(Tl * Tl);
                GiniR -= (countR[i] * countR[i])/(double)(Tr * Tr);
            }
            // Deal with potential division by 0
            if(Tl == 0)
                GiniL = 1;
            if(Tr == 0)
                GiniR = 1;
            impurity[t_z][t_x+t_y*$n] = (Tl * GiniL + Tr * GiniR)/(double)(Tl+Tr);
            hyperplan[t_z][t_x+t_y*$n][$d] = split;
            hyperplan[t_z][t_x+t_y*$n][t_y] = 1;
        }
    }
}

// This kernel reduces an array full of impurity values in order to find the
// minimum and its index in the original array.
__global__ void reduce_impurity(double impurity[$s][$t],
                                unsigned int index[$s][$t],
                                unsigned int size) {

    int t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the items
    int t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the sets
    int next;

    if(size % 2 == 1)
        next = t_x + size/2 + 1;
    else
        next = t_x + size/2;

    if(next < size && t_y < $s) {
        if(impurity[t_y][t_x] > impurity[t_y][next]) {
            impurity[t_y][t_x] = impurity[t_y][next];
            index[t_y][t_x] = index[t_y][next];
        }
    }
}

// Compute the U values for every hyperplan in every set.
__global__ void compute_U(double samples[$n][$d],
                          unsigned int sieve[$s][$n],
                          double hyperplan[$s][$t][$d+1],
                          unsigned int m,
                          double U[$s][$t][$n]) {

    int t_x, t_y, t_z;   // Thread id
    int i;          // Index variable
    double V;

    t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the hyperplans
    t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the sets
    t_z = blockIdx.z * blockDim.z + threadIdx.z; // Indexes the samples

    if(t_x < $t && t_y < $s && t_z < $n) {
        // The computation can be useless but checking if the sample is in
        // the set adds branching.
        V = hyperplan[t_y][t_x][$d];
        for(i = 0 ; i < $d ; i++) {
            V += hyperplan[t_y][t_x][i] * samples[t_z][i];
        }
        U[t_y][t_x][t_z] = (hyperplan[t_y][t_x][m] * samples[t_z][m] - V)/samples[t_z][m];
    }
}

// Find splits of U and compute impurity of it. This is similar to
// parallel_splits kernel but instead of dealing with features, it is
// dealing with U.
__global__ void compute_impurity_U(double samples[$n][$d],
                                   unsigned int sieve[$s][$n],
                                   unsigned int cat[$n],
                                   double splits[$s][$t][$n],
                                   double impurity[$s][$t][$n],
                                   double U[$s][$t][$n]) {

    int t_x, t_y, t_z;   // Thread id
    int i;          // Index variable
    double next = 1000000; //XXX Fix that
    double split;
    double GiniL = 1;
    double GiniR = 1;
    int countL[$c] = {};
    int countR[$c] = {};
    int Tl = 0;
    int Tr = 0;

    t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the samples
    t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the hyperplans
    t_z = blockIdx.z * blockDim.z + threadIdx.z; // Indexes the sets


    if(t_x < $n && t_y < $t && t_z < $s) {
        // Find the next sample
        for(i = 0 ; i < $n ; i++)
            if(sieve[t_z][i] > 0 &&
               U[t_z][t_y][i] > U[t_z][t_y][t_x] && U[t_z][t_y][i] < next)
                next = U[t_z][t_y][i];
        // If a next point was found, compute impurity of the split
        if(next != 1000000) {
            split = (U[t_z][t_y][t_x] + next)/2.0;
            splits[t_z][t_y][t_x] = split;
            for(i = 0 ; i < $n ; i++) {
                if(sieve[t_z][i] && U[t_z][t_y][i] > split) {
                    countL[cat[i]] += 1;
                    Tl += 1;
                }
                else {
                    countR[cat[i]] += 1;
                    Tr += 1;
                }
            }
            // Compute Gini score
            for(i = 0 ; i < $c ; i++) {
                GiniL -= (countL[i] * countL[i])/(double)(Tl * Tl);
                GiniR -= (countR[i] * countR[i])/(double)(Tr * Tr);
            }
            // Deal with potential division by 0
            if(Tl == 0)
                GiniL = 1;
            if(Tr == 0)
                GiniR = 1;
            impurity[t_z][t_y][t_x] = (Tl * GiniL + Tr * GiniR)/(double)(Tl+Tr);
        }
        else
            impurity[t_z][t_y][t_x] = 1;
    }
}

// Similar to reduce_impurity but with an additionnal dimension and the
// reduction is not on the same axis.
__global__ void reduce_impurity_U(double impurity[$s][$t][$n],
                                  unsigned int index[$s][$t][$n],
                                  unsigned int size) {
    int next;
    int t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the items
    int t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the sets
    int t_z = blockIdx.z * blockDim.z + threadIdx.z; // Indexes the hyperplans

    if(size % 2 == 1)
        next = t_x + size/2 + 1;
    else
        next = t_x + size/2;

    if(next < size && t_y < $s && t_z < $t) {
        if(impurity[t_y][t_z][t_x] > impurity[t_y][t_z][next]) {
            impurity[t_y][t_z][t_x] = impurity[t_y][t_z][next];
            index[t_y][t_z][t_x] = index[t_y][t_z][next];
        }
    }
}

// Just a kernel that initialiaze the memory correctly
__global__ void init_index_U(unsigned int index[$s][$t][$n]) {
    int t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the items
    int t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the sets
    int t_z = blockIdx.z * blockDim.z + threadIdx.z; // Indexes the hyperplans

    if(t_x < $n && t_y < $s && t_z < $t)
        index[t_y][t_z][t_x] = t_x;
}

// Just a kernel that initialiaze the memory correctly
__global__ void init_index_hyperplans(unsigned int index[$s][$t]) {
    int t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the sets
    int t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the hyperplans

    if(t_x < $s && t_y < $t)
        index[t_x][t_y] = t_y;
}

// Compute the position of each sample compared to each hyperplan
__global__ void compute_position_hyperplans(double samples[$n][$d],
                                            unsigned int sieve[$s][$n],
                                            double hyperplan[$s][$t][$d+1],
                                            unsigned int position[$s][$t][$n]) {
    int t_x, t_y, t_z;
    int i;          // Index variable
    double point;

    t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the sets
    t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the hyperplans
    t_z = blockIdx.z * blockDim.z + threadIdx.z; // Indexes the samples


    if(t_x < $s && t_y < $t && t_z < $n) {
        if(sieve[t_x][t_z] > 0) {
            point = hyperplan[t_x][t_y][$d];
            for(i = 0 ; i < $d ; i++)
                point += hyperplan[t_x][t_y][i] * samples[t_z][i];

            if(point > 0) {
                position[t_x][t_y][t_z] = 1;
            }
            else {
                position[t_x][t_y][t_z] = 0;
            }
        }
        else {
            position[t_x][t_y][t_z] = 2;
        }
    }
}

// Compute how many samples are above or below an hyperplan per category
__global__ void compute_count_hyperplans(unsigned int cat[$n],
                                         unsigned int position[$s][$t][$n],
                                         unsigned int countL[$s][$t][$c],
                                         unsigned int countR[$s][$t][$c],
                                         unsigned int Tl[$s][$t],
                                         unsigned int Tr[$s][$t]) {

    int t_x, t_y, t_z;
    int i;          // Index variable

    t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the sets
    t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the hyperplans
    t_z = blockIdx.z * blockDim.z + threadIdx.z; // Indexes the categories

    if(t_x < $s && t_y < $t && t_z < $c) {
        countL[t_x][t_y][t_z] = 0;
        countR[t_x][t_y][t_z] = 0;
        for(i = 0 ; i < $n ; i++) {
            if(cat[i] == t_z) {
                if(position[t_x][t_y][i] == 1)
                    countL[t_x][t_y][t_z] += 1;
                else {
                    if(position[t_x][t_y][i] == 0)
                        countR[t_x][t_y][t_z] += 1;
                }
            }
        }
    }
    // Using a separate thread is better than using an atomicAdd in the previous
    // loop because the threads are likely to add a the same moment creating a
    // bottleneck
    if(t_z == $c && t_x < $s && t_y < $t) {
        Tl[t_x][t_y] = 0;
        Tr[t_x][t_y] = 0;
        for(i = 0 ; i < $n ; i++) {
            if(position[t_x][t_y][i] == 1) {
                Tl[t_x][t_y] += 1;
            }
            else {
                if(position[t_x][t_y][i] == 0) {
                    Tr[t_x][t_y] += 1;
                }
            }
        }
    }
}

// Compute impurity using the result of the previous kernel. A reduction could
// be used to compute the Gini score. However, the number of categories is
// usually small: < 100.
__global__ void compute_impurity_hyperplans(unsigned int countL[$s][$t][$c],
                                            unsigned int countR[$s][$t][$c],
                                            unsigned int Tl[$s][$t],
                                            unsigned int Tr[$s][$t],
                                            double impurity[$s][$t]) {

    int t_x, t_y;
    int i;
    double GiniR = 1;
    double GiniL = 1;

    t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the sets
    t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the hyperplans

    if(t_x < $s && t_y < $t) {
        for(i = 0 ; i < $c ; i++) {
            GiniL -= (countL[t_x][t_y][i] * countL[t_x][t_y][i])/
                     (double)(Tl[t_x][t_y] * Tl[t_x][t_y]);
            GiniR -= (countR[t_x][t_y][i] * countR[t_x][t_y][i])/
                     (double)(Tr[t_x][t_y] * Tr[t_x][t_y]);
        }
        // Deal with potential division by 0
        if(Tl[t_x][t_y] == 0)
            GiniL = 1;
        if(Tr[t_x][t_y] == 0)
            GiniR = 1;
        impurity[t_x][t_y] = (Tl[t_x][t_y] * GiniL + Tr[t_x][t_y] * GiniR)/
                             (double)(Tl[t_x][t_y] + Tr[t_x][t_y]);
    }
}

// Derives two sieves from a position array for every hyperplan
// selected in each set.
__global__ void compute_sieves(unsigned int position[$s][$t][$n],
                               unsigned int best_hyperplan[$s],
                               unsigned int sieves[$s][2][$n]) {

    int t_x, t_y;
    t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the sets
    t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the samples

    if(t_x < $s && t_y < $n) {
        if(position[t_x][best_hyperplan[t_x]][t_y] == 0) {
            sieves[t_x][0][t_y] = 1;
        }
        else {
            if(position[t_x][best_hyperplan[t_x]][t_y] == 1)
                sieves[t_x][1][t_y] = 1;
        }
    }
}

// TODO implement this in two kernels like the computation of impurity
// of hyperplans
__global__ void set_impurity(unsigned int cat[$n],
                             unsigned int sieves[$s][2][$n],
                             unsigned int counts[$s][2][$c],
                             unsigned int T[$s][2]) {

    int t_x, t_y, t_z;
    int i;

    t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the categories
    t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the sets
    t_z = blockIdx.z * blockDim.z + threadIdx.z; // Indexes the side

    if(t_x < $c && t_y < $s && t_z < 2) {
        for(i = 0 ; i < $n ; i++) {
            if(sieves[t_y][t_z][i] > 0) {
                if(cat[i] == t_x)
                    counts[t_y][t_z][t_x] += 1;
                T[t_y][t_z] += 1;
            }
        }
    }
}
