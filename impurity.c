// $c: number of categories
// $d: number of features
// $n: number of samples
// $s: number of splits

// Compute if each point is above or below the given hyperplans
__global__ void impurity1(double samples[$n][$d],
                          double hyperplan[$s][$d+2],
                          unsigned int position[$s][$n]) {

    int t_x, t_y;   // Thread id
    int i;          // Index variable
    double point;   // Value of the point (above or below hyperplan)

    t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the samples
    t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the splits

    // Check boundaries
    if(t_x < $n && t_y < $s) {
        // Check if the hyperplan is valid
        if(hyperplan[t_y][$d+1] > 0) {
            // Compute if the point is above or below the hyperplan
            point = 0;
            for(i = 0 ; i < $d ; i++)
                point += samples[t_x][i] * hyperplan[t_y][i];
            point += hyperplan[t_y][$d];

            if(point > 0)
                position[t_y][t_x] = 0;
            else
                position[t_y][t_x] = 1;
        }
    }
}

// Counting per category per side
// Each thread is responsible of one category per split
__global__ void impurity2(unsigned int categories[$n],
                          unsigned int count[$s][$c][2],
                          unsigned int Tl[$s],
                          unsigned int position[$s][$n]) {

    int t_x, t_y;   // Thread id
    int i;          // Index variable

    t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the categories
    t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the splits

    // Check boundaries
    if(t_x < $c && t_y < $s) {
        // Reset counts
        count[t_y][t_x][0] = 0;
        count[t_y][t_x][1] = 0;
        for(i = 0 ; i < $n ; i++)
            if(categories[i] == t_x) {
                if(position[t_y][i] == 0)
                    count[t_y][t_x][0] += 1;
                else {
                    if(position[t_y][i] == 1)
                        count[t_y][t_x][1] += 1;
                }
            }
    }

    // Thread counting samples above the hyperplan
    if(t_x == $c && t_y < $s) {
        Tl[t_y] = 0;
        for(i = 0 ; i < $n ; i++)
            if(position[t_y][i] == 0)
                Tl[t_y] += 1;
    }
}

// TODO: implement it as a reduction
__global__ void impurity3(unsigned int count[$s][$c][2],
                          unsigned int Tl[$s],
                          double impurity[$s]) {

    int tid;            // Thread id
    int i;              // Index variable
    double GiniL = 1;   // Gini value Left
    double GiniR = 1;   // Gini value Right

    tid = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the splits

    if(tid < $s) {
        if(count[tid][0][0] == 0 && count[tid][0][1] == 0) {
            impurity[tid] = 10;
        }
        else {
            for(i = 1 ; i < $c ; i++) {
                GiniL -= (count[tid][i][0] * count[tid][i][0])/(double)((Tl[tid])*(Tl[tid]));
                GiniR -= (count[tid][i][1] * count[tid][i][1])/(double)(($n-Tl[tid])*($n-Tl[tid]));
            }
            impurity[tid] = (Tl[tid] * GiniL + ($n-Tl[tid]) * GiniR)/((float)$n);
        }
    }
}

// First step of impurity computation in one dimension
__global__ void impurity4(double U[$n],
                          double splits[$n][2],
                          unsigned int position[$n][$n]) {

    int t_x, t_y;   // Thread id

    t_x = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the samples
    t_y = blockIdx.y * blockDim.y + threadIdx.y; // Indexes the splits

    // Check boundaries
    if(t_x < $n && t_y < $s) {
        // Check if the hyperplan is valid
        if(splits[t_y][1] > 0) {
            if(U[t_x] < splits[t_y][0])
                position[t_y][t_x] = 0;
            else
                position[t_y][t_x] = 1;
        }
        else {
            position[t_y][t_x] = 2;
        }
    }
}


