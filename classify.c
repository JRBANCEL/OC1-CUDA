// $d: number of features
// $n: number of samples
// $l: number of nodes in the tree
// $w: width of the tree structure

// Each thread go through the tree in order to find the category of the sample
__global__ void classify1(double samples[$n][$d],
                          double tree[$l][$w],
                          unsigned int categories[$n]) {

    int tid;        // Thread id
    int i;          // Index variable
    int node = 0;   // Node index
    double point;   // Auxiliary variable for computation

    tid = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the samples

    if(tid < $n) {
        while(tree[node][0] != 0) {
            // Compute on which side of the hyperplan the sample is
            point = tree[node][$d+1];
            for(i = 0 ; i < $d ; i++)
                point += samples[tid][i] * tree[node][i+1];

            // Go on right child
            if(point > 0)
                node = tree[node][0] + 1;
            // Go on left child
            else
                node = tree[node][0];
        }
        categories[tid] = node;
    }
}
