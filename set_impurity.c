// Counting per category
// Each thread is responsible of one category per split
// I don't know if it would be faster using one thread per sample
// and an atomicInc.
__global__ void set_impurity1(unsigned int categories[$n],
                              unsigned int count[$c]) {

    int tid;        // Thread id
    int i;          // Index variable

    tid = blockIdx.x * blockDim.x + threadIdx.x; // Indexes the categories

    if(tid < $c) {
        for(i = 0 ; i < $n ; i++)
            if(categories[i] == tid)
                count[tid] += 1;
    }
}
