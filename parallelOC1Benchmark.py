import matplotlib.pyplot as plt
import parallelOC1

def benchmark(n, plot=False):
    """Generate a data set and run the training method on it"""
    samples = numpy.zeros((4*n, 3))

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # Generating samples
    for i in range(n):
        x = numpy.random.uniform(low=0, high=11)
        y = numpy.random.uniform(x+0.5)
        samples[i][0] = x
        samples[i][1] = y
        samples[i][2] = 0
        if plot:
            ax.plot(x, y, "ro")
    for i in range(n, 2*n):
        y = numpy.random.uniform(10)
        x = numpy.random.uniform(y+0.5)
        samples[i][0] = x
        samples[i][1] = y
        samples[i][2] = 1
        if plot:
            ax.plot(x, y, "bo")
    for i in range(2*n, 3*n):
        x = numpy.random.uniform(low=-5, high=5)
        y = numpy.random.uniform(low=-x-5, high=-x+0.5)
        samples[i][0] = x
        samples[i][1] = y
        samples[i][2] = 2
        if plot:
            ax.plot(x, y, "go")
    for i in range(3*n, 4*n):
        x = numpy.random.uniform(low=-5, high=0)
        y = numpy.random.uniform(low=-x+0.5, high=-x+5)
        samples[i][0] = x
        samples[i][1] = y
        samples[i][2] = 3
        if plot:
            ax.plot(x, y, "ko")

    classifier = parallelOC1.ParallelOC1()
    start = time.time()
    classifier.trainClassifier(samples)
    print "Training Time:", time.time() - start
    start = time.time()
    classifier.classifyInstance(samples)
    print "Classifying Time:", time.time() - start

    # Drawing hyperplans - The result is minimal for the moment...
    if plot:
        queue = list([classifier.DT])
        while len(queue) > 0:
            node = queue.pop()
            if not node.leaf:
                print "Hyperplan:", node.hyperplan
                if node.hyperplan[1] == 0:
                    ax.plot([-node.hyperplan[2]/node.hyperplan[0] for _ in range(len(numpy.linspace(0, 4)))] ,numpy.linspace(0, 4))
                else:
                    f = lambda x: (-node.hyperplan[0]*x -node.hyperplan[2])/node.hyperplan[1]
                    x = numpy.linspace(0, 4)
                    ax.plot(x, f(x))
                queue.insert(0, node.leftChild)
                queue.insert(0, node.rightChild)
        plt.show()

if __name__ == "__main__":
    benchmark(int(sys.argv[1]), plot=True)
