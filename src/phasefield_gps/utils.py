
import numpy as np

def convex_hull(f, g):
    """
    Compute the convex hull of two functions f and g on the interval [0,1]
    """
    pts = np.linspace(0,1,1000)
    min_both = [min(f(x), g(x)) for x in pts]
    increasing = False

    # find local minima
    mins = []
    for i in range(1,len(pts)):
        if increasing:
            if min_both[i] < min_both[i-1]:
                increasing = False
        else:
            if min_both[i] > min_both[i-1]:
                increasing = True
                mins.append((pts[i-1], min_both[i-1]))

    def hull(x):
        if len(mins) == 0:
            return min(f(x), g(x))
        if x < mins[0][0]:
            return min(f(x), g(x))
        for i in range(1,len(mins)):
            if x < mins[i][0]:
                return mins[i-1][1] + (x-mins[i-1][0])*(mins[i][1]-mins[i-1][1])/(mins[i][0]-mins[i-1][0])
        else:
            return min(f(x), g(x))
    return hull

if __name__ == "__main__":
    # Example
    f = lambda x: (x-0.2)**2
    g = lambda x: (x-0.6)**2
    hull = convex_hull(f, g)
    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, 100)
    plt.plot(x, f(x), label="f(x)")
    plt.plot(x, g(x), label="g(x)")
    plt.plot(x, [hull(x_) for x_ in x], label="convex hull")
    plt.legend()
    plt.show()
