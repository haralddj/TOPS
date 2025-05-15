import numpy as np
import matplotlib.pyplot as plt

#Defining constants
def makeStochastic(theta, mu, sigma, T, N, numSim, Y_init):
    dt=T/N
    points=np.full(numSim, Y_init)
    print(points)

    for i in range(1,N):
        dW=np.random.normal(0,np.sqrt(dt), size=(numSim))
        y= points[i-1]
        y_new=y+theta*(mu-y)*dt+sigma*dW
        points=np.vstack((points, y_new))
    tt = np.arange(0, T, dt)
    plt.plot(tt, points)
    plt.xlabel("Tid $(t)$")
    plt.ylabel("Verdi $(S_t)$")
    plt.title(f"Solution of \n $dT_t = \\Theta \\dot (\\mu-Y_t)dt + \\sigma dW_t$\n $Y_0={Y_init}, \\mu = {mu}, \\sigma={sigma}, \\theta = {theta}$")
    plt.show()
    return points, tt

points,tt=makeStochastic(0.05, 1, 0.05, 30, 1000, 1, 0)

