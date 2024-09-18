import numpy as np
import matplotlib.pyplot as plt

#Defining constants
def makeStochastic():
    theta = 0.05
    mu=0
    sigma=0.005
    T=1000
    N=1000
    dt=T/N
    numSim=5
    Y_init=0
    points=np.full(numSim, Y_init)
    print(points)

    for i in range(1,N):
        dW=np.random.normal(0,np.sqrt(dt), size=(numSim))
        y= points[i-1]
        y_new=y+theta*(mu-y)*dt+sigma*dW
        points=np.vstack((points, y_new))
    tt = np.arange(0, T, dt)
    return points, tt

points,tt=makeStochastic()

plt.plot(tt, points)
plt.xlabel("Tid $(t)$")
plt.ylabel("Verdi $(S_t)$")
plt.title(f"Solution of \n $dT_t = \\Theta \\dot (\\mu-Y_t)dt + \\sigma dW_t$\n $Y_0={0}, \\mu = {0}, \\sigma={2}$")
plt.show()
