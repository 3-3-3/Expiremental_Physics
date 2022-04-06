import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def f(x, t, omega_0, omega, beta, gamma, c):
    #x[0] = omega, x[1] = velocity,
    f_0 = x[1]
    f_1 = gamma*omega_0**2*np.cos(omega*t)-omega_0**2*np.sin(x[0])-2*beta*x[1]-c*x[0]
    return np.array([f_0, f_1])

def oscilator(x_0, v_0, omega_0=1, omega=1, beta=0, gamma=0, c=0, periods=5, tau=0.1):
    '''
    x_0: Initial state of oscilator.
    v_0: Initial velocity (time derivative) of oscilator
    omega: angular frequency
    omega_0:
    beta: driving term
    gamma: dampening term
    periods: number of periods to return
    '''
    T = 2*np.pi/omega_0
    t_max = T*periods

    t = np.arange(0,t_max,tau)
    X_0 = [x_0, v_0]
    return (t, odeint(f, X_0, t, args=((omega_0,omega,beta,gamma, c))))


if __name__ == '__main__':
    beta = 0.5
    gamma = 1
    c=0.2

    x_0 = 2
    dx_0 = 0.01


    for i in range(10):
        X = oscilator(1+dx_0*i, 0, omega_0=2, omega=1, beta=beta, gamma=gamma, c=c, periods=20, tau=0.01)
        x = X[1][:,0]
        v = X[1][:,1]
        t = X[0]
        plt.plot(t,x,label=i)

    #plt.plot(X[0], np.zeros(len(X[0])))
    #plt.plot(X[0], x)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    plt.savefig('Chaos.png')
    plt.show()
