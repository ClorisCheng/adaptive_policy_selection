
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

from GAPS import GAPSEstimator

class MassSpringDamper:
    def __init__(self, s, d, m, dt):
        self.s = s # spring constant
        self.d = d # damping constant
        self.m = m # mass
        
        # continuous linear system matrices
        self.A = np.array([[0, 1], [-self.s/self.m, -self.d/self.m]])
        self.B = np.array([[0], [1/self.m]])
        n, m = self.B.shape
        self.C = np.eye(n)
        self.D = np.zeros((n, m))

        # get discrete linear system matrices
        self.dt = dt
        sys_c = (self.A, self.B, self.C, self.D)
        Ad, Bd, Cd, Dd, _ = sig.cont2discrete(sys_c, dt)
        self.Ad = Ad
        self.Bd = Bd

    def step(self, x, u):
        return self.Ad @ x + self.Bd @ u
    

def get_cost(x, u, Q, R):
    ''' Quadratic cost function
    Args:
        x: (2, 1)
        u: (1, 1)
        Q: (2, 2)
        R: (1, 1)
    Returns:
        cost: (1, 1)
    '''
    return x.T @ Q @ x + u.T @ R @ u

def get_cost_grad(x, u, Q, R):
    '''
    Args:
        x: (2, 1)
        u: (1, 1)
        Q: (2, 2)
        R: (1, 1)
    Returns:
        dx: (2, 1)
        du: (1, 1)
    '''
    # return gradient of cost function wrt x and u
    return 2 * x.T @ Q, 2 * u @ R

def get_u(x, theta):
    '''
    Args:
        x: (2, 1)
        theta: (1, 2)
    Returns:
        u: (1, 1)
    '''
    return -1 * np.dot(theta, x)

def get_u_grad(x, theta):
    ''' u = -theta^T x
    Args:
        x: (2, 1)
        theta: (1, 2)
    Returns:
        dx: (1, 2)
        dtheta: (1, 2)
    '''
    return -theta, -x.T

def run(d, s, m, dt, buf_len, x0, theta0, T, disturbance, gradient_disturbance, lr):
    sys = MassSpringDamper(s, d, m, dt)
    Q = np.eye(2)
    R = 0.1 * np.eye(1)
    xt = x0 # (2, 1)
    thetat = theta0 # (1, 2)
    estimator = GAPSEstimator(buffer_length=buf_len)
    x_log = []
    cost_log = []
    prev_dgdx = None
    prev_dgdu = None

    for t in range(T):
        # get control input
        ut = get_u(xt, thetat) # (1, 1)
        dudx, dudtheta = get_u_grad(xt, thetat) # (1, 2), (1, 2)
        estimator.add_partial_u(dudx, dudtheta)
        
        # log data
        x_log.append(xt)
        cost_log.append(get_cost(xt, ut, Q, R))
        
        # g: dynamics function
        # get system derivatives
        dgdx = sys.A + gradient_disturbance * np.random.randn(2, 2)
        dgdu = sys.B + gradient_disturbance * np.random.randn(2, 1)
        prev_dgdx = dgdx
        prev_dgdu = dgdu
        # f: cost function
        # get quadratic cost derivatives
        dfdx = 2 * xt.T @ Q
        dfdu = 2 * ut.T @ R

        # dfdx and dfdu should be (2, )
        G = estimator.update(dfdx, dfdu, prev_dgdx, prev_dgdu)
        thetat -= lr * G
        xt = sys.step(xt, ut)
        # add disturbance to 2nd entry of state vector
        xt[1] += disturbance * np.random.randn()

    return x_log, cost_log

def plot_x(x_log):
    x_log = np.hstack(x_log).T
    plt.scatter(x_log[0, :], x_log[1, :])
    plt.show()
        
def main():
    dtype = np.float32
    d = 0.1
    s = 1
    m = 1
    dt = 0.01
    buf_len = 10
    x0 = np.array([0, 0], dtype=dtype).reshape(2, 1)
    theta0 = np.array([1, 1], dtype=dtype).reshape(1, 2)
    T = 10
    disturbance = 0.1
    gradient_disturbance = 0.0
    lr = 0.1
    
    x_log, cost_log = run(d, s, m, dt, buf_len, x0, theta0, T, disturbance, gradient_disturbance, lr)
    print(x_log)
    print(cost_log)
    plot_x(x_log)



if __name__ == "__main__":
    main()