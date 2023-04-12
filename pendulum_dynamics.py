from pendulum import *
import matplotlib.pyplot as plt

_Q = torch.eye(2, dtype=torch.double)
_R = 0.1 * torch.eye(1, dtype=torch.double)

def _cost(xs, us):
    xQs = xs @ _Q
    uRs = us @ _R
    return torch.sum(xs * xQs, axis=-1) + torch.sum(us * uRs, axis=-1)


def run_gaps(grad_disturbance, T=100, path=None, walk=False):
    dt = 1e-2
    N = 10

    buf_len = 10*int(1.0 / dt)
    rate = 1e-2
    estimator = GAPSEstimator(buffer_length=buf_len)
    theta = torch.tensor(pendulum_gains_lqrd(1.0, 1.0, dt)) # torch.tensor(1, 2)
    prev_dgdx = None
    prev_dgdu = None

    np.random.seed(100)
    masses = 2 ** np.random.uniform(-1, 1, size=N)
    if walk:
        disturbance = ulprocess(seed=0, noise=0.5 * dt, gamma=0.95)
    else:
        disturbance = ulprocess(seed=0, noise=8.0 * dt, gamma=0.0)
    xs = torch.zeros((2, 2), dtype=torch.double)

    x_log = []
    mass_log = []
    theta_log_LQ = []
    theta_log_ours = []
    cost_log = []

    for mass in tqdm.tqdm(masses):
        system = InvertedPendulum(m=mass, l=1.0)

        def dynamics(x, u):
            return system.step(x[None, :], u[None, :], 0.0, dt)[0]

        kp_LQ, kd_LQ = pendulum_gains_lqrd(mass, l=1.0, dt=dt)
        controller_LQ = PDController(kp_LQ, kd_LQ)

        for i in range(T):
            # Get actions.
            us = torch.concat([
                controller_ours(xs[0], theta)[None, :],
                controller_LQ(xs[1][None, :]),
            ], axis=0)
            # xs[0]: torch.Tensor(2, ). xs[0][None, :]: torch.Tensor(1, 2)

            # Log everything.
            x_log.append(xs)
            theta_log_LQ.append(np.array([kp_LQ, kd_LQ]))
            theta_log_ours.append(theta)
            cost_log.append(_cost(xs, us))
            mass_log.append(mass)

            

            # Get controller derivatives.
            dudx, dudtheta = torch.autograd.functional.jacobian(
                controller_ours,
                (xs[0], theta),
                vectorize=True,
            )
            estimator.add_partial_u(dudx.detach().numpy(), dudtheta.detach().numpy())

            # Get system derivatives.
            dgdx, dgdu = torch.autograd.functional.jacobian(dynamics, (xs[0], us[0]), vectorize=True)

            eps_dgdx = torch.rand_like(dgdx) * grad_disturbance
            eps_dgdu = torch.rand_like(dgdu) * grad_disturbance

            dgdx = dgdx + eps_dgdx
            dgdu = dgdu + eps_dgdu

            dfdx, dfdu = torch.autograd.functional.jacobian(_cost, (xs[0], us[0]), vectorize=True)
            # Gradient sanity check.
            assert np.dot(dfdx, xs[0]) >= 0
            assert np.dot(dfdu, us[0]) >= 0
            derivatives = (dfdx, dfdu, prev_dgdx, prev_dgdu)
            prev_dgdx = dgdx
            prev_dgdu = dgdu

            # Gradient step.
            G = estimator.update(*map(t2np, derivatives))
            theta = theta - rate * G

            # Dynamics step.
            xs = system.step(xs, us, 0.0, dt)
            xs[:, 1] += next(disturbance)

    # Save data.
    x_log = np.stack(x_log)
    cost_log = np.stack(cost_log)
    
    if path is not None:
        np.savez(
            # args.outpath,
            path,
            dt=dt,
            x_log=x_log,
            theta_log_LQ=np.stack(theta_log_LQ),
            theta_log_ours=np.stack(theta_log_ours),
            cost_log=cost_log,
            mass_log=np.array(mass_log),
        )
    else:
        pass

    return x_log[:, 0], cost_log[:, 0]



def plot_gaps(x_log, cost_log, path=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(cost_log)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cost")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        # plt.show()
    else:
        plt.show()

def plot(cost_arr, path=None):
    std = np.std(cost_arr, axis=0)
    mean = np.mean(cost_arr, axis=0)
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    ax[0].plot(mean)
    ax[0].fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.5)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Cost")

    for i in range(cost_arr.shape[0]):
        ax[1].plot(cost_arr[i], label=f"{i}")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Cost")
    ax[1].legend()
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        # plt.show()
    else:
        plt.show()
    
def main():
    disturbance_list = torch.linspace(0.0, 10, 10)
    cost_arr = None # np.zeros((len(disturbance_list), T))
    for i, disturbance in enumerate(disturbance_list):
        x_log, cost_log = run_gaps(disturbance, path=f"pendulum_gaps_{i}.npz")
        cost_arr = cost_log if cost_arr is None else np.vstack((cost_arr, cost_log))
    plot(cost_arr, path=f"graphs/pendulum_gaps_disturbance.png")


if __name__ == "__main__":
    main()