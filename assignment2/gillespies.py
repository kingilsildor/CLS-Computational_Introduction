import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.integrate import solve_ivp


def SIR(t, y0, N, beta, gamma, mu):
    S, I, R = y0
    dSdt = mu * N - (beta * S * I) / N - mu * S
    dIdt = (beta * S * I) / N - gamma * I - mu * I
    dRdt = gamma * I - mu * R
    return [dSdt, dIdt, dRdt]


def calc_rates(N, S, I, R, beta, gamma, mu):
    return np.array([
        mu * N,             # Births                S = S + 1
        beta * S * I / N,   # Infection             S = S - 1, I = I + 1
        gamma * I,          # Recovery              I = I - 1, R = R + 1
        mu * S,             # Death of Susceptible  S = S - 1
        mu * I,             # Death of Infected     I = I - 1
        mu * R              # Death of Recovered    R = R - 1
    ])


def calc_delta_t(rates_total):
    return -np.log(np.random.uniform()) / rates_total


def calc_deterministic(t, init_values, init_param):
    t_eval = np.linspace(0, t[-1], len(t))
    sol = solve_ivp(SIR, [0, t[-1]], y0=init_values,
                    args=init_param, t_eval=t_eval, dense_output=True)
    return sol.y


def calc_interpolate(results, epochs, time_points, t):
    results_interp = np.empty((epochs, 3, len(time_points)))

    for i in range(epochs):
        history = results[i]
        if len(history) > 0:
            t_list = np.linspace(0, t, len(history))
            results_interp[i, 0, :] = np.interp(
                time_points, t_list, history[:, 0])
            results_interp[i, 1, :] = np.interp(
                time_points, t_list, history[:, 1])
            results_interp[i, 2, :] = np.interp(
                time_points, t_list, history[:, 2])

    return results_interp


def calc_equilibrium(beta, gamma, mu):
    S_eq = (gamma + mu) / beta
    I_eq = mu * (1 - S_eq) / (gamma + mu)

    return (round(S_eq, 2), round(I_eq, 2))


def calc_coord(matrix, coord, value=1):
    detail = matrix.shape[0]
    x_index = int(coord[0] * detail)
    y_index = int(coord[1] * detail)

    if 0 <= x_index < detail and 0 <= y_index < detail:
        matrix[y_index, x_index] += value

    return matrix


def calc_last_value(arr):
    last_values = np.empty((3, 1))

    for col in range(arr.shape[1]):
        data = arr[:, col]
        not_nan_indices = ~np.isnan(data)
        last_values[col] = data[not_nan_indices][-1]

    return last_values


def gillespie(N, S, I, beta, gamma, mu, t_max, chunk_size=1000):
    R = 0
    t_list = [0]
    extinction_event = False
    # Allocate memory for history
    history = np.empty((chunk_size, 3), dtype=float)

    step = 0
    while t_list[-1] < t_max:
        rates = calc_rates(N, S, I, R, beta, gamma, mu)
        rates_total = rates.sum()
        delta_t = np.random.exponential(1/rates_total)

        P = np.random.uniform() * rates_total
        cumulative_rates = np.cumsum(rates)

        event_index = np.searchsorted(cumulative_rates, P)

        if event_index == 0:
            if S > 0:
                S += 1
        elif event_index == 1:
            if S > 0:
                S -= 1
                I += 1
        elif event_index == 2:
            if I > 0:
                I -= 1
                R += 1
        elif event_index == 3:
            if S > 0:
                S -= 1
        elif event_index == 4:
            if I > 0:
                I -= 1
        elif event_index == 5:
            if R > 0:
                R -= 1

        # Increase the size of history if needed
        if step >= history.shape[0]:
            history = np.resize(history, (history.shape[0] + chunk_size, 3))
        history[step] = [S, I, R]

        if S <= 0 or I <= 0:
            history[step] = [np.nan, np.nan, np.nan]
            extinction_event = True

        t_list.append(t_list[-1] + delta_t)
        step += 1

    history = history[:step]
    return history, t_list, extinction_event


def simulate(epochs, N, S, I, beta, gamma, mu, t_max, chunk_size=1000):
    results = []
    extinsions = []

    for _ in range(epochs):
        history, t_list, extinsion_event = gillespie(
            N, S, I, beta, gamma, mu, t_max, chunk_size)
        results.append(history)
        extinsions.append(extinsion_event)

    results = np.array(results, dtype=object)
    return results, extinsions


def plot_simulation(results, extinsions, epochs, N, S, I, beta, gamma, mu, t):
    time_points = np.linspace(0, t, 1000)
    results_interp = calc_interpolate(results, epochs, time_points, t)

    plt.figure(figsize=(12, 6))
    for i in range(epochs):
        plt.plot(time_points, results_interp[i, 0, :], color='blue', alpha=0.1)
        plt.plot(time_points, results_interp[i, 1, :], color='red', alpha=0.1)
        plt.plot(
            time_points, results_interp[i, 2, :], color='green', alpha=0.1)

    sol = calc_deterministic(time_points, [S, I, 0], [N, beta, gamma, mu])
    plt.plot(time_points, sol[0], color='blue', linestyle='dashed')
    plt.plot(time_points, sol[1], color='red', linestyle='dashed')
    plt.plot(time_points, sol[2], color='green', linestyle='dashed')

    mean_S = np.nanmean(results_interp[:, 0, :], axis=0)
    mean_I = np.nanmean(results_interp[:, 1, :], axis=0)
    mean_R = np.nanmean(results_interp[:, 2, :], axis=0)
    plt.plot(time_points, mean_S, color='blue',
             label='Suseptible', linewidth=2)
    plt.plot(time_points, mean_I, color='red', label=' Infected', linewidth=2)
    plt.plot(time_points, mean_R, color='green', label='Removed', linewidth=2)

    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title(
        f'Gillespie Simulation Over {epochs} Epochs (SIR Model)\n Extinctions: {sum(extinsions)}, β={beta}, γ={gamma}, μ={mu}')
    plt.grid()
    plt.legend()
    plt.show()


def plot_phase(N, epochs, beta, gamma, mu, lines, t):
    time_points = np.linspace(0, t, 1000)
    colors = sns.color_palette(None, lines)
    plt.figure(figsize=(12, 6))

    for color, infected_porpotion in enumerate(np.linspace(0, 1, lines)):
        I = int(N * infected_porpotion)
        S = N - I
        results, _ = simulate(epochs, N, S, I, beta, gamma, mu, t)
        results_interp = calc_interpolate(results, epochs, time_points, t)

        S_mean = np.nanmean(results_interp[:, 0, :], axis=0)
        I_mean = np.nanmean(results_interp[:, 1, :], axis=0)

        for i in range(epochs):
            plt.plot(results_interp[i, 0, :],
                    results_interp[i, 1, :], color=colors[color], alpha=0.1)
        # plt.plot(S_mean, I_mean, color=colors[color])

    x_equilibrium, y_equilibrium = calc_equilibrium(beta, gamma, mu)
    plt.scatter(x_equilibrium * N, y_equilibrium *
                N, color='red')

    plt.xlabel('Suseptible Population (S)')
    plt.ylabel('Infected Population (I)')
    plt.title(
        f'Phase Space Over {epochs} Epochs (SIR Model)\n β={beta}, γ={gamma}, μ={mu}')
    plt.grid()
    plt.show()


def plot_dynamics(N_list, epochs, beta, gamma, mu, t):
    time_points = np.linspace(0, t, 1000)
    plt.figure(figsize=(12, 6))

    for N in N_list:
        I = N // 100
        S = N - I

        results, _ = simulate(epochs, N, S, I, beta, gamma, mu, t)
        results_interp = calc_interpolate(results, epochs, time_points, t)
        results_mean = np.nanmean(results_interp[:, 1, :], axis=0)
        plt.plot(time_points, results_mean,
                 label=f'N={N}', linewidth=2, color="black")

    plt.yscale('log')
    plt.yticks(N_list)
    plt.xlabel('Time (days)')
    plt.ylabel('Infected Population (I)')
    plt.title(
        f'Infected Dynamics Over {epochs} Epochs (SIR Model)\n β={beta}, γ={gamma}, μ={mu}')
    plt.grid()
    plt.show()


def plot_equilibrium_heatmap(detail, epochs, N, S, I, beta, gamma, mu, t):
    matrix = np.zeros((detail, detail))
    time_points = np.linspace(0, t, 1000)

    x_equilibrium, y_equilibrium = calc_equilibrium(beta, gamma, mu)
    calc_coord(matrix, (x_equilibrium, y_equilibrium), value=10e6)

    results, _ = simulate(epochs, N, S, I, beta, gamma, mu, t)

    for history in results:
        print(history)

        S, I, _ = calc_last_value(history)
        S, I = S[0], I[0]

        x_coord = S / N
        y_coord = I / N
        calc_coord(matrix, (x_coord, y_coord))

    matrix = np.log1p(matrix)
    plt.figure(figsize=(12, 6))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.show()

def plot_variance(epochs, N, S, I, beta, gamma, mu, t):
    time_points = np.linspace(0, t, 1000)
    results, _ = simulate(epochs, N, S, I, beta, gamma, mu, t)
    results_interp = calc_interpolate(results, epochs, time_points, t)

    S_all = results_interp[:, 0, :]
    I_all = results_interp[:, 1, :]
    S_mean = np.nanmean(results_interp[:, 0, :], axis=0)
    I_mean = np.nanmean(results_interp[:, 1, :], axis=0)
    S_var = np.nanvar(results_interp[:, 0, :], axis=0)
    I_var = np.nanvar(results_interp[:, 1 :], axis=0)

    # covariance = np.nanmean((S_all - S_mean) * (I_all - I_mean), axis=0)
    plt.figure(figsize=(12, 6))
    sol = calc_deterministic(time_points, [S, I, 0], [N, beta, gamma, mu])
    plt.plot(time_points, sol[0], color='blue', label='Suseptible deterministic' ,linestyle='dashed')
    plt.plot(time_points, S_mean, color='blue', label='Suseptible mean')
    plt.fill_between(time_points, sol[0], S_mean, color='blue', alpha=0.2)
    # plt.plot(time_points, S_var, color='blue', label='Suseptible', linewidth=2, linestyle='dashed')
    
    # plt.fill_between(time_points, S_mean - S_var, S_mean + S_var, color='blue', alpha=0.2)

    # plt.plot(time_points, sol[1], color='red', linestyle='dashed')
    # plt.plot(time_points, I_mean, color='red', label=' Infected', linewidth=2)
    # plt.fill_between(time_points, I_mean - I_var, I_mean + I_var, color='red', alpha=0.2)

    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title(
        f'Gillespie Simulation Over {epochs} Epochs (SIR Model)\n β={beta}, γ={gamma}, μ={mu}')
    plt.grid()
    plt.legend()
    plt.show()

def plot_scatter_si(epochs, N, S, I, beta, gamma, mu, t):
    time_points = np.linspace(0, t, 1000)
    results, _ = simulate(epochs, N, S, I, beta, gamma, mu, t)
    results_interp = calc_interpolate(results, epochs, time_points, t)

    S_all = results_interp[:, 0, :]
    I_all = results_interp[:, 1, :]

    plt.figure(figsize=(8, 6))
    plt.scatter(S_all, I_all, color='purple')
    plt.xlabel('S (Susceptibles)')
    plt.ylabel('I (Infected)')
    plt.show()



if __name__ == '__main__':
    N = 10e3
    I = N // 100
    S = N - I
    beta = 1.67
    gamma = 0.47
    mu = 0.01
    t = 356 * 1
    epochs = 5

    start_time = time.time()
    # results, extinsions = simulate(epochs, N, S, I, beta, gamma, mu, t)
    # plot_equilibrium_heatmap(100, epochs, [10e1, 10e2], beta, gamma, mu, t)
    # plot_equilibrium_heatmap(10, epochs, N, S, I, beta, gamma, mu, t)
    # plot_phase(N, epochs, beta, gamma, mu, 10, t)
    # plot_variance(epochs, N, S, I, beta, gamma, mu, t)
    plot_scatter_si(epochs, N, S, I, beta, gamma, mu, t)
    # plot_dynamics([10e1, 10e2, 10e3], epochs, beta, gamma, mu, t)
    end_time = time.time()
    # plot_simulation(results, extinsions, epochs, N, S, I, beta, gamma, mu, t)

    print(
        f"parameters: epochs={epochs}, N={N}, I={I}, beta={beta}, gamma={gamma}, mu={mu}, t={t}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
