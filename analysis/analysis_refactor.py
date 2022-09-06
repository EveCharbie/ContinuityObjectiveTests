
import os
import pickle
import biorbd
import numpy as np
from matplotlib import pyplot as plt

class DataFrame:

    def __init__(self, *keys):
        self._keys = keys
        self._data = []

    def _select(self, **keywords):
        if (result := set(keywords.keys()) - set(self._keys)) != set():
            raise KeyError(f"Invalid keys {result}. Keys must be in {set(self._keys)}.")

        out = []
        for keys, data in self._data:
            select = True
            for key, value in keywords.items():
                if keys[key] != value:
                    select = False
                    break
            if select:
                out.append(data)

        return out

    def __call__(self, **keywords):
        return self._select(**keywords)

    def add(self, data, **keyword):
        if set(keyword.keys()) != set(self._keys):
            raise KeyError(f"Keys must match {set(self._keys)}.")

        d = dict.fromkeys(self._keys)
        for key, value in keyword.items():
            d[key] = value

        self._data.append((d, data))


def load_sol(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def extract_keys(filename):
    filename = filename.split("-")

    if filename[0] == "objective":
        type_, var, phase, case, weight, iter_, weight_sphere = filename[0], filename[1], filename[2], filename[3], filename[4], filename[5], None
        case = int(case)
        if weight == "1000.0":
            weight = "1K"
        elif weight == "1000000.0":
            weight = "1M"
        elif weight == "1000000000.0":
            weight = "1G"
        iter_ = int(iter_)
        return dict(type=type_, var=var, phase=phase, case=case, weight=weight, iter=iter_, weight_sphere=weight_sphere)

    elif filename[0] == "constraint":
        type_, var, phase, case, weight, iter_, weight_sphere = filename[0], None, None, filename[1], None, None, None
        case = int(case)
        return dict(type=type_, var=var, phase=phase, case=case, weight=weight, iter=iter_, weight_sphere=weight_sphere)

    elif filename[0] == "unconstrained":
        type_, var, phase, case, weight, iter_, weight_sphere = filename[0], None, None, filename[1], filename[2], filename[3], filename[4]
        case = int(case)
        return dict(type=type_, var=var, phase=phase, case=case, weight=weight, iter=iter_, weight_sphere=weight_sphere)

    else:
        raise Exception(filename)


def extract_opt_time(solution):
    return solution.solver_time_to_optimize


def extract_total_time(solution):
    return solution.total_time


def extract_phase_time(solution):
    return solution.phase_time[-1]


def extract_iteration(solution):
    return solution.iterations


def extract_status(solution):
    return solution.status


def extract_q(solution):
    return solution.states["q"]


def extract_qdot(solution):
    return solution.states["qdot"]


def extract_tau(solution):
    return solution.controls["tau"]


def extract_cost(solution):
    return solution.cost

def analyse(extractor, solutions):
    vals = np.array([*map(extractor, solutions)])
    vals.shape = vals.size
    val_avg = np.mean(vals)
    val_std = np.std(vals)
    return vals, val_avg, val_std


def convergence_rate(solutions):
    status = np.array([*map(extract_status, solutions)])
    converged = status == 0
    convergence = len(status[converged] ) /len(status)
    return status, converged, convergence


def plot_hist(ax, dist, title, xlabel, bins="rice"):
    _, bins, _ = ax.hist(dist, bins=bins)
    ax.set_xticks(bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def compute_transpersion(m, sol, nb_shooting):

    def transpersion_dist_sum(x):

        shpere_list = [(0, 0.05, 0), (0, 0.55, -0.85), (0, 0.75, 0.2), (0, -0.45, 0), (0, 1.4, 0.5), (0, 2, 1.2)]
        bound = 0.35
        marker = m.marker(biorbd.GeneralizedCoordinates(x[:m.nbQ()]), 1).to_array()

        sum = 0
        for i in range(len(shpere_list)):
            distance = np.linalg.norm(np.array(shpere_list[i]) - marker) - bound
            if distance < 0:
                sum += np.abs(distance)

        return sum

    def runge_kutta_4(m, x0, u, t, N, n_step):
        h = t / (N - 1) / n_step
        x = np.zeros((x0.shape[0], n_step + 1))
        x[:, 0] = x0
        for i in range(1, n_step + 1):
            k1_q = x[m.nbQ():, i - 1]
            k1_qdot = m.ForwardDynamics(x[:m.nbQ(), i - 1], k1_q, u).to_array()
            k2_q = x[m.nbQ():, i - 1] + h / 2 * k1_qdot
            k2_qdot = m.ForwardDynamics(x[:m.nbQ(), i - 1] + h / 2 * k1_q, k2_q, u).to_array()
            k3_q = x[m.nbQ():, i - 1] + h / 2 * k2_qdot
            k3_qdot = m.ForwardDynamics(x[:m.nbQ(), i - 1] + h / 2 * k2_q, k3_q, u).to_array()
            k4_q = x[m.nbQ():, i - 1] + h * k3_qdot
            k4_qdot = m.ForwardDynamics(x[:m.nbQ(), i - 1] + h * k3_q, k4_q, u).to_array()
            x[:, i] = x[:, i - 1] + h / 6 * (np.hstack((k1_q, k1_qdot)) + 2 * np.hstack((k2_q, k2_qdot)) + 2 * np.hstack((k3_q, k3_qdot)) + np.hstack((k4_q, k4_qdot)))
        return x

    nb_sub_intervals = 10
    x = sol.states["all"]
    u = sol.controls['all']
    t = sol.parameters['time'][0][0]
    # dt = t / nb_shooting

    x_sub_interval = np.zeros((4, nb_sub_intervals * nb_shooting + 1))
    x_sub_interval[:, 0] = x[:, 0]
    for i in range(nb_shooting):
        x_sub_interval[:, nb_sub_intervals * i + 1 : nb_sub_intervals * (i+1) + 1] = runge_kutta_4(m, x[:, i], u[:, i], t, nb_shooting, nb_sub_intervals)[:, 1:]

    transpersion = np.zeros((nb_sub_intervals * nb_shooting + 1))
    for i in range(nb_sub_intervals * nb_shooting + 1):
        transpersion[i] = transpersion_dist_sum(x_sub_interval[:, i])

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(x_sub_interval[0, :], '-b')
    ax[1].plot(x_sub_interval[1, :], '-b')
    ax[0].plot(range(0, nb_shooting*nb_sub_intervals+1, nb_sub_intervals), x[0, :], 'ok')
    ax[1].plot(range(0, nb_shooting*nb_sub_intervals+1, nb_sub_intervals), x[1, :], 'ok')
    ax[2].plot(transpersion)
    ax[2].plot(range(0, nb_shooting*nb_sub_intervals+1, nb_sub_intervals), transpersion[range(0, nb_shooting*nb_sub_intervals+1, nb_sub_intervals)], 'ok')
    plt.show()

    return transpersion


def graph_convergence(df, m, nb_shooting):

    # plt.figure()

    constraint = np.array([])
    for i, sol in enumerate(df(type='constraint')):
        if sol.status == 0:
            if len(constraint) == 0:
                sol_transpersion = compute_transpersion(m, sol, nb_shooting)
                constraint = np.array([sol.cost(), sol.time(), sol_transpersion])

    # plt.scatter(constraint[0, :], constraint[1, :], c=[constraint], marker='s', label='Constrained')
    # plt.xlabel("")
    # plt.legend()
    # plt.show()
    return


#########   Loading data   #########

directory = "../solutions"
filenames = [*filter(lambda filename: ".pickle" in filename, os.listdir(directory))]

solutions = map(load_sol, map(lambda f: f"{directory}/{f}", filenames))
keys = map(extract_keys, filenames)

df = DataFrame("type", "var", "phase", "case", "weight", "weight_sphere", "iter")

nb_shooting = 500
m = biorbd.Model("../models/pendulum_maze.bioMod")

for k, sol in zip(keys, solutions):
    sol.case = k["case"]
    df.add(sol, **k)



#########   Continuity constraint   #########

constraints = np.array(df(type="constraint"))

status, converged, convergence = convergence_rate(constraints)
print("Convergence rate for constraints:", convergence *100, "%")

costs, cost_avg, cost_std = analyse(extract_cost, constraints[converged])

opt_times, opt_time_avg, opt_time_std = analyse(extract_opt_time, constraints[converged])

phase_times, phase_time_avg, phase_time_std = analyse(extract_phase_time, constraints[converged])
shooting_time_intervals = phase_times / nb_shooting

fig, (cax, tax, pax) = plt.subplots(1, 3, figsize=(25, 5))
plot_hist(cax, costs, "Costs distribution", "cost")
plot_hist(tax, opt_times, "Optimization times distribution", "time (s)")
plot_hist(pax, phase_times, "Phase times distribution", "time (s)")
plt.tight_layout()
fig.suptitle("All 10000 max iterations, 500 shootings, converged only")
fig.savefig("../figures/constraint.pdf")



#########   Variable first max iterations   #########

init100it = df(type="objective", phase="initial", var="varit", iter=100)
init1000it = df(type="objective", phase="initial", var="varit", iter=1000)
init10000it = df(type="objective", phase="initial", var="varit", iter=10000)
initialvarits = np.array([init100it, init1000it, init10000it])

final100it = df(type="objective", phase="final", var="varit", iter=100)
final1000it = df(type="objective", phase="final", var="varit", iter=1000)
final10000it = df(type="objective", phase="final", var="varit", iter=10000)
finalvarits = np.array([final100it, final1000it, final10000it])

final_convergence_rate = []
final_converged = []
for varit in finalvarits:
    _, converged, convergence = convergence_rate(varit)
    final_convergence_rate.append(convergence)
    final_converged.append(converged)

print("Final convergence rate for objective->contrains (varit):", *map(lambda x: f"{ x *100:.1f} %", final_convergence_rate))

final_costs = []
final_opt_times = []
final_phase_times = []
for i in range(len(finalvarits)):
    varit = finalvarits[i][final_converged[i]]
    costs, _, _ = analyse(extract_cost, varit)
    opt_times, _, _ = analyse(extract_total_time, varit)
    phase_times, _, _ = analyse(extract_phase_time, varit)
    final_costs.append(costs)
    final_opt_times.append(opt_times)
    final_phase_times.append(phase_times)

initial_convergence_rate = []
for varit in initialvarits:
    _, converged, convergence = convergence_rate(varit)
    initial_convergence_rate.append(convergence)

initial_costs = []
initial_opt_times = []
initial_phase_times = []
for i in range(len(initialvarits)):
    varit = initialvarits[i][final_converged[i]]
    costs, _, _ = analyse(extract_cost, varit)
    opt_times, _, _ = analyse(extract_opt_time, varit)
    phase_times, _, _ = analyse(extract_phase_time, varit)
    initial_costs.append(costs)
    initial_opt_times.append(opt_times)
    initial_phase_times.append(phase_times)

print("Initial convergence rate for objective (varit):", *map(lambda x: f"{ x *100:.1f} %", initial_convergence_rate))

fig, ((cax100, tax100, pax100), (cax1K, tax1K, pax1K), (cax10K, tax10K, pax10K)) = plt.subplots(3, 3, figsize=(25, 15))
plot_hist(cax100, final_costs[0], "Costs distribution of final optimisation with 100 initial iterations", "cost")
plot_hist(cax1K, final_costs[1], "Costs distribution of final optimisation with 1000 initial iterations", "cost")
plot_hist(cax10K, final_costs[2], "Costs distribution of final optimisation with 10000 initial iterations", "cost")

plot_hist(tax100, final_opt_times[0], "Total optimistation time distribution of final optimisation with 100 initial iterations", "time (s)")
plot_hist(tax1K, final_opt_times[1], "Total optimistation time distribution of final optimisation with 1000 initial iterations", "time (s)")
plot_hist(tax10K, final_opt_times[2], "Total optimistation time distribution of final optimisation with 10000 initial iterations", "time (s)")

plot_hist(pax100, final_phase_times[0], "Phase time distribution of final optimisation with 100 initial iterations", "time (s)")
plot_hist(pax1K, final_phase_times[1], "Phase time distribution of final optimisation with 1000 initial iterations", "time (s)")
plot_hist(pax10K, final_phase_times[2], "Phase time distribution of final optimisation with 10000 initial iterations", "time (s)")
plt.tight_layout()
fig.suptitle("All weight 1M, 500 shootings, converged only")
fig.savefig("../figures/final_varit.pdf")

fig, ((cax100, tax100, pax100), (cax1K, tax1K, pax1K), (cax10K, tax10K, pax10K)) = plt.subplots(3, 3, figsize=(25, 15))
plot_hist(cax100, initial_costs[0], "Costs distribution of initial optimisation with 100 initial iterations", "cost")
plot_hist(cax1K, initial_costs[1], "Costs distribution of initial optimisation with 1000 initial iterations", "cost")
plot_hist(cax10K, initial_costs[2], "Costs distribution of initial optimisation with 10000 initial iterations", "cost")

plot_hist(tax100, initial_opt_times[0], "Optimistation time distribution of initial optimisation with 100 initial iterations", "time (s)")
plot_hist(tax1K, initial_opt_times[1], "Optimistation time distribution of initial optimisation with 1000 initial iterations", "time (s)")
plot_hist(tax10K, initial_opt_times[2], "Optimistation time distribution of initial optimisation with 10000 initial iterations", "time (s)")

plot_hist(pax100, initial_phase_times[0], "Phase time distribution of initial optimisation with 100 initial iterations", "time (s)")
plot_hist(pax1K, initial_phase_times[1], "Phase time distribution of initial optimisation with 1000 initial iterations", "time (s)")
plot_hist(pax10K, initial_phase_times[2], "Phase time distribution of initial optimisation with 10000 initial iterations", "time (s)")
plt.tight_layout()
fig.suptitle("All weight 1M, 500 shootings, converged only")
fig.savefig("../figures/initial_varit.pdf")



#########   Variable weight on continuity objective   #########

init100poids = df(type="objective", phase="initial", var="varpoids", weight="1K")
init1000poids = df(type="objective", phase="initial", var="varpoids", weight="1M")
init10000poids = df(type="objective", phase="initial", var="varpoids", weight="1G")
initialvarpoids = np.array([init100poids, init1000poids, init10000poids])

final100poids = df(type="objective", phase="final", var="varpoids", weight="1K")
final1000poids = df(type="objective", phase="final", var="varpoids", weight="1M")
final10000poids = df(type="objective", phase="final", var="varpoids", weight="1G")
finalvarpoids = np.array([final100poids, final1000poids, final10000poids])

final_convergence_rate = []
final_converged = []
for varpoids in finalvarpoids:
    _, converged, convergence = convergence_rate(varpoids)
    final_convergence_rate.append(convergence)
    final_converged.append(converged)

print("Final convergence rate for objective->contrains (varpoids):", *map(lambda x: f"{ x *100:.1f} %", final_convergence_rate))

final_costs = []
final_opt_times = []
final_phase_times = []
for i in range(len(finalvarpoids)):
    varpoids = finalvarpoids[i][final_converged[i]]
    costs, _, _ = analyse(extract_cost, varpoids)
    opt_times, _, _ = analyse(extract_total_time, varpoids)
    phase_times, _, _ = analyse(extract_phase_time, varpoids)
    final_costs.append(costs)
    final_opt_times.append(opt_times)
    final_phase_times.append(phase_times)

initial_convergence_rate = []
for varpoids in initialvarpoids:
    _, converged, convergence = convergence_rate(varpoids)
    initial_convergence_rate.append(convergence)

initial_costs = []
initial_opt_times = []
initial_phase_times = []
for i in range(len(initialvarpoids)):
    varpoids = initialvarpoids[i][final_converged[i]]
    costs, _, _ = analyse(extract_cost, varpoids)
    opt_times, _, _ = analyse(extract_opt_time, varpoids)
    phase_times, _, _ = analyse(extract_phase_time, varpoids)
    initial_costs.append(costs)
    initial_opt_times.append(opt_times)
    initial_phase_times.append(phase_times)

print("Initial convergence rate for objective->constraints (varpoids):", *map(lambda x: f"{ x *100:.1f} %", initial_convergence_rate))

fig, ((cax100, tax100, pax100), (cax1K, tax1K, pax1K), (cax10K, tax10K, pax10K)) = plt.subplots(3, 3, figsize=(25, 15))
plot_hist(cax100, final_costs[0], "Costs distribution of final optimisation with initial weight 1K", "cost")
plot_hist(cax1K, final_costs[1], "Costs distribution of final optimisation with initial weight 1M", "cost")
plot_hist(cax10K, final_costs[2], "Costs distribution of final optimisation with initial weight 1G", "cost")

plot_hist(tax100, final_opt_times[0], "Total optimistation time distribution of final optimisation with initial weight 1K", "time (s)")
plot_hist(tax1K, final_opt_times[1], "Total optimistation time distribution of final optimisation with initial weight 1M", "time (s)")
plot_hist(tax10K, final_opt_times[2], "Total optimistation time distribution of final optimisation with initial weight 1G", "time (s)")

plot_hist(pax100, final_phase_times[0], "Phase time distribution of final optimisation with initial weight 1K", "time (s)")
plot_hist(pax1K, final_phase_times[1], "Phase time distribution of final optimisation with initial weight 1M", "time (s)")
plot_hist(pax10K, final_phase_times[2], "Phase time distribution of final optimisation with initial weight 1G", "time (s)")
plt.tight_layout()
fig.suptitle("All 10000 initial max iterations, 500 shootings, converged only")
fig.savefig("../figures/final_varpo.pdf")

fig, ((cax100, tax100, pax100), (cax1K, tax1K, pax1K), (cax10K, tax10K, pax10K)) = plt.subplots(3, 3, figsize=(25, 15))
plot_hist(cax100, initial_costs[0], "Costs distribution of initial optimisation with initial weight 1K", "cost")
plot_hist(cax1K, initial_costs[1], "Costs distribution of initial optimisation with initial weight 1M", "cost")
plot_hist(cax10K, initial_costs[2], "Costs distribution of initial optimisation with initial weight 1G", "cost")

plot_hist(tax100, initial_opt_times[0], "Optimistation time distribution of initial optimisation with initial weight 1K", "time (s)")
plot_hist(tax1K, initial_opt_times[1], "Optimistation time distribution of initial optimisation with initial weight 1M", "time (s)")
plot_hist(tax10K, initial_opt_times[2], "Optimistation time distribution of initial optimisation with initial weight 1G", "time (s)")

plot_hist(pax100, initial_phase_times[0], "Phase time distribution of initial optimisation with initial weight 1K", "time (s)")
plot_hist(pax1K, initial_phase_times[1], "Phase time distribution of initial optimisation with initial weight 1M", "time (s)")
plot_hist(pax10K, initial_phase_times[2], "Phase time distribution of initial optimisation with initial weight 1G", "time (s)")
plt.tight_layout()
fig.suptitle("All 10000 initial max iterations, 500 shootings, converged only")
fig.savefig("../figures/initial_varpo.pdf")



#########   Other   #########

objfin = df(type="objective", var="varopt", phase="final")
mincost = min(objfin, key=extract_cost)
objfin.index(mincost)
mincost.case
mincostinit = next(filter(lambda s: s.case == mincost.case, df(type="objective", var="varopt", phase="initial")))


graph_convergence(df, m, nb_shooting)

# import bioviz
# viz = bioviz.Viz("../models/pendulum_maze.bioMod", show_floor=False)
# viz.load_movement(mincostinit.states["q"])
# viz.exec()











