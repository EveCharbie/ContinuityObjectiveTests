
import os
import pickle
import biorbd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython import embed

def compute_cost(states, controls, time):
    dt = time / 500
    lagrange_min_controls = np.sum(1 * controls**2 * dt)
    mayer_min_time = 100 * time
    return lagrange_min_controls + mayer_min_time

def extract_data_sol(m, filename, data_sol):
    
    # 0: type_ 
    # 1: case
    # 2: phase

    # 3: weight
    # 4: iter_
    # 5: weight_sphere
    # 6: time (time of the trial)
    # 7: cost
    # 8: time_opt (time spent in the solver)
    # 9: status
    # 10: transpersion
    
    filename = filename.split("-")

    states = data_sol.states['all']
    controls = data_sol.controls['all']
    time = data_sol.parameters['time'][0][0]
    time_opt = data_sol.real_time_to_optimize
    status = data_sol.status
    cost = compute_cost(states, controls, time)

    transpersion = compute_transpersion(m, states, controls, time, nb_shooting)

    if filename[0] == "unconstrained":
        type_, phase, case, weight, iter_, weight_sphere = filename[0], filename[1], filename[2], filename[3], filename[4], filename[5]
        case = int(case)
        return np.array([type_, case, phase, weight, iter_, weight_sphere, time, cost, time_opt, status, transpersion]), states, controls

    elif filename[0] == "objective_sphere":
        type_, phase, case, weight, iter_ = filename[0], filename[1], filename[2], filename[3], filename[4]
        weight_sphere = None
        case = int(case)
        return np.array([type_, case, phase, weight, iter_, weight_sphere, time, cost, time_opt, status, transpersion]), states, controls

    elif filename[0] == "objective_continuity":
        type_, phase, case, weight_sphere, iter_ = filename[0], filename[1], filename[2], filename[3], filename[4]
        weight = None
        case = int(case)
        return np.array([type_, case, phase, weight, iter_, weight_sphere, time, cost, time_opt, status, transpersion]), states, controls

    if filename[0] == "constraint":
        type_, case = filename[0], filename[1]
        phase, weight, iter_, weight_sphere = None, None, None, None
        case = int(case)
        return np.array([type_, case, phase, weight, iter_, weight_sphere, time, cost, time_opt, status, transpersion]), states, controls

    else:
        raise Exception(filename)


def analyse(extractor, solutions):
    vals = np.array([*map(extractor, solutions)])
    vals.shape = vals.size
    val_avg = np.mean(vals)
    val_std = np.std(vals)
    return vals, val_avg, val_std


def plot_hist(ax, dist, title, xlabel, bins="rice"):
    _, bins, _ = ax.hist(dist, bins=bins)
    ax.set_xticks(bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def compute_transpersion(m, x, u, t, nb_shooting):

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

    x_sub_interval = np.zeros((4, nb_sub_intervals * nb_shooting + 1))
    x_sub_interval[:, 0] = x[:, 0]
    for i in range(nb_shooting):
        x_sub_interval[:, nb_sub_intervals * i + 1 : nb_sub_intervals * (i+1) + 1] = runge_kutta_4(m, x[:, i], u[:, i], t, nb_shooting, nb_sub_intervals)[:, 1:]

    transpersion = np.zeros((nb_sub_intervals * nb_shooting + 1))
    for i in range(nb_sub_intervals * nb_shooting + 1):
        transpersion[i] = transpersion_dist_sum(x_sub_interval[:, i])

    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(x_sub_interval[0, :], '-b')
    # ax[1].plot(x_sub_interval[1, :], '-b')
    # ax[0].plot(range(0, nb_shooting*nb_sub_intervals+1, nb_sub_intervals), x[0, :], 'ok')
    # ax[1].plot(range(0, nb_shooting*nb_sub_intervals+1, nb_sub_intervals), x[1, :], 'ok')
    # ax[2].plot(transpersion)
    # ax[2].plot(range(0, nb_shooting*nb_sub_intervals+1, nb_sub_intervals), transpersion[range(0, nb_shooting*nb_sub_intervals+1, nb_sub_intervals)], 'ok')
    # plt.show()

    return np.sum(transpersion)


def graph_convergence(properties_constrained_converged, 
                      properties_objective_sphere_final_converged, 
                      properties_objective_continuity_final_converged, 
                      properties_unconstrained_converged,
                      min_transpersion,
                      max_transpersion):

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.ravel()
    plt_0 = ax[0].scatter(properties_constrained_converged[:, 8], properties_constrained_converged[:, 9], c=properties_constrained_converged[:, 11], vmin=min_transpersion, vmax=max_transpersion, marker='.', label='Constrained')
    plt_1 = ax[1].scatter(properties_objective_sphere_final_converged[:, 8], properties_objective_sphere_final_converged[:, 9], c=properties_objective_sphere_final_converged[:, 11], vmin=min_transpersion, vmax=max_transpersion, marker='.', label='Objective')
    plt_2 = ax[2].scatter(properties_objective_continuity_final_converged[:, 8], properties_objective_continuity_final_converged[:, 9], c=properties_objective_continuity_final_converged[:, 11], vmin=min_transpersion, vmax=max_transpersion, marker='.')
    plt_3 = ax[3].scatter(properties_unconstrained_converged[:, 8], properties_unconstrained_converged[:, 9], c=properties_unconstrained_converged[:, 11], vmin=min_transpersion, vmax=max_transpersion, marker='.', label='Unconstrained')
    ax[0].errorbar(np.mean(properties_constrained_converged[:, 8]), np.mean(properties_constrained_converged[:, 9]),
                   xerr=np.std(properties_constrained_converged[:, 8]), yerr=np.std(properties_constrained_converged[:, 9]),
                   color=np.mean(properties_constrained_converged[:, 11]))
    ax[1].errorbar(np.mean(properties_objective_sphere_final_converged[:, 8]), np.mean(properties_objective_sphere_final_converged[:, 9]),
                   xerr=np.std(properties_objective_sphere_final_converged[:, 8]), yerr=np.std(properties_objective_sphere_final_converged[:, 9]),
                   color=np.mean(properties_objective_sphere_final_converged[:, 11]))
    ax[2].errorbar(np.mean(properties_objective_continuity_final_converged[:, 8]), np.mean(properties_objective_continuity_final_converged[:, 9]),
                   xerr=np.std(properties_objective_continuity_final_converged[:, 8]), yerr=np.std(properties_objective_continuity_final_converged[:, 9]),
                   color=np.mean(properties_objective_continuity_final_converged[:, 11]))
    ax[3].errorbar(np.mean(properties_unconstrained_converged[:, 8]), np.mean(properties_unconstrained_converged[:, 9]),
                   xerr=np.std(properties_unconstrained_converged[:, 8]), yerr=np.std(properties_unconstrained_converged[:, 9]),
                   color=np.mean(properties_unconstrained_converged[:, 11]))
    ax[0].set_title("Constrained")
    ax[1].set_title("Objective sphere")
    ax[2].set_title("Objective continuity")
    ax[3].set_title("Unconstrained")

    plt.xlabel("Cost")
    plt.ylabel("time to optimize")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False)
    cbar = plt.colorbar(plt_0, ax=ax) # format=ticker.FuncFormatter(fmt)
    cbar.set_label('Transpersion sum')
    plt.savefig("../figures/convergence_info_graph.png", dpi=300)
    # plt.show()
    return



def graph_kinmatics(properties_constrained_converged, 
                    properties_objective_sphere_final_converged, 
                    properties_objective_continuity_final_converged, 
                    properties_unconstrained_converged,
                    states_constrained_converged,
                    states_objective_sphere_final_converged,
                    states_objective_continuity_final_converged,
                    states_unconstrained_converged,    
                    max_cost, 
                    min_cost,
                    nb_shooting):

    cmap = cm.get_cmap('viridis')

    def plot_lines(key_word, linestyle, properti, state, nb_shooting, cmap, ax):

        for i in range(np.shape(properti)[0]):
            color = cmap(properti[i, 8] / (max_cost - min_cost))
            time_vector = np.linspace(0, properti[i, 7], nb_shooting+1)
            if i == 0:
                ax[0].plot(time_vector, state[0, :, i],
                           color=color, linestyle=linestyle, label=key_word)
            else:
                ax[0].plot(time_vector, state[0, :, i],
                           color=color, linestyle=linestyle)
                ax[1].plot(time_vector, state[1, :, i],
                           color=color, linestyle=linestyle)
        return

    fig, ax = plt.subplots(2, 1)
    fig.subplots_adjust(top=0.8, right=0.75)

    plot_lines('Constraint', (0, (1, 1)), properties_constrained_converged, states_constrained_converged, nb_shooting, cmap, ax)
    plot_lines('Objective (sphere)', (0, (5, 1)), properties_objective_sphere_final_converged, states_objective_sphere_final_converged, nb_shooting, cmap, ax)
    plot_lines('Objective (continuity)', (0, (5, 1)), properties_objective_continuity_final_converged, states_objective_continuity_final_converged, nb_shooting, cmap, ax)
    plot_lines('Unconstrained', 'solid', properties_unconstrained_converged, states_unconstrained_converged, nb_shooting, cmap, ax)

    ax[1].set_xlabel("Time")
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, frameon=False)
    ax[0].tick_params(axis='x', bottom=False, labelbottom=False)
    fake_plot_for_color = ax[0].scatter(np.array([0]), np.array([0]), marker='.', c=np.array([min_cost]), vmin=min_cost, vmax=max_cost)
    cbar_ax = fig.add_axes([0.8, 0.11, 0.03, 0.7])
    cbar = plt.colorbar(fake_plot_for_color, cax=cbar_ax)
    cbar.set_label('Cost')
    # cbar.ax.set_title('This i')
    plt.savefig("../figures/kinematics_graph.png", dpi=300)
    # plt.show()


    return

#########   Loading data   #########

MEAN_FLAG = False
HISTOGRAM_FLAG = False

# directory = "../solutions/mini_folder"
# directory = "../solutions/small_folder"
directory = "../solutions_IPOPT/small_folder"

nb_shooting = 500
m = biorbd.Model("../models/pendulum_maze.bioMod")


# loop over the files in the folder to load them
properties_all = []
states_all = []
for filename in os.listdir(directory):
    if filename[-7:] == ".pickle":
        with open(f"{directory}/{filename}", "rb") as f:
            data_sol = pickle.load(f)
            properties, states, _ = extract_data_sol(m, filename, data_sol)
            if np.shape(properties_all) == (0, ):
                properties_all = properties
                states_all = states
            else:
                properties_all = np.vstack((properties_all, properties))
                states_all = np.dstack((states_all, states))
            print(f"Loaded successfully {filename}")

embed()
# Sort the files loaded + print convergence rate
properties_constrained = properties_all[np.where(properties_all[:, 0] == "constraint"), :][0]
states_constrained = states_all[:, :, np.where(properties_all[:, 0] == "constraint")][:, :, 0, :]
constrained_index_converged = np.where(properties_constrained[:, 9] == 0)
properties_constrained_converged = properties_constrained[constrained_index_converged, :][0]
states_constrained_converged = states_constrained[:, :, constrained_index_converged][:, :, 0, :]
constrained_convergence_rate = len(constrained_index_converged[0]) / len(properties_constrained) * 100
print("Convergence rate fully constrained OCP : ", constrained_convergence_rate, "%")


properties_objective_sphere = properties_all[np.where(properties_all[:, 0] == "objective_sphere"), :][0]
states_objective_sphere = states_all[:, :, np.where(properties_all[:, 0] == "objective_sphere")][:, :, 0, :]
properties_objective_sphere_initial = properties_objective_sphere[np.where(properties_objective_sphere[:, 2] == "initial"), :][0]
states_objective_sphere_initial = states_objective_sphere[:, :, np.where(properties_objective_sphere[:, 2] == "initial")][:, :, 0, :]
properties_objective_sphere_final = properties_objective_sphere[np.where(properties_objective_sphere[:, 2] == "final"), :][0]
states_objective_sphere_final = states_objective_sphere[:, :, np.where(properties_objective_sphere[:, 2] == "final")][:, :, 0, :]
objective_sphere_initial_index_converged = np.where(properties_objective_sphere_initial[:, 9] == 0)
properties_objective_sphere_initial_converged = properties_objective_sphere_initial[objective_sphere_initial_index_converged, :][0]
states_objective_sphere_initial_converged = states_objective_sphere_initial[:, :, objective_sphere_initial_index_converged][:, :, 0, :]
objective_sphere_initial_convergence_rate = len(objective_sphere_initial_index_converged[0]) / len(properties_objective_sphere_initial) * 100
print("Convergence rate objective_sphere->constraint OCP initial step : ", objective_sphere_initial_convergence_rate, "%")
objective_sphere_final_index_converged = np.where(properties_objective_sphere_final[:, 9] == 0)
properties_objective_sphere_final_converged = properties_objective_sphere_final[objective_sphere_final_index_converged, :][0]
states_objective_sphere_final_converged = states_objective_sphere_final[:, :, objective_sphere_final_index_converged][:, :, 0, :]
objective_sphere_final_convergence_rate = len(objective_sphere_final_index_converged[0]) / len(properties_objective_sphere_final) * 100
print("Convergence rate objective_sphere->constraint OCP final step : ", objective_sphere_final_convergence_rate, "%")


properties_objective_continuity = properties_all[np.where(properties_all[:, 0] == "objective_continuity"), :][0]
states_objective_continuity = states_all[:, :, np.where(properties_all[:, 0] == "objective_continuity")][:, :, 0, :]
properties_objective_continuity_initial = properties_objective_continuity[np.where(properties_objective_continuity[:, 2] == "initial"), :][0]
states_objective_continuity_initial = states_objective_continuity[:, :, np.where(properties_objective_continuity[:, 2] == "initial")][:, :, 0, :]
properties_objective_continuity_final = properties_objective_continuity[np.where(properties_objective_continuity[:, 2] == "final"), :][0]
states_objective_continuity_final = states_objective_continuity[:, :, np.where(properties_objective_continuity[:, 2] == "final")][:, :, 0, :]
objective_continuity_initial_index_converged = np.where(properties_objective_continuity_initial[:, 9] == 0)
properties_objective_continuity_initial_converged = properties_objective_continuity_initial[objective_continuity_initial_index_converged, :][0]
states_objective_continuity_initial_converged = states_objective_continuity_initial[:, :, objective_continuity_initial_index_converged][:, :, 0, :]
objective_continuity_initial_convergence_rate = len(objective_continuity_initial_index_converged[0]) / len(properties_objective_continuity_initial) * 100
print("Convergence rate objective_continuity->constraint OCP initial step : ", objective_continuity_initial_convergence_rate, "%")
objective_continuity_final_index_converged = np.where(properties_objective_continuity_final[:, 9] == 0)
properties_objective_continuity_final_converged = properties_objective_continuity_final[objective_continuity_final_index_converged, :][0]
states_objective_continuity_final_converged = states_objective_continuity_final[:, :, objective_continuity_final_index_converged][:, :, 0, :]
objective_continuity_final_convergence_rate = len(objective_continuity_final_index_converged[0]) / len(properties_objective_continuity_final) * 100
print("Convergence rate objective_continuity->constraint OCP final step : ", objective_continuity_final_convergence_rate, "%")


properties_unconstrained = properties_all[np.where(properties_all[:, 0] == "unconstrained"), :][0]
states_unconstrained = states_all[:, :, np.where(properties_all[:, 0] == "unconstrained")][:, :, 0, :]
properties_unconstrained_initial = properties_unconstrained[np.where(properties_unconstrained[:, 2] == "initial"), :][0]
states_unconstrained_initial = states_unconstrained[:, :, np.where(properties_unconstrained[:, 2] == "initial")][:, :, 0, :]
properties_unconstrained_final = properties_unconstrained[np.where(properties_unconstrained[:, 2] == "final"), :][0]
states_unconstrained_final = states_unconstrained[:, :, np.where(properties_unconstrained[:, 2] == "final")][:, :, 0, :]
objective_continuity_initial_index_converged = np.where(properties_unconstrained_initial[:, 9] == 0)
properties_unconstrained_initial_converged = properties_unconstrained_initial[objective_continuity_initial_index_converged, :][0]
states_unconstrained_initial_converged = states_unconstrained_initial[:, :, objective_continuity_initial_index_converged][:, :, 0, :]
objective_continuity_initial_convergence_rate = len(objective_continuity_initial_index_converged[0]) / len(properties_unconstrained_initial) * 100
print("Convergence rate unconstrained->constraint OCP initial step : ", objective_continuity_initial_convergence_rate, "%")
objective_continuity_final_index_converged = np.where(properties_unconstrained_final[:, 9] == 0)
properties_unconstrained_final_converged = properties_unconstrained_final[objective_continuity_final_index_converged, :][0]
states_unconstrained_final_converged = states_unconstrained_final[:, :, objective_continuity_final_index_converged][:, :, 0, :]
objective_continuity_final_convergence_rate = len(objective_continuity_final_index_converged[0]) / len(properties_unconstrained_final) * 100
print("Convergence rate unconstrained->constraint OCP final step : ", objective_continuity_final_convergence_rate, "%")


properties_all_converged = properties_all[np.where(properties_all[:, 9] == 0), :][0]

max_cost = np.max(properties_all_converged[:, 7])
min_cost = np.min(properties_all_converged[:, 7])
max_time_to_optimize = np.max(properties_all_converged[:, 8])
min_time_to_optimize = np.min(properties_all_converged[:, 8])
max_transpersion = np.max(properties_all_converged[:, 10])
min_transpersion = np.min(properties_all_converged[:, 10])

cost_90th_percentile = np.percentile(properties_all_converged[:, 7], 90)
idx_90_constrainted = np.where(properties_constrained_converged[:, 7] < cost_90th_percentile)
idx_90_sphere_final = np.where(properties_objective_sphere_final_converged[:, 7] < cost_90th_percentile)
idx_90_continuity_final = np.where(properties_unconstrained_final_converged[:, 7] < cost_90th_percentile)
idx_90_unconstrainted_final = np.where(properties_unconstrained_final_converged[:, 7] < cost_90th_percentile)
pourcentage_constrainted = len(idx_90_constrainted) / np.shape(properties_constrained_converged)[0] * 100
pourcentage_sphere_final = len(idx_90_sphere_final) / np.shape(properties_objective_sphere_final_converged)[0] * 100
pourcentage_continuity_final = len(idx_90_continuity_final) / np.shape(properties_unconstrained_final_converged)[0] * 100
pourcentage_unconstrainted_final = len(idx_90_unconstrainted) / np.shape(properties_unconstrained_converged)[0] * 100
print(f"{pourcentage_constrainted} % of the constrained solutions were below the 90th percentile for the cost function value")
print(f"{pourcentage_sphere_final} % of the objective_sphere solutions were below the 90th percentile for the cost function value")
print(f"{pourcentage_continuity_final} % of the objective_continuity solutions were below the 90th percentile for the cost function value")
print(f"{pourcentage_unconstrainted_final} % of the unconstrained solutions were below the 90th percentile for the cost function value")

graph_convergence(properties_constrained_converged,
                  properties_objective_sphere_final_converged, 
                  properties_objective_continuity_final_converged, 
                  properties_unconstrained_converged,
                  min_transpersion,
                  max_transpersion)

graph_kinmatics(properties_constrained_converged[idx_90_constrainted, :],
                properties_objective_sphere_final_converged[idx_90_sphere_final, :],
                properties_objective_continuity_final_converged[idx_90_continuity_final, :],
                properties_unconstrained_converged[idx_90_unconstrainted, :],
                states_constrained_converged[idx_90_constrainted, :],
                states_objective_sphere_final_converged[idx_90_sphere_final, :],
                states_objective_continuity_final_converged[idx_90_continuity_final, :],
                states_unconstrained_converged[idx_90_unconstrainted, :],
                max_cost,
                min_cost,
                nb_shooting)


# if HISTOGRAM_FLAG:
#     fig, (cax, tax, pax) = plt.subplots(1, 3, figsize=(25, 5))
#     plot_hist(cax, costs, "Costs distribution", "cost")
#     plot_hist(tax, opt_times, "Optimization times distribution", "time (s)")
#     plot_hist(pax, phase_times, "Phase times distribution", "time (s)")
#     plt.tight_layout()
#     fig.suptitle("All 10000 max iterations, 500 shootings, converged only")
#     fig.savefig("../figures/constraint.pdf")

# if HISTOGRAM_FLAG:
#     fig, ((cax100, tax100, pax100), (cax1K, tax1K, pax1K), (cax10K, tax10K, pax10K)) = plt.subplots(3, 3, figsize=(25, 15))
#     plot_hist(cax100, final_costs[0], "Costs distribution of final optimisation with 100 initial iterations", "cost")
#     plot_hist(cax1K, final_costs[1], "Costs distribution of final optimisation with 1000 initial iterations", "cost")
#     plot_hist(cax10K, final_costs[2], "Costs distribution of final optimisation with 10000 initial iterations", "cost")
#
#     plot_hist(tax100, final_opt_times[0], "Total optimistation time distribution of final optimisation with 100 initial iterations", "time (s)")
#     plot_hist(tax1K, final_opt_times[1], "Total optimistation time distribution of final optimisation with 1000 initial iterations", "time (s)")
#     plot_hist(tax10K, final_opt_times[2], "Total optimistation time distribution of final optimisation with 10000 initial iterations", "time (s)")
#
#     plot_hist(pax100, final_phase_times[0], "Phase time distribution of final optimisation with 100 initial iterations", "time (s)")
#     plot_hist(pax1K, final_phase_times[1], "Phase time distribution of final optimisation with 1000 initial iterations", "time (s)")
#     plot_hist(pax10K, final_phase_times[2], "Phase time distribution of final optimisation with 10000 initial iterations", "time (s)")
#     plt.tight_layout()
#     fig.suptitle("All weight 1M, 500 shootings, converged only")
#     fig.savefig("../figures/final_continuity.pdf")
#
#     fig, ((cax100, tax100, pax100), (cax1K, tax1K, pax1K), (cax10K, tax10K, pax10K)) = plt.subplots(3, 3, figsize=(25, 15))
#     plot_hist(cax100, initial_costs[0], "Costs distribution of initial optimisation with 100 initial iterations", "cost")
#     plot_hist(cax1K, initial_costs[1], "Costs distribution of initial optimisation with 1000 initial iterations", "cost")
#     plot_hist(cax10K, initial_costs[2], "Costs distribution of initial optimisation with 10000 initial iterations", "cost")
#
#     plot_hist(tax100, initial_opt_times[0], "Optimistation time distribution of initial optimisation with 100 initial iterations", "time (s)")
#     plot_hist(tax1K, initial_opt_times[1], "Optimistation time distribution of initial optimisation with 1000 initial iterations", "time (s)")
#     plot_hist(tax10K, initial_opt_times[2], "Optimistation time distribution of initial optimisation with 10000 initial iterations", "time (s)")
#
#     plot_hist(pax100, initial_phase_times[0], "Phase time distribution of initial optimisation with 100 initial iterations", "time (s)")
#     plot_hist(pax1K, initial_phase_times[1], "Phase time distribution of initial optimisation with 1000 initial iterations", "time (s)")
#     plot_hist(pax10K, initial_phase_times[2], "Phase time distribution of initial optimisation with 10000 initial iterations", "time (s)")
#     plt.tight_layout()
#     fig.suptitle("All weight 1M, 500 shootings, converged only")
#     fig.savefig("../figures/initial_continuity.pdf")
#
#
#
# #########   Other   #########
#
# # objfin = df(type="objective", var="varopt", phase="final")
# # mincost = min(objfin, key=extract_cost)
# # objfin.index(mincost)
# # mincost.case
# # mincostinit = next(filter(lambda s: s.case == mincost.case, df(type="objective", var="varopt", phase="initial")))
#

# import bioviz
# viz = bioviz.Viz("models/pendulum_maze.bioMod", show_floor=False)
# viz.load_movement(mincostinit.states["q"])
# viz.exec()











