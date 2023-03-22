
import os
import pickle
import biorbd
import bioviz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from IPython import embed

def compute_cost(states, controls, time):
    dt = time / 500
    lagrange_min_controls = np.nansum(1 * controls**2 * dt)
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
    # 11: iterations
    # 12: mean time per iteration
    
    filename = filename.split("-")

    states = data_sol.states['all']
    controls = data_sol.controls['all']
    time = data_sol.parameters['time'][0][0]
    status = data_sol.status
    cost = compute_cost(states, controls, time)
    iterations = data_sol.iterations
    time_per_iteration = data_sol.real_time_to_optimize / iterations

    transpersion = compute_max_transpersion(m, states, controls, time, nb_shooting)


    if filename[0] == "objective_sphere":
        type_, phase, case, weight_sphere, iter_ = "continuity_in_objective", filename[1], filename[2], filename[3], filename[4]
        weight = None
        case = int(case)
        if phase == 'initial':
            time_opt = data_sol.real_time_to_optimize
        else:
            time_opt = data_sol.total_time
        return np.array([type_, case, phase, weight, iter_, weight_sphere, time, cost, time_opt, status, transpersion, iterations, time_per_iteration]), states, controls

    elif filename[0] == "objective_continuity":
        type_, phase, case, weight, iter_ = "spheres_in_objective", filename[1], filename[2], filename[3], filename[4]
        weight_sphere = None
        case = int(case)
        if phase == 'initial':
            time_opt = data_sol.real_time_to_optimize
        else:
            time_opt = data_sol.total_time
        return np.array([type_, case, phase, weight, iter_, weight_sphere, time, cost, time_opt, status, transpersion, iterations, time_per_iteration]), states, controls

    elif filename[0] == "constraint":
        type_, case = filename[0], filename[1]
        phase, weight, iter_, weight_sphere = None, None, None, None
        case = int(case)
        time_opt = data_sol.total_time
        return np.array([type_, case, phase, weight, iter_, weight_sphere, time, cost, time_opt, status, transpersion, iterations, time_per_iteration]), states, controls

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


def compute_max_transpersion(m, x, u, t, nb_shooting):

    def transpersion_dist_max(x):

        shpere_list = [(0, 0.05, 0), (0, 0.75, 0.2), (0, -0.95, 0), (0, 1.4, 0.5)]
        bound = 0.35
        marker = m.marker(biorbd.GeneralizedCoordinates(x[:m.nbQ()]), 1).to_array()

        max_trans = 0
        for i in range(len(shpere_list)):
            distance = np.linalg.norm(np.array(shpere_list[i]) - marker) - bound
            if distance < 0 and np.abs(distance) > max_trans:
                max_trans = np.abs(distance)

        return max_trans

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

    max_trans = 0
    for i in range(nb_sub_intervals * nb_shooting + 1):
        if transpersion_dist_max(x_sub_interval[:, i]) > max_trans:
            max_trans = transpersion_dist_max(x_sub_interval[:, i])

    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(x_sub_interval[0, :], '-b')
    # ax[1].plot(x_sub_interval[1, :], '-b')
    # ax[0].plot(range(0, nb_shooting*nb_sub_intervals+1, nb_sub_intervals), x[0, :], 'ok')
    # ax[1].plot(range(0, nb_shooting*nb_sub_intervals+1, nb_sub_intervals), x[1, :], 'ok')
    # ax[2].plot(transpersion)
    # ax[2].plot(range(0, nb_shooting*nb_sub_intervals+1, nb_sub_intervals), transpersion[range(0, nb_shooting*nb_sub_intervals+1, nb_sub_intervals)], 'ok')
    # plt.show()

    return max_trans

def graph_convergence(properties_constrained_converged,
                      properties_objective_sphere_final_converged,
                      properties_objective_continuity_final_converged,
                      max_cost,
                      max_time_to_optimize,
                      max_transpersion):

    FLAG_VERTCIAL_FIG = False

    cmap = cm.get_cmap('viridis')

    if FLAG_VERTCIAL_FIG:
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
        ax = ax.ravel()
        plt_0 = ax[0].scatter(properties_constrained_converged[:, 7], properties_constrained_converged[:, 8], c=properties_constrained_converged[:, 10], vmin=0, vmax=max_transpersion, marker='.', cmap='viridis')
        plt_1 = ax[1].scatter(properties_objective_continuity_final_converged[:, 7], properties_objective_continuity_final_converged[:, 8], c=properties_objective_continuity_final_converged[:, 10], vmin=0, vmax=max_transpersion, marker='.', cmap='viridis')
        plt_2 = ax[2].scatter(properties_objective_sphere_final_converged[:, 7], properties_objective_sphere_final_converged[:, 8], c=properties_objective_sphere_final_converged[:, 10], vmin=0, vmax=max_transpersion, marker='.', cmap='viridis')

        ax[0].errorbar(np.mean(properties_constrained_converged[:, 7]), np.mean(properties_constrained_converged[:, 8]),
                       xerr=np.std(properties_constrained_converged[:, 7]), yerr=np.std(properties_constrained_converged[:, 8]),
                       color=cmap(np.mean(properties_constrained_converged[:, 10]) / max_transpersion), capsize=3, capthick=1)
        ax[1].errorbar(np.mean(properties_objective_continuity_final_converged[:, 7]), np.mean(properties_objective_continuity_final_converged[:, 8]),
                       xerr=np.std(properties_objective_continuity_final_converged[:, 7]), yerr=np.std(properties_objective_continuity_final_converged[:, 8]),
                       color=cmap(np.mean(properties_objective_continuity_final_converged[:, 10]) / max_transpersion), capsize=3, capthick=1)
        ax[2].errorbar(np.mean(properties_objective_sphere_final_converged[:, 7]), np.mean(properties_objective_sphere_final_converged[:, 8]),
                       xerr=np.std(properties_objective_sphere_final_converged[:, 7]), yerr=np.std(properties_objective_sphere_final_converged[:, 8]),
                       color=cmap(np.mean(properties_objective_sphere_final_converged[:, 10]) / max_transpersion), capsize=3, capthick=1)

        ax[0].set_title('Constrained', fontsize=12, fontweight='bold', style='italic')
        ax[1].set_title('Penalized continuity', fontsize=12, fontweight='bold', style='italic')
        ax[2].set_title('Penalized obstacles', fontsize=12, fontweight='bold', style='italic')

        ax[2].set_xlabel('Cost', fontsize=12)
        ax[0].set_ylabel('Computational time [s]', fontsize=12)
        ax[1].set_ylabel('Computational time [s]', fontsize=12)
        ax[2].set_ylabel('Computational time [s]', fontsize=12)

        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')

        ax[0].set_xlim((80, max_cost))
        ax[1].set_xlim((80, max_cost))
        ax[2].set_xlim((80, max_cost))
        ax[0].set_ylim((1, max_time_to_optimize))
        ax[1].set_ylim((1, max_time_to_optimize))
        ax[2].set_ylim((1, max_time_to_optimize))

        ax[0].plot(np.array([194.74616649275964, 194.74616649275964]), np.array([1, max_time_to_optimize]), '--k', alpha=0.5, linewidth=0.8)
        ax[1].plot(np.array([194.74616649275964, 194.74616649275964]), np.array([1, max_time_to_optimize]), '--k', alpha=0.5, linewidth=0.8)
        ax[2].plot(np.array([194.74616649275964, 194.74616649275964]), np.array([1, max_time_to_optimize]), '--k', alpha=0.5, linewidth=0.8)

        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.2, right=0.8)
        cbar_ax = fig.add_axes([0.84, 0.1, 0.03, 0.8])
        cbar = plt.colorbar(plt_0, cax=cbar_ax)
        cbar.ax.set_title('Max penetration\n[m]', fontsize=12)
        plt.savefig("../figures/convergence_info_graph.png", dpi=300)
        # plt.show()

    else:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
        ax = ax.ravel()
        plt_0 = ax[0].scatter(properties_constrained_converged[:, 7], properties_constrained_converged[:, 8], c=properties_constrained_converged[:, 10], vmin=0, vmax=max_transpersion, marker='.', cmap='viridis')
        plt_1 = ax[1].scatter(properties_objective_continuity_final_converged[:, 7], properties_objective_continuity_final_converged[:, 8], c=properties_objective_continuity_final_converged[:, 10], vmin=0, vmax=max_transpersion, marker='.', cmap='viridis')
        plt_2 = ax[2].scatter(properties_objective_sphere_final_converged[:, 7], properties_objective_sphere_final_converged[:, 8], c=properties_objective_sphere_final_converged[:, 10], vmin=0, vmax=max_transpersion, marker='.', cmap='viridis')

        ax[0].errorbar(np.mean(properties_constrained_converged[:, 7]), np.mean(properties_constrained_converged[:, 8]),
                       xerr=np.std(properties_constrained_converged[:, 7]), yerr=np.std(properties_constrained_converged[:, 8]),
                       color=cmap(np.mean(properties_constrained_converged[:, 10]) / max_transpersion), capsize=3, capthick=1)
        ax[1].errorbar(np.mean(properties_objective_continuity_final_converged[:, 7]), np.mean(properties_objective_continuity_final_converged[:, 8]),
                       xerr=np.std(properties_objective_continuity_final_converged[:, 7]), yerr=np.std(properties_objective_continuity_final_converged[:, 8]),
                       color=cmap(np.mean(properties_objective_continuity_final_converged[:, 10]) / max_transpersion), capsize=3, capthick=1)
        ax[2].errorbar(np.mean(properties_objective_sphere_final_converged[:, 7]), np.mean(properties_objective_sphere_final_converged[:, 8]),
                       xerr=np.std(properties_objective_sphere_final_converged[:, 7]), yerr=np.std(properties_objective_sphere_final_converged[:, 8]),
                       color=cmap(np.mean(properties_objective_sphere_final_converged[:, 10]) / max_transpersion), capsize=3, capthick=1)

        ax[0].set_title('Constrained', fontsize=12, fontweight='bold', style='italic')
        ax[1].set_title('Penalized continuity', fontsize=12, fontweight='bold', style='italic')
        ax[2].set_title('Penalized obstacles', fontsize=12, fontweight='bold', style='italic')

        ax[0].set_xlabel('Cost', fontsize=12)
        ax[1].set_xlabel('Cost', fontsize=12)
        ax[2].set_xlabel('Cost', fontsize=12)
        ax[0].set_ylabel('Computational time [s]', fontsize=12)

        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')

        ax[0].set_xlim((80, max_cost))
        ax[1].set_xlim((80, max_cost))
        ax[2].set_xlim((80, max_cost))
        ax[0].set_ylim((1, max_time_to_optimize))
        ax[1].set_ylim((1, max_time_to_optimize))
        ax[2].set_ylim((1, max_time_to_optimize))

        ax[0].plot(np.array([194.74616649275964, 194.74616649275964]), np.array([1, max_time_to_optimize]), '--k', alpha=0.5, linewidth=0.8)
        ax[1].plot(np.array([194.74616649275964, 194.74616649275964]), np.array([1, max_time_to_optimize]), '--k', alpha=0.5, linewidth=0.8)
        ax[2].plot(np.array([194.74616649275964, 194.74616649275964]), np.array([1, max_time_to_optimize]), '--k', alpha=0.5, linewidth=0.8)

        plt.subplots_adjust(bottom=0.15, top=0.85, left=0.1, right=0.85, wspace=0.2)
        cbar_ax = fig.add_axes([0.89, 0.15, 0.03, 0.7])
        cbar = plt.colorbar(plt_0, cax=cbar_ax)
        cbar.ax.set_title('Max penetration\n[m]', fontsize=12)
        plt.savefig("../figures/convergence_info_graph_horz.png", dpi=300)
        # plt.show()

    return


def graph_convergence_side(properties_constrained_converged,
                      properties_objective_sphere_final_converged,
                      properties_objective_continuity_final_converged,
                      max_cost,
                      max_time_to_optimize,
                      max_transpersion):

    cmap = cm.get_cmap('viridis')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    ax = ax.ravel()
    plt_0 = ax[0].scatter(properties_constrained_converged[:, 7], properties_constrained_converged[:, 8], c=properties_constrained_converged[:, 10], vmin=0, vmax=max_transpersion, marker='.', cmap='viridis')
    plt_1 = ax[1].scatter(properties_objective_continuity_final_converged[:, 7], properties_objective_continuity_final_converged[:, 8], c=properties_objective_continuity_final_converged[:, 10], vmin=0, vmax=max_transpersion, marker='.', cmap='viridis')
    plt_2 = ax[2].scatter(properties_objective_sphere_final_converged[:, 7], properties_objective_sphere_final_converged[:, 8], c=properties_objective_sphere_final_converged[:, 10], vmin=0, vmax=max_transpersion, marker='.', cmap='viridis')

    ax[0].errorbar(np.mean(properties_constrained_converged[:, 7]), np.mean(properties_constrained_converged[:, 8]),
                   xerr=np.std(properties_constrained_converged[:, 7]), yerr=np.std(properties_constrained_converged[:, 8]),
                   color=cmap(np.mean(properties_constrained_converged[:, 10]) / max_transpersion), capsize=3, capthick=1)
    ax[1].errorbar(np.mean(properties_objective_continuity_final_converged[:, 7]), np.mean(properties_objective_continuity_final_converged[:, 8]),
                   xerr=np.std(properties_objective_continuity_final_converged[:, 7]), yerr=np.std(properties_objective_continuity_final_converged[:, 8]),
                   color=cmap(np.mean(properties_objective_continuity_final_converged[:, 10]) / max_transpersion), capsize=3, capthick=1)
    ax[2].errorbar(np.mean(properties_objective_sphere_final_converged[:, 7]), np.mean(properties_objective_sphere_final_converged[:, 8]),
                   xerr=np.std(properties_objective_sphere_final_converged[:, 7]), yerr=np.std(properties_objective_sphere_final_converged[:, 8]),
                   color=cmap(np.mean(properties_objective_sphere_final_converged[:, 10]) / max_transpersion), capsize=3, capthick=1)

    ax[0].set_title('Constrained', fontsize=12, fontweight='bold', style='italic')
    ax[1].set_title('Penalized continuity', fontsize=12, fontweight='bold', style='italic')
    ax[2].set_title('Penalized obstacles', fontsize=12, fontweight='bold', style='italic')

    ax[2].set_xlabel('Cost', fontsize=12)
    ax[0].set_ylabel('Computational time [s]', fontsize=12)
    ax[1].set_ylabel('Computational time [s]', fontsize=12)
    ax[2].set_ylabel('Computational time [s]', fontsize=12)

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')

    ax[0].set_xlim((80, max_cost))
    ax[1].set_xlim((80, max_cost))
    ax[2].set_xlim((80, max_cost))
    ax[0].set_ylim((1, max_time_to_optimize))
    ax[1].set_ylim((1, max_time_to_optimize))
    ax[2].set_ylim((1, max_time_to_optimize))

    ax[0].plot(np.array([194.74616649275964, 194.74616649275964]), np.array([1, max_time_to_optimize]), '--k', alpha=0.5, linewidth=0.8)
    ax[1].plot(np.array([194.74616649275964, 194.74616649275964]), np.array([1, max_time_to_optimize]), '--k', alpha=0.5, linewidth=0.8)
    ax[2].plot(np.array([194.74616649275964, 194.74616649275964]), np.array([1, max_time_to_optimize]), '--k', alpha=0.5, linewidth=0.8)

    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.2, right=0.8)
    cbar_ax = fig.add_axes([0.84, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(plt_0, cax=cbar_ax)
    cbar.ax.set_title('Max penetration\n[m]', fontsize=12)
    plt.savefig("../figures/convergence_info_graph.png", dpi=300)
    # plt.show()
    return


def graph_kinmatics_each_good(properties_constrained_converged,
                    properties_objective_sphere_final_converged,
                    properties_objective_continuity_final_converged,
                    states_constrained_converged,
                    states_objective_sphere_final_converged,
                    states_objective_continuity_final_converged,
                    idx_clusters_Constrained,
                    idx_clusters_sphere_final,
                    idx_clusters_continuity_final,
                    nb_shooting):


    max_time = 0
    min_cost = 1000
    max_cost = 0
    for i in range(4):
        if idx_clusters_Constrained[i][0].shape != (0,):
            max_time_tempo = np.max(properties_constrained_converged[idx_clusters_Constrained[i][0], 6])
            min_cost_tempo = np.max(properties_constrained_converged[idx_clusters_Constrained[i][0], 7])
            max_cost_tempo = np.max(properties_constrained_converged[idx_clusters_Constrained[i][0], 7])
            if max_time_tempo > max_time:
                max_time = max_time_tempo
            if min_cost_tempo < min_cost:
                min_cost = min_cost_tempo
            if max_cost_tempo > max_cost:
                max_cost = max_cost_tempo
        if idx_clusters_sphere_final[i][0].shape != (0,):
            max_time_tempo = np.max(properties_objective_sphere_final_converged[idx_clusters_sphere_final[i][0], 6])
            min_cost_tempo = np.max(properties_objective_sphere_final_converged[idx_clusters_sphere_final[i][0], 7])
            max_cost_tempo = np.max(properties_objective_sphere_final_converged[idx_clusters_sphere_final[i][0], 7])
            if max_time_tempo > max_time:
                max_time = max_time_tempo
            if min_cost_tempo < min_cost:
                min_cost = min_cost_tempo
            if max_cost_tempo > max_cost:
                max_cost = max_cost_tempo
        if idx_clusters_continuity_final[i][0].shape != (0,):
            max_time_tempo = np.max(properties_objective_continuity_final_converged[idx_clusters_continuity_final[i][0], 6])
            min_cost_tempo = np.max(properties_objective_continuity_final_converged[idx_clusters_continuity_final[i][0], 7])
            max_cost_tempo = np.max(properties_objective_continuity_final_converged[idx_clusters_continuity_final[i][0], 7])
            if max_time_tempo > max_time:
                max_time = max_time_tempo
            if min_cost_tempo < min_cost:
                min_cost = min_cost_tempo
            if max_cost_tempo > max_cost:
                max_cost = max_cost_tempo

    cmap = cm.get_cmap('plasma')

    def plot_lines(key_word, linestyle, properti, state, nb_shooting, cmap, ax):

        duration = []
        for i in range(np.shape(properti)[0]):
            duration = properti[i, 6]
            time_vector = np.linspace(0, properti[i, 6], nb_shooting + 1)
            ax[0].scatter(time_vector, state[0, :, i], c=np.ones((501, ))*properti[0, 7], vmin=min_cost, vmax=max_cost,
                          s=0.1, cmap='plasma')
            ax[1].scatter(time_vector, state[1, :, i], c=np.ones((501, ))*properti[0, 7], vmin=min_cost, vmax=max_cost,
                          s=0.1, cmap='plasma')

        ax[0].set_xlim((0, max_time))
        ax[0].set_ylim((-4, 4))
        ax[1].set_xlim((0, max_time))
        ax[1].set_ylim((-5, 5))
        return duration

    fig, ax = plt.subplots(2, 3, figsize=(12.5, 4))

    # cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.73])
    # fake_plot_for_color = ax[0, 0].scatter(np.array([-1]), np.array([0]), marker='.', c=np.array([min_cost]),
    #                                        vmin=min_cost, vmax=max_cost, cmap='plasma')
    # cbar = plt.colorbar(fake_plot_for_color, cax=cbar_ax, cmap='plasma')
    # cbar.ax.set_title('Cost', fontsize=12)

    plt.scatter(np.array([0]), np.array([0]), vmin=min_cost, vmax=max_cost, c=np.array([properties_objective_sphere_final_converged[idx_clusters_sphere_final[0][0][0], 7]]), label=u"Cluster #1 \u2014 " + f"{round(mean_cluster[0], 2)}", cmap='plasma')
    plt.scatter(np.array([0]), np.array([0]), vmin=min_cost, vmax=max_cost, c=np.array([properties_objective_sphere_final_converged[idx_clusters_sphere_final[1][0][0], 7]]), label=u"Cluster #2 \u2014 " + f"{round(mean_cluster[1], 2)}", cmap='plasma')
    plt.scatter(np.array([0]), np.array([0]), vmin=min_cost, vmax=max_cost, c=np.array([properties_objective_sphere_final_converged[idx_clusters_sphere_final[2][0][0], 7]]), label=u"Cluster #3 \u2014 " + f"{round(mean_cluster[2], 2)}", cmap='plasma')
    plt.scatter(np.array([0]), np.array([0]), vmin=min_cost, vmax=max_cost, c=np.array([properties_objective_sphere_final_converged[idx_clusters_sphere_final[3][0][0], 7]]), label=u"Cluster #4 \u2014 " + f"{round(mean_cluster[3], 2)}", cmap='plasma')

    plt.legend(loc='upper center', bbox_to_anchor=(-23, -0.15), ncol=4, frameon=False)

    duration_max = 0
    for i in range(4):
        duration = plot_lines('Constraint', '-', properties_constrained_converged[idx_clusters_Constrained[i][0], :], states_constrained_converged[:, :, idx_clusters_Constrained[i][0]], nb_shooting, cmap, ax[:, 0])
        duration = plot_lines('Objective (continuity)', '-', properties_objective_continuity_final_converged[idx_clusters_continuity_final[i][0], :], states_objective_continuity_final_converged[:, :, idx_clusters_continuity_final[i][0]], nb_shooting, cmap, ax[:, 1])
        duration = plot_lines('Objective (sphere)', '-', properties_objective_sphere_final_converged[idx_clusters_sphere_final[i][0], :], states_objective_sphere_final_converged[:, :, idx_clusters_sphere_final[i][0]], nb_shooting, cmap, ax[:, 2])
    if duration > duration_max:
        duration_max = duration

    ax[1, 0].set_xlabel("Time [s]", fontsize=12)
    ax[1, 1].set_xlabel("Time [s]", fontsize=12)
    ax[1, 2].set_xlabel("Time [s]", fontsize=12)
    ax[0, 0].set_title('Constrained', fontsize=12, fontweight='bold', style='italic')
    ax[0, 1].set_title('Penalized continuity', fontsize=12, fontweight='bold', style='italic')
    ax[0, 2].set_title('Penalized obstacles', fontsize=12, fontweight='bold', style='italic')
    ax[0, 0].set_ylabel("Translation [m]", fontsize=12)
    ax[1, 0].set_ylabel("Rotation [rad]", fontsize=12)

    ax[0, 0].set_xlim(0, duration_max)

    ax[0, 0].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[0, 1].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[0, 2].tick_params(axis='x', bottom=False, labelbottom=False)

    plt.subplots_adjust(bottom=0.2, top=0.93, left=0.05, right=0.9, hspace=0.1, wspace=0.15)
    plt.savefig("../figures/kinematics_graph.png", dpi=300)
    # plt.show()
    return min_cost, max_cost


def plot_stats(constrained_convergence_rate,
               objective_sphere_initial_convergence_rate,
               objective_sphere_final_convergence_rate,
               objective_continuity_initial_convergence_rate,
               objective_continuity_final_convergence_rate,
               pourcentage_cost_Constrained,
               pourcentage_cost_sphere_final,
               pourcentage_cost_continuity_final,
               pourcentage_transpersion_Constrained,
               pourcentage_transpersion_sphere_final,
               pourcentage_transpersion_continuity_final,
               mean_cluster,
               std_cluster,
               min_cost_clusters,
               max_cost_clusters,
               ):

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 5))

    cmap = cm.get_cmap('viridis')
    color_initial = cmap(0.25)
    color_final = cmap(0.75)

    cmap_cost = cm.get_cmap('plasma')
    color_0 = cmap_cost((mean_cluster[0] - min_cost_clusters) / (max_cost_clusters - min_cost_clusters))
    color_1 = cmap_cost((mean_cluster[1] - min_cost_clusters) / (max_cost_clusters - min_cost_clusters))
    color_2 = cmap_cost((mean_cluster[2] - min_cost_clusters) / (max_cost_clusters - min_cost_clusters))
    color_3 = cmap_cost((mean_cluster[3] - min_cost_clusters) / (max_cost_clusters - min_cost_clusters))
    colors = [color_0, color_1, color_2, color_3]


    ax[0].bar(1+0.2, constrained_convergence_rate, width=0.3, color=color_final)
    ax[0].bar(2-0.2, objective_continuity_initial_convergence_rate, width=0.3, color=color_initial)
    ax[0].bar(2+0.2, objective_continuity_final_convergence_rate, width=0.3, color=color_final)
    ax[0].bar(3-0.2, objective_sphere_initial_convergence_rate, width=0.3, color=color_initial, label=r"$\bf{First~optimization}$ step")
    ax[0].bar(3+0.2, objective_sphere_final_convergence_rate, width=0.3, color=color_final, label=r"$\bf{Second~optimization}$ step")

    ax[0].text(1-0.2-0.2, 0+1, 'NA')
    ax[0].text(1+0.2-0.2, constrained_convergence_rate + 1, f'{round(constrained_convergence_rate, 1)}%')
    ax[0].text(2-0.2-0.2, objective_continuity_initial_convergence_rate + 1, f'{round(objective_continuity_initial_convergence_rate, 1)}%')
    ax[0].text(2+0.2-0.2, objective_continuity_final_convergence_rate + 1, f'{round(objective_continuity_final_convergence_rate, 1)}%')
    ax[0].text(3-0.2-0.2, objective_sphere_initial_convergence_rate + 1, f'{round(objective_sphere_initial_convergence_rate, 1)}%')
    ax[0].text(3+0.2-0.2, objective_sphere_final_convergence_rate + 1, f'{round(objective_sphere_final_convergence_rate, 1)}%')

    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.17))

    ax[0].set_xlim((0.5, 3.5))
    ax[0].set_ylim((0, 110))
    ax[0].set_xticks([1, 2, 3])
    ax[0].set_xticklabels(['Constrained', 'Continuity\npenalized', 'Spheres\npenalized'], weight='bold', style='italic')
    # ax[0].xaxis.set_tick_params(weight='bold', style='italic')
    # ax[0].set_ylabel('[%]')
    ax[0].set_title('Convergence rate [%]')


    sum_constrained = 0
    sum_objective_sphere = 0
    sum_bjective_continuity = 0
    for i in range(4):
        ax[1].bar(1, pourcentage_cost_Constrained[i], bottom=sum_constrained, width=0.5, color=colors[i], label="Cluster #{} {:} ".format(i+1, round(mean_cluster[i], 2)) + u'\u00B1' + " {:.1e}".format(std_cluster[i]))
        ax[1].bar(2, pourcentage_cost_continuity_final[i], bottom=sum_bjective_continuity, width=0.5, color=colors[i])
        ax[1].bar(3, pourcentage_cost_sphere_final[i], bottom=sum_objective_sphere, width=0.5, color=colors[i])
        sum_constrained += pourcentage_cost_Constrained[i]
        sum_objective_sphere += pourcentage_cost_sphere_final[i]
        sum_bjective_continuity += pourcentage_cost_continuity_final[i]

    ax[1].text(1-0.2, sum_constrained + 1, f'{round(sum_constrained, 1)}%')
    ax[1].text(2-0.2, sum_bjective_continuity + 1, f'{round(sum_bjective_continuity, 1)}%')
    ax[1].text(3-0.2, sum_objective_sphere + 1, f'{round(sum_objective_sphere, 1)}%')

    ax[1].set_ylim((0, 110))
    ax[1].set_xticks([1, 2, 3])
    ax[1].set_xticklabels(['Constrained', 'Continuity\npenalized', 'Spheres\npenalized'], weight='bold', style='italic')
    # ax[1].xaxis.set_tick_params(weight='bold', style='italic')
    ax[1].set_title('Solutions in the $\it{recommended}$ clusters [%]')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.17))
    plt.subplots_adjust(bottom=0.30)


    ax[2].bar(1, pourcentage_transpersion_Constrained, width=0.5, color=color_final)
    ax[2].bar(2, pourcentage_transpersion_continuity_final, width=0.5, color=color_final)
    ax[2].bar(3, pourcentage_transpersion_sphere_final, width=0.5, color=color_final)

    ax[2].text(1-0.2, pourcentage_transpersion_Constrained + 1, f'{round(pourcentage_transpersion_Constrained, 1)}%')
    ax[2].text(2-0.2, pourcentage_transpersion_continuity_final + 1, f'{round(pourcentage_transpersion_continuity_final, 1)}%')
    ax[2].text(3-0.2, pourcentage_transpersion_sphere_final + 1, f'{round(pourcentage_transpersion_sphere_final, 1)}%')

    # ax[2].set_xlim((0.4, 3.6))
    ax[2].set_ylim((0, 110))
    ax[2].set_xticks([1, 2, 3])
    ax[2].set_xticklabels(['Constrained', 'Continuity\npenalized', 'Spheres\npenalized'], weight='bold', style='italic')
    # ax[2].xaxis.set_tick_params(weight='bold', style='italic')
    # ax[0].set_ylabel('[%]')
    ax[2].set_title('$\it{Admissible}$ solutions rate [%]')

    # plt.show()
    plt.savefig("../figures/pourcentage_stats.png", dpi=300)

    return

def weight_iter_plot(properties_objective_sphere_final_converged, properties_objective_sphere_final, properties_objective_continuity_final_converged, properties_objective_continuity_final, max_transpersion, max_cost, max_time_to_optimize):


    cmap = cm.get_cmap('viridis')

    def scatter_the_right_one(properti, properti_all_convergence_status, weight, iteration, ax, index_weight):
        idx = np.where(np.logical_and(properti[:, index_weight].astype(np.float) == weight, properti[:, 4].astype(np.float) == iteration))[0]
        plt_0 = ax.scatter(properti[idx, 7], properti[idx, 8], c=properti[idx, 10], vmin=0, vmax=max_transpersion, marker='.',
                   cmap='viridis')
        ax.errorbar(np.mean(properti[idx, 7]), np.mean(properti[idx, 8]), xerr=np.std(properti[idx, 7]),
            yerr=np.std(properti[idx, 8]), color=cmap(np.mean(properti[idx, 10]) / max_transpersion), capsize=3, capthick=1)
        # ax.set_title(f"iter = {iteration}, weight = {weight}", fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((80, max_cost))
        ax.set_ylim((1, max_time_to_optimize))
        ax.plot(np.array([194.74616649275964, 194.74616649275964]), np.array([1, max_time_to_optimize]), '--k',
                      alpha=0.5, linewidth=0.8)
        idx_all_convergence_status = np.where(
            np.logical_and(properti_all_convergence_status[:, index_weight].astype(np.float) == weight,
                           properti_all_convergence_status[:, 4].astype(np.float) == iteration))[0]
        ax.text(1800, 1500, '{:3.1f}%'.format(len(idx) / len(idx_all_convergence_status) * 100))
        return plt_0

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

    plt_0 = scatter_the_right_one(properties_objective_sphere_final_converged, properties_objective_sphere_final, weight=1000, iteration=100, ax=ax[0, 0], index_weight=3)
    plt_0 = scatter_the_right_one(properties_objective_sphere_final_converged, properties_objective_sphere_final, weight=100000, iteration=100, ax=ax[0, 1], index_weight=3)
    plt_0 = scatter_the_right_one(properties_objective_sphere_final_converged, properties_objective_sphere_final, weight=1000000, iteration=100, ax=ax[0, 2], index_weight=3)
    plt_0 = scatter_the_right_one(properties_objective_sphere_final_converged, properties_objective_sphere_final, weight=1000, iteration=1000, ax=ax[1, 0], index_weight=3)
    plt_0 = scatter_the_right_one(properties_objective_sphere_final_converged, properties_objective_sphere_final, weight=100000, iteration=1000, ax=ax[1, 1], index_weight=3)
    plt_0 = scatter_the_right_one(properties_objective_sphere_final_converged, properties_objective_sphere_final, weight=1000000, iteration=1000, ax=ax[1, 2], index_weight=3)
    plt_0 = scatter_the_right_one(properties_objective_sphere_final_converged, properties_objective_sphere_final, weight=1000, iteration=10000, ax=ax[2, 0], index_weight=3)
    plt_0 = scatter_the_right_one(properties_objective_sphere_final_converged, properties_objective_sphere_final, weight=100000, iteration=10000, ax=ax[2, 1], index_weight=3)
    plt_0 = scatter_the_right_one(properties_objective_sphere_final_converged, properties_objective_sphere_final, weight=1000000, iteration=10000, ax=ax[2, 2], index_weight=3)

    ax[2, 0].set_xlabel('Cost', fontsize=12)
    ax[2, 1].set_xlabel('Cost', fontsize=12)
    ax[2, 2].set_xlabel('Cost', fontsize=12)
    ax[0, 0].set_ylabel('Computational time [s]', fontsize=12)
    ax[1, 0].set_ylabel('Computational time [s]', fontsize=12)
    ax[2, 0].set_ylabel('Computational time [s]', fontsize=12)

    ax[0, 0].text(160, 6000, 'weight=1000', fontsize=15, weight='bold')
    ax[0, 1].text(140, 6000, 'weight=100000', fontsize=15, weight='bold')
    ax[0, 2].text(130, 6000, 'weight=1000000', fontsize=15, weight='bold')
    ax[0, 0].text(17, 12, 'iter=100', fontsize=15, weight='bold', rotation=90)
    ax[1, 0].text(17, 12, 'iter=1000', fontsize=15, weight='bold', rotation=90)
    ax[2, 0].text(17, 12, 'iter=10000', fontsize=15, weight='bold', rotation=90)

    ax[0, 1].text(110, 20000, 'Penalized obstacles', fontsize=18)

    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.2, right=0.8)
    cbar_ax = fig.add_axes([0.84, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(plt_0, cax=cbar_ax)
    cbar.ax.set_title('Max penetration\n[m]', fontsize=12)
    plt.savefig("../figures/convergence_info_graph_sphere.png", dpi=300)
    # plt.show()


    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

    plt_0 = scatter_the_right_one(properties_objective_continuity_final_converged, properties_objective_continuity_final, weight=1000, iteration=100, ax=ax[0, 0], index_weight=5)
    plt_0 = scatter_the_right_one(properties_objective_continuity_final_converged, properties_objective_continuity_final, weight=100000, iteration=100, ax=ax[0, 1], index_weight=5)
    plt_0 = scatter_the_right_one(properties_objective_continuity_final_converged, properties_objective_continuity_final, weight=1000000, iteration=100, ax=ax[0, 2], index_weight=5)
    plt_0 = scatter_the_right_one(properties_objective_continuity_final_converged, properties_objective_continuity_final, weight=1000, iteration=1000, ax=ax[1, 0], index_weight=5)
    plt_0 = scatter_the_right_one(properties_objective_continuity_final_converged, properties_objective_continuity_final, weight=100000, iteration=1000, ax=ax[1, 1], index_weight=5)
    plt_0 = scatter_the_right_one(properties_objective_continuity_final_converged, properties_objective_continuity_final, weight=1000000, iteration=1000, ax=ax[1, 2], index_weight=5)
    plt_0 = scatter_the_right_one(properties_objective_continuity_final_converged, properties_objective_continuity_final, weight=1000, iteration=10000, ax=ax[2, 0], index_weight=5)
    plt_0 = scatter_the_right_one(properties_objective_continuity_final_converged, properties_objective_continuity_final, weight=100000, iteration=10000, ax=ax[2, 1], index_weight=5)
    plt_0 = scatter_the_right_one(properties_objective_continuity_final_converged, properties_objective_continuity_final, weight=1000000, iteration=10000, ax=ax[2, 2], index_weight=5)

    ax[2, 0].set_xlabel('Cost', fontsize=12)
    ax[2, 1].set_xlabel('Cost', fontsize=12)
    ax[2, 2].set_xlabel('Cost', fontsize=12)
    ax[0, 0].set_ylabel('Computational time [s]', fontsize=12)
    ax[1, 0].set_ylabel('Computational time [s]', fontsize=12)
    ax[2, 0].set_ylabel('Computational time [s]', fontsize=12)

    ax[0, 0].text(160, 6000, 'weight=1000', fontsize=15, weight='bold')
    ax[0, 1].text(140, 6000, 'weight=100000', fontsize=15, weight='bold')
    ax[0, 2].text(130, 6000, 'weight=1000000', fontsize=15, weight='bold')
    ax[0, 0].text(17, 12, 'iter=100', fontsize=15, weight='bold', rotation=90)
    ax[1, 0].text(17, 12, 'iter=1000', fontsize=15, weight='bold', rotation=90)
    ax[2, 0].text(17, 12, 'iter=10000', fontsize=15, weight='bold', rotation=90)

    ax[0, 1].text(85, 20000, 'Penalized continuity', fontsize=18)

    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.2, right=0.8)
    cbar_ax = fig.add_axes([0.84, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(plt_0, cax=cbar_ax)
    cbar.ax.set_title('Max penetration\n[m]', fontsize=12)
    plt.savefig("../figures/convergence_info_graph_continuity.png", dpi=300)
    # plt.show()

    return


def plot_marker_trajectory(states_objective_sphere_final_converged, idx_clusters_sphere_final, mean_cluster,
                           min_cost_clusters, max_cost_clusters,):

    m = biorbd.Model('../models/pendulum_maze.bioMod')
    markers = np.zeros((4, states_objective_sphere_final_converged.shape[1], 3))
    for i in range(4):
        for j in range(states_objective_sphere_final_converged.shape[1]):
            markers[i, j, :] = m.markers(states_objective_sphere_final_converged[:, j, idx_clusters_sphere_final[i][0][0]])[0].to_array()

    cmap_cost = cm.get_cmap('plasma')
    color_0 = cmap_cost((mean_cluster[0] - min_cost_clusters) / (max_cost_clusters - min_cost_clusters))
    color_1 = cmap_cost((mean_cluster[1] - min_cost_clusters) / (max_cost_clusters - min_cost_clusters))
    color_2 = cmap_cost((mean_cluster[2] - min_cost_clusters) / (max_cost_clusters - min_cost_clusters))
    color_3 = cmap_cost((mean_cluster[3] - min_cost_clusters) / (max_cost_clusters - min_cost_clusters))
    colors = [color_0, color_1, color_2, color_3]

    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0.05, 0), 0.35, color='r', clip_on=False))
    ax.add_patch(plt.Circle((0.75, 0.2), 0.35, color='r', clip_on=False))
    ax.add_patch(plt.Circle((-0.95, 0), 0.35, color='r', clip_on=False))
    ax.add_patch(plt.Circle((1.4, 0.5), 0.35, color='r', clip_on=False))

    for i in range(4):
        ax.plot(markers[i, :, 1], markers[i, :, 2], linestyle='-', color=colors[i], label=f"Cluster #{i+1}" + u"\u2014 " + f"{round(mean_cluster[0], 2)}")
        ax.plot(markers[i, :, 1], markers[i, :, 2], '.', color=colors[i])
    ax.plot(markers[i, 0, 1], markers[i, 0, 2], 'ok')


    ax.set_xlabel('Position Y [m]')
    ax.set_ylabel('Position Z [m]')
    ax.axis('equal')
    plt.subplots_adjust(bottom=0.25)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    plt.savefig("../figures/marker_trajectory.png", dpi=300)
    # plt.show()



    cmap_cost = cm.get_cmap('plasma')
    colors = [cmap_cost(mean_cluster[0]), cmap_cost(mean_cluster[1]), cmap_cost(mean_cluster[2]), cmap_cost(mean_cluster[3])]

    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0.05, 0), 0.35, color='r', clip_on=True))
    ax.add_patch(plt.Circle((0.75, 0.2), 0.35, color='r', clip_on=True))
    ax.add_patch(plt.Circle((-0.95, 0), 0.35, color='r', clip_on=True))
    ax.add_patch(plt.Circle((1.4, 0.5), 0.35, color='r', clip_on=True))

    for i in range(4):
        ax.plot(markers[i, :, 1], markers[i, :, 2], linestyle='-', marker='.', color=colors[i])
    ax.plot(markers[i, 0, 1], markers[i, 0, 2], 'ok')

    ax.set_aspect('equal', 'box')
    ax.set_xlim((1, 1.2)),
    ax.set_ylim((0.2, 0.4)),
    ax.set_xlabel('Position Y [m]')
    ax.set_ylabel('Position Z [m]')
    plt.subplots_adjust(bottom=0.25)
    plt.savefig("../figures/marker_trajectory_zoomed.png", dpi=300)
    # plt.show()

    return



def plot_iteration_time(
                    constrained_iteration,
                    constrained_time_per_iteration,
                    objective_sphere_initial_iteration,
                    objective_sphere_initial_time_per_iteration,
                    objective_sphere_final_iteration,
                    objective_sphere_final_time_per_iteration,
                    objective_continuity_initial_iteration,
                    objective_continuity_initial_time_per_iteration,
                    objective_continuity_final_iteration,
                    objective_continuity_final_time_per_iteration,
                    ):

    FLAG_COLORMAP = False

    cmap = cm.get_cmap('viridis')
    color_initial = cmap(0.25)
    color_final = cmap(0.75)

    print(f"{np.median(constrained_iteration)} : Median # iteration constrained")
    print(f"{np.median(objective_continuity_initial_iteration)} : Median # iteration continuity initial")
    print(f"{np.median(objective_continuity_final_iteration)} : Median # iteration continuity final")
    print(f"{np.median(objective_sphere_initial_iteration)} : Median # iteration spheres initial")
    print(f"{np.median(objective_sphere_final_iteration)} : Median # iteration sphere final")

    print(f"{np.median(constrained_time_per_iteration)} : Median mean time per iteration constrained")
    print(f"{np.median(objective_continuity_initial_time_per_iteration)} : Median mean time per iteration continuity initial")
    print(f"{np.median(objective_continuity_final_time_per_iteration)} : Median mean time per iteration continuity final")
    print(f"{np.median(objective_sphere_initial_time_per_iteration)} : Median mean time per iteration spheres initial")
    print(f"{np.median(objective_sphere_final_time_per_iteration)} : Median mean time per iteration sphere final")

    if FLAG_COLORMAP:

        fig, ax = plt.subplots(figsize=(9, 6))

        min_time_per_iteration = min(np.min(constrained_time_per_iteration),
                                     np.min(objective_continuity_initial_time_per_iteration),
                                     np.min(objective_continuity_final_time_per_iteration),
                                     np.min(objective_sphere_initial_time_per_iteration),
                                     np.min(objective_sphere_final_time_per_iteration))
        # max_time_per_iteration = max(np.max(constrained_time_per_iteration),
        #                              np.max(objective_continuity_initial_time_per_iteration),
        #                              np.max(objective_continuity_final_time_per_iteration),
        #                              np.max(objective_sphere_initial_time_per_iteration),
        #                              np.max(objective_sphere_final_time_per_iteration))
        max_time_per_iteration = np.max(objective_sphere_initial_time_per_iteration)

        plt_0 = ax.scatter(np.random.random((len(constrained_iteration))) * 0.2 + (1 + 0.2) - 0.1, constrained_iteration, c=constrained_time_per_iteration, norm=colors.LogNorm(vmin=min_time_per_iteration, vmax=max_time_per_iteration), alpha=0.6, edgecolors=None, s=0.4)
        ax.scatter(np.random.random((len(objective_continuity_initial_iteration))) * 0.2 + (2 - 0.2) - 0.1, objective_continuity_initial_iteration, c=objective_continuity_initial_time_per_iteration, norm=colors.LogNorm(vmin=min_time_per_iteration, vmax=max_time_per_iteration), alpha=0.6, edgecolors=None, s=0.4)
        ax.scatter(np.random.random((len(objective_continuity_final_iteration))) * 0.2 + (2 + 0.2) - 0.1, objective_continuity_final_iteration, c=objective_continuity_final_time_per_iteration, norm=colors.LogNorm(vmin=min_time_per_iteration, vmax=max_time_per_iteration), alpha=0.6, edgecolors=None, s=0.4)
        ax.scatter(np.random.random((len(objective_sphere_initial_iteration))) * 0.2 + (3 - 0.2) - 0.1, objective_sphere_initial_iteration, c=objective_sphere_initial_time_per_iteration, norm=colors.LogNorm(vmin=min_time_per_iteration, vmax=max_time_per_iteration), alpha=0.6, edgecolors=None, s=0.4)
        ax.scatter(np.random.random((len(objective_sphere_final_iteration))) * 0.2 + (3 + 0.2) - 0.1, objective_sphere_final_iteration, c=objective_sphere_final_time_per_iteration, norm=colors.LogNorm(vmin=min_time_per_iteration, vmax=max_time_per_iteration), alpha=0.6, edgecolors=None, s=0.4)

        ax.plot(np.array([0.4, 3.6]), np.array([10000, 10000]), '--k', alpha=0.5, linewidth=0.8)
        ax.text(3.7, 10000, 'Maximum\nIteration')

        ax.set_xlim((0.4, 3.6))
        ax.set_xticks([1-0.2, 1+0.2, 2-0.2, 2+0.2, 3-0.2, 3+0.2], fontsize=12)
        ax.set_xticklabels(['First\noptim', 'Second\noptim', 'First\noptim', 'Second\noptim', 'First\noptim', 'Second\noptim'], fontsize=12)

        ax.text(1-0.25, 0, 'NA', fontsize=12)
        ax.text(1-0.22, -2000, 'Constrained', weight='bold', fontsize=12)
        ax.text(2-0.37, -2000, 'Penalized continuity', weight='bold', fontsize=12)
        ax.text(3-0.33, -2000, 'Penalized obstacles', weight='bold', fontsize=12)
        ax.set_ylabel('Number of iterations', fontsize=12)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)

        plt.subplots_adjust(bottom=0.15, top=0.85, right=0.85)
        cbar_ax = fig.add_axes([0.89, 0.15, 0.03, 0.7])
        cbar = plt.colorbar(plt_0, cax=cbar_ax)
        cbar.ax.set_title('Mean time\nper iteration\n[s]', fontsize=12)

        plt.savefig("../figures/time_per_iteration_color.png", dpi=300)
        # plt.show()

    else:

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.plot(np.random.random((len(constrained_iteration))) * 0.2 + (1 + 0.2) - 0.1,
                           constrained_iteration, '.', color=color_final, markersize=0.4, alpha=0.95, mec=None)
        ax.plot(np.random.random((len(objective_continuity_initial_iteration))) * 0.2 + (2 - 0.2) - 0.1,
                   objective_continuity_initial_iteration, '.', color=color_initial, markersize=0.4, alpha=0.95, mec=None)
        ax.plot(np.random.random((len(objective_continuity_final_iteration))) * 0.2 + (2 + 0.2) - 0.1,
                   objective_continuity_final_iteration, '.', color=color_final, markersize=0.4, alpha=0.95, mec=None)
        ax.plot(np.random.random((len(objective_sphere_initial_iteration))) * 0.2 + (3 - 0.2) - 0.1,
                   objective_sphere_initial_iteration, '.', color=color_initial, markersize=0.4, alpha=0.95, mec=None)
        ax.plot(np.random.random((len(objective_sphere_final_iteration))) * 0.2 + (3 + 0.2) - 0.1,
                   objective_sphere_final_iteration, '.', color=color_final, markersize=0.4, alpha=0.95, mec=None)

        ax.plot(np.array([1+0.2-0.15, 1+0.2+0.15]), np.ones((2,)) * np.median(constrained_iteration), color=color_final, linewidth=1.2)
        ax.plot(np.array([2-0.2-0.15, 2-0.2+0.15]), np.ones((2,)) * np.median(objective_continuity_initial_iteration), color=color_initial, linewidth=1.2)
        ax.plot(np.array([2+0.2-0.15, 2+0.2+0.15]), np.ones((2,)) * np.median(objective_continuity_final_iteration), color=color_final, linewidth=1.2)
        ax.plot(np.array([3-0.2-0.15, 3-0.2+0.15]), np.ones((2,)) * np.median(objective_sphere_initial_iteration), color=color_initial, linewidth=1.2)
        ax.plot(np.array([3+0.2-0.15, 3+0.2+0.15]), np.ones((2,)) * np.median(objective_sphere_final_iteration), color=color_final, linewidth=1.2)

        ax.plot(np.array([0.4, 3.6]), np.array([10000, 10000]), '--k', alpha=0.5, linewidth=0.8)
        ax.text(3.7, 10000, 'Maximum\nIteration')

        ax.set_yscale('log')
        ax.set_xlim((0.4, 3.6))
        ax.set_xticks([1 - 0.2, 1 + 0.2, 2 - 0.2, 2 + 0.2, 3 - 0.2, 3 + 0.2], fontsize=12)
        ax.set_xticklabels(
            ['First\noptim', 'Second\noptim', 'First\noptim', 'Second\noptim', 'First\noptim', 'Second\noptim'],
            fontsize=12)

        ax.text(1 - 0.25, 20, 'NA', fontsize=12)
        ax.text(1 - 0.22, 6.5, 'Constrained', weight='bold', fontsize=12)
        ax.text(2 - 0.37, 6.5, 'Penalized continuity', weight='bold', fontsize=12)
        ax.text(3 - 0.33, 6.5, 'Penalized obstacles', weight='bold', fontsize=12)
        ax.set_ylabel('Number of iterations', fontsize=12)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)

        plt.subplots_adjust(bottom=0.15, top=0.85, right=0.85)

        plt.savefig("../figures/iteration.png", dpi=300)
        # plt.show()


        fig, ax = plt.subplots(figsize=(9, 6))

        ax.plot(np.random.random((len(constrained_time_per_iteration))) * 0.2 + (1 + 0.2) - 0.1,
                           constrained_time_per_iteration, '.', color=color_final, markersize=0.4, alpha=0.95, mec=None)
        ax.plot(np.random.random((len(objective_continuity_initial_time_per_iteration))) * 0.2 + (2 - 0.2) - 0.1,
                   objective_continuity_initial_time_per_iteration, '.', color=color_initial, markersize=0.4, alpha=0.95, mec=None)
        ax.plot(np.random.random((len(objective_continuity_final_time_per_iteration))) * 0.2 + (2 + 0.2) - 0.1,
                   objective_continuity_final_time_per_iteration, '.', color=color_final, markersize=0.4, alpha=0.95, mec=None)
        ax.plot(np.random.random((len(objective_sphere_initial_time_per_iteration))) * 0.2 + (3 - 0.2) - 0.1,
                   objective_sphere_initial_time_per_iteration, '.', color=color_initial, markersize=0.4, alpha=0.95, mec=None)
        ax.plot(np.random.random((len(objective_sphere_final_time_per_iteration))) * 0.2 + (3 + 0.2) - 0.1,
                   objective_sphere_final_time_per_iteration, '.', color=color_final, markersize=0.4, alpha=0.95, mec=None)

        ax.plot(np.array([1 + 0.2 - 0.15, 1 + 0.2 + 0.15]), np.ones((2,)) * np.median(constrained_time_per_iteration), color=color_final, linewidth=1.2)
        ax.plot(np.array([2 - 0.2 - 0.15, 2 - 0.2 + 0.15]), np.ones((2,)) * np.median(objective_continuity_initial_time_per_iteration),
                color=color_initial, linewidth=1.2)
        ax.plot(np.array([2 + 0.2 - 0.15, 2 + 0.2 + 0.15]), np.ones((2,)) * np.median(objective_continuity_final_time_per_iteration),
                color=color_final, linewidth=1.2)
        ax.plot(np.array([3 - 0.2 - 0.15, 3 - 0.2 + 0.15]), np.ones((2,)) * np.median(objective_sphere_initial_time_per_iteration),
                color=color_initial, linewidth=1.2)
        ax.plot(np.array([3 + 0.2 - 0.15, 3 + 0.2 + 0.15]), np.ones((2,)) * np.median(objective_sphere_final_time_per_iteration),
                color=color_final, linewidth=1.2)

        ax.set_yscale('log')
        ax.set_xlim((0.4, 3.6))
        ax.set_xticks([1 - 0.2, 1 + 0.2, 2 - 0.2, 2 + 0.2, 3 - 0.2, 3 + 0.2], fontsize=12)
        ax.set_xticklabels(
            ['First\noptim.', 'Second\noptim.', 'First\noptim.', 'Second\noptim.', 'First\noptim.', 'Second\noptim.'],
            fontsize=12)

        ax.text(1 - 0.25, 0.09, 'NA', fontsize=12)
        ax.text(1 - 0.22, 0.035, 'Constrained', weight='bold', fontsize=12)
        ax.text(2 - 0.37, 0.035, 'Penalized continuity', weight='bold', fontsize=12)
        ax.text(3 - 0.33, 0.035, 'Penalized obstacles', weight='bold', fontsize=12)
        ax.set_ylabel('Computational time per iteration [s]', fontsize=12)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)

        plt.subplots_adjust(bottom=0.15, top=0.85, right=0.85)

        plt.savefig("../figures/time_per_iteration.png", dpi=300)
        # plt.show()

    return



def plot_why_penetrating_sol_high_cost(states, cost, transpersion, FLAG_GENERATE_VIDEO):

    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0.05, 0), 0.35, color='r', clip_on=False))
    ax.add_patch(plt.Circle((0.75, 0.2), 0.35, color='r', clip_on=False))
    ax.add_patch(plt.Circle((-0.95, 0), 0.35, color='r', clip_on=False))
    ax.add_patch(plt.Circle((1.4, 0.5), 0.35, color='r', clip_on=False))

    for i in range(np.shape(states)[2]):
        if transpersion[i] > 0:
            m = biorbd.Model('../models/pendulum_maze.bioMod')
            markers = np.zeros((states.shape[1], 3))
            for j in range(states.shape[1]):
                markers[j, :] = m.markers(states[:2, j, i])[0].to_array()

            ax.plot(markers[:, 1], markers[:, 2], linestyle='-', marker='.', color='k', linewidth=0.5)
            ax.text(2.8, 2, f'Cost = {round(cost[i], 3)}')
            break

    # ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Position Y [m]')
    ax.set_ylabel('Position Z [m]')
    ax.axis('equal')
    plt.savefig("../figures/bad_solution.png", dpi=300)
    # plt.legend()
    # plt.show()

    if FLAG_GENERATE_VIDEO:
        embed()
        animate(states[:2, :, i])
    return

def animate(states):
    model_path = "/home/mickaelbegon/Documents/Eve/ContinuityObjectiveTests-master/models/pendulum_maze.bioMod"
    b = bioviz.Viz(model_path, show_segments_center_of_mass=False,
                   show_local_ref_frame=False, background_color=(1, 1, 1), markers_size=0.01)
    b.load_movement(states)
    b.exec()
    return

def animate_clusters(idx_clusters_sphere_final, states_objective_sphere_final_converged):
    for i in range(4):
        animate(states_objective_sphere_final_converged[:2, :, idx_clusters_sphere_final[i][0][0]])
    return

def why_penetrating_sol_high_cost_metrics(controls_bad, transpersion_bad, duration_bad,
                                          nb_shooting,
                                          controls_clusters, duration_clusters,
                                          idx_clusters):

    for i in range(np.shape(controls_bad)[2]):
        if transpersion_bad[i] > 0:
            dt_bad = duration_bad[i]/nb_shooting
            print("Inter-shooting node duration bad solution: ", dt_bad)
            print("Lagrange control bad solution: ", np.nansum(np.abs(controls_bad[:, :, i] * dt_bad)))
            break

    dt_best = duration_clusters[idx_clusters[0][0][0]] / nb_shooting
    print("Inter-shooting node duration best solution: ", dt_best)
    print("Lagrange control best solution: ", np.nansum(np.abs(controls_clusters[:, :, idx_clusters[0][0][0]] * dt_best)))

    return

#########   Loading data   #########

LOAD_DATA_FLAG = False  # True
FLAG_GENERATE_VIDEO = False # True
nb_shooting = 500

if LOAD_DATA_FLAG:
    directory = "../solutions_IPOPT"

    m = biorbd.Model("../models/pendulum_maze.bioMod")

    # loop over the files in the folder to load them
    properties_all = []
    states_all = []
    for filename in os.listdir(directory):
        if filename[-7:] == ".pickle":
            if filename.split("-")[0] == 'unconstrained':
                continue
            with open(f"{directory}/{filename}", "rb") as f:
                data_sol = pickle.load(f)
            properties, states, controls = extract_data_sol(m, filename, data_sol)
            if np.shape(properties_all) == (0, ):
                properties_all = properties
                states_all = states
                controls_all = controls
            else:
                properties_all = np.vstack((properties_all, properties))
                states_all = np.dstack((states_all, states))
                controls_all = np.dstack((controls_all, controls))
            print(f"Loaded successfully {filename}")
    
    with open('data.pkl', 'wb') as file:
        data = [properties_all, states_all, controls_all]
        pickle.dump(data, file)
else:
    file = open('data.pkl', 'rb')
    data = pickle.load(file)
    properties_all = data[0]
    states_all = data[1]
    controls_all = data[2]


# Sort the files loaded + print convergence rate
properties_constrained = properties_all[np.where(properties_all[:, 0] == "constraint"), :][0]
states_constrained = states_all[:, :, np.where(properties_all[:, 0] == "constraint")][:, :, 0, :]
constrained_index_converged = np.where(properties_constrained[:, 9] == 0)
properties_constrained_converged = properties_constrained[constrained_index_converged, :][0]
states_constrained_converged = states_constrained[:, :, constrained_index_converged][:, :, 0, :]
constrained_convergence_rate = len(constrained_index_converged[0]) / len(properties_constrained) * 100
print("Convergence rate fully constrained OCP : ", constrained_convergence_rate, "%")


properties_objective_sphere = properties_all[np.where(properties_all[:, 0] == "spheres_in_objective"), :][0]
states_objective_sphere = states_all[:, :, np.where(properties_all[:, 0] == "spheres_in_objective")][:, :, 0, :]
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


properties_objective_continuity = properties_all[np.where(properties_all[:, 0] == "continuity_in_objective"), :][0]
states_objective_continuity = states_all[:, :, np.where(properties_all[:, 0] == "continuity_in_objective")][:, :, 0, :]
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

print("\n\n")

properties_all_converged = properties_all[np.where(properties_all[:, 9] == 0), :][0]

max_cost = np.max(properties_all_converged[:, 7])
min_cost = np.min(properties_all_converged[:, 7])
max_time_to_optimize = np.max(properties_all_converged[:, 8])
min_time_to_optimize = np.min(properties_all_converged[:, 8])
max_transpersion = 0.12 # np.max(properties_all_converged[:, 10])
min_transpersion = 0 # np.min(properties_all_converged[:, 10])


# In the 4 best solutions clusters
cost_clusters = [[138.41,  138.42], [146.26, 146.30], [160.05, 160.07], [194.74, 194.76]]
mean_cluster = []
std_cluster = []
for i in range(4):
    idx_this_cluster = np.where(np.logical_and(properties_all_converged[:, 7] > cost_clusters[i][0], properties_all_converged[:, 7] < cost_clusters[i][1]))
    mean_cluster += [np.mean(properties_all_converged[idx_this_cluster, 7])]
    std_cluster += [np.std(properties_all_converged[idx_this_cluster, 7])]


idx_clusters_Constrained = []
idx_clusters_sphere_final = []
idx_clusters_continuity_final = []
for i in range(4):
    idx_clusters_Constrained += [np.where(np.logical_and(properties_constrained_converged[:, 7] > cost_clusters[i][0], properties_constrained_converged[:, 7] < cost_clusters[i][1] ))]
    idx_clusters_sphere_final += [np.where(np.logical_and(properties_objective_sphere_final_converged[:, 7] > cost_clusters[i][0], properties_objective_sphere_final_converged[:, 7] < cost_clusters[i][1] ))]
    idx_clusters_continuity_final += [np.where(np.logical_and(properties_objective_continuity_final_converged[:, 7] > cost_clusters[i][0], properties_objective_continuity_final_converged[:, 7] < cost_clusters[i][1] ))]

if FLAG_GENERATE_VIDEO:
    animate_clusters(idx_clusters_sphere_final, states_objective_sphere_final_converged)

# The solutions that did not transperse
idx_no_trans_Constrained = np.where(properties_constrained_converged[:, 10] == 0)[0]
idx_no_trans_sphere_final = np.where(properties_objective_sphere_final_converged[:, 10] == 0)[0]
idx_no_trans_continuity_final = np.where(properties_objective_continuity_final_converged[:, 10] == 0)[0]

# 90th percentile pf the cost value AND 5th percentile of the transpersion
idx_good_Constrained = []
idx_good_sphere_final = []
idx_good_continuity_final = []
for i in range(4):
    idx_good_Constrained += [np.intersect1d(idx_clusters_Constrained[i], idx_no_trans_Constrained)]
    idx_good_sphere_final += [np.intersect1d(idx_clusters_sphere_final[i], idx_no_trans_sphere_final)]
    idx_good_continuity_final += [np.intersect1d(idx_clusters_continuity_final[i], idx_no_trans_continuity_final)]

pourcentage_cost_Constrained = []
pourcentage_cost_sphere_final = []
pourcentage_cost_continuity_final = []
for i in range(4):
    pourcentage_cost_Constrained += [np.shape(idx_clusters_Constrained[i][0])[0] / np.shape(properties_constrained_converged)[0] * 100]
    pourcentage_cost_sphere_final += [np.shape(idx_clusters_sphere_final[i][0])[0] / np.shape(properties_objective_sphere_final_converged)[0] * 100]
    pourcentage_cost_continuity_final += [np.shape(idx_clusters_continuity_final[i][0])[0] / np.shape(properties_objective_continuity_final_converged)[0] * 100]
print(f"{pourcentage_cost_Constrained} % of the constrained solutions in the clusters for the cost function value")
print(f"{pourcentage_cost_sphere_final} % of the objective_sphere solutions in the clusters for the cost function value")
print(f"{pourcentage_cost_continuity_final} % of the objective_continuity solutions in the clusters for the cost function value")

pourcentage_transpersion_Constrained = len(idx_no_trans_Constrained) / np.shape(properties_constrained_converged)[0] * 100
pourcentage_transpersion_sphere_final = len(idx_no_trans_sphere_final) / np.shape(properties_objective_sphere_final_converged)[0] * 100
pourcentage_transpersion_continuity_final = len(idx_no_trans_continuity_final) / np.shape(properties_objective_continuity_final_converged)[0] * 100
print(f"{pourcentage_transpersion_Constrained} % of the constrained solutions did not transperse the spheres betwee the shooting nodes")
print(f"{pourcentage_transpersion_sphere_final} % of the objective_sphere solutions did not transperse the spheres betwee the shooting nodes")
print(f"{pourcentage_transpersion_continuity_final} % of the objective_continuity solutions did not transperse the spheres betwee the shooting nodes")

pourcentage_good_Constrained = []
pourcentage_good_sphere_final = []
pourcentage_good_continuity_final = []
for i in range(4):
    pourcentage_good_Constrained += [len(idx_good_Constrained[i]) / np.shape(properties_constrained_converged)[0] * 100]
    pourcentage_good_sphere_final += [len(idx_good_sphere_final[i]) / np.shape(properties_objective_sphere_final_converged)[0] * 100]
    pourcentage_good_continuity_final += [len(idx_good_continuity_final[i]) / np.shape(properties_objective_continuity_final_converged)[0] * 100]
print(f"{pourcentage_good_Constrained} % of the constrained solutions did not transperse and were in the clusters")
print(f"{pourcentage_good_sphere_final} % of the objective_sphere solutions did not transperse and were in the clusters")
print(f"{pourcentage_good_continuity_final} % of the objective_continuity solutions did not transperse and were in the clusters")

print("\n\n")

constrained_mean_iteration = np.mean(properties_constrained_converged[:, 11])
constrained_mean_time_per_iteration = np.mean(properties_constrained_converged[:, 12])
objective_sphere_initial_mean_iteration = np.mean(properties_objective_sphere_initial_converged[:, 11])
objective_sphere_initial_mean_time_per_iteration = np.mean(properties_objective_sphere_initial_converged[:, 12])
objective_sphere_final_mean_iteration = np.mean(properties_objective_sphere_final_converged[:, 11])
objective_sphere_final_mean_time_per_iteration = np.mean(properties_objective_sphere_final_converged[:, 12])
objective_continuity_initial_mean_iteration = np.mean(properties_objective_continuity_initial_converged[:, 11])
objective_continuity_initial_mean_time_per_iteration = np.mean(properties_objective_continuity_initial_converged[:, 12])
objective_continuity_final_mean_iteration = np.mean(properties_objective_continuity_final_converged[:, 11])
objective_continuity_final_mean_time_per_iteration = np.mean(properties_objective_continuity_final_converged[:, 12])
constrained_std_iteration = np.std(properties_constrained_converged[:, 11])
constrained_std_time_per_iteration = np.std(properties_constrained_converged[:, 12])
objective_sphere_initial_std_iteration = np.std(properties_objective_sphere_initial_converged[:, 11])
objective_sphere_initial_std_time_per_iteration = np.std(properties_objective_sphere_initial_converged[:, 12])
objective_sphere_final_std_iteration = np.std(properties_objective_sphere_final_converged[:, 11])
objective_sphere_final_std_time_per_iteration = np.std(properties_objective_sphere_final_converged[:, 12])
objective_continuity_initial_std_iteration = np.std(properties_objective_continuity_initial_converged[:, 11])
objective_continuity_initial_std_time_per_iteration = np.std(properties_objective_continuity_initial_converged[:, 12])
objective_continuity_final_std_iteration = np.std(properties_objective_continuity_final_converged[:, 11])
objective_continuity_final_std_time_per_iteration = np.std(properties_objective_continuity_final_converged[:, 12])
print(f"{constrained_mean_iteration} +- {constrained_std_iteration} iterations needed for the constrained solutions to converge with a mean of {constrained_mean_time_per_iteration} +- {constrained_std_time_per_iteration}s per iteration")
print(f"{objective_continuity_initial_mean_iteration} +- {objective_continuity_initial_std_iteration} iterations needed for the objective continuity initial solutions to converge with a mean of {objective_continuity_initial_mean_time_per_iteration} +- {objective_continuity_initial_std_time_per_iteration}s per iteration")
print(f"{objective_continuity_final_mean_iteration} +- {objective_continuity_final_std_iteration} iterations needed for the objective continuity final solutions to converge with a mean of {objective_continuity_initial_mean_time_per_iteration} +- {objective_continuity_initial_std_time_per_iteration}s per iteration")
print(f"{objective_sphere_initial_mean_iteration} +- {objective_sphere_initial_std_iteration} iterations needed for the objective sphere initial solutions to converge with a mean of {objective_sphere_initial_mean_time_per_iteration} +- {objective_sphere_initial_std_time_per_iteration}s per iteration")
print(f"{objective_sphere_final_mean_iteration} +- {objective_sphere_final_std_iteration} iterations needed for the objective sphere final solutions to converge with a mean of {objective_sphere_final_mean_time_per_iteration} +- {objective_sphere_final_std_time_per_iteration}s per iteration")

constrained_computational_time_median = np.median(properties_constrained_converged[:, 8])
objective_sphere_initial_computational_time_median = np.median(properties_objective_sphere_initial_converged[:, 8])
objective_sphere_final_computational_time_median = np.median(properties_objective_sphere_final_converged[:, 8])
objective_continuity_initial_computational_time_median = np.median(properties_objective_continuity_initial_converged[:, 8])
objective_continuity_final_computational_time_median = np.median(properties_objective_continuity_final_converged[:, 8])
constrained_computational_time_std = np.std(properties_constrained_converged[:, 8])
objective_sphere_initial_computational_time_std = np.std(properties_objective_sphere_initial_converged[:, 8])
objective_sphere_final_computational_time_std = np.std(properties_objective_sphere_final_converged[:, 8])
objective_continuity_initial_computational_time_std = np.std(properties_objective_continuity_initial_converged[:, 8])
objective_continuity_final_computational_time_std = np.std(properties_objective_continuity_final_converged[:, 8])
print(f"{constrained_computational_time_median} +- {constrained_computational_time_std}s needed for the constrained solutions to converge")
print(f"{objective_sphere_final_computational_time_median} +- {objective_sphere_final_computational_time_std}s needed for the initial + final spheres solutions to converge")
print(f"{objective_continuity_final_computational_time_median} +- {objective_continuity_final_computational_time_std}s needed for the initial + final continuity solutions to converge")

print("\n\n")

graph_convergence(properties_constrained_converged,
                  properties_objective_sphere_final_converged, 
                  properties_objective_continuity_final_converged,
                  max_cost,
                  max_time_to_optimize,
                  max_transpersion)

min_cost_clusters, max_cost_clusters = graph_kinmatics_each_good(properties_constrained_converged,
                properties_objective_sphere_final_converged,
                properties_objective_continuity_final_converged,
                states_constrained_converged,
                states_objective_sphere_final_converged,
                states_objective_continuity_final_converged,
                idx_clusters_Constrained,
                idx_clusters_sphere_final,
                idx_clusters_continuity_final,
                nb_shooting)

plot_stats(constrained_convergence_rate,
           objective_sphere_initial_convergence_rate,
           objective_sphere_final_convergence_rate,
           objective_continuity_initial_convergence_rate,
           objective_continuity_final_convergence_rate,
           pourcentage_cost_Constrained,
           pourcentage_cost_sphere_final,
           pourcentage_cost_continuity_final,
           pourcentage_transpersion_Constrained,
           pourcentage_transpersion_sphere_final,
           pourcentage_transpersion_continuity_final,
           mean_cluster,
           std_cluster,
           min_cost_clusters,
           max_cost_clusters,
           )

weight_iter_plot(properties_objective_sphere_final_converged,
                 properties_objective_sphere_final,
                 properties_objective_continuity_final_converged,
                 properties_objective_continuity_final,
                 max_transpersion,
                 max_cost,
                 max_time_to_optimize)

plot_marker_trajectory(states_objective_sphere_final_converged, idx_clusters_sphere_final, mean_cluster, min_cost_clusters,
               max_cost_clusters,)


plot_iteration_time(
                    properties_constrained_converged[:, 11],
                    properties_constrained_converged[:, 12],
                    properties_objective_sphere_initial_converged[:, 11],
                    properties_objective_sphere_initial_converged[:, 12],
                    properties_objective_sphere_final_converged[:, 11],
                    properties_objective_sphere_final_converged[:, 12],
                    properties_objective_continuity_initial_converged[:, 11],
                    properties_objective_continuity_initial_converged[:, 12],
                    properties_objective_continuity_final_converged[:, 11],
                    properties_objective_continuity_final_converged[:, 12]
                    )

plot_why_penetrating_sol_high_cost(states_objective_continuity_final_converged,
                                   properties_objective_continuity_final_converged[:, 7],
                                   properties_objective_continuity_final_converged[:, 10],
                                   FLAG_GENERATE_VIDEO)


controls_objective_continuity = controls_all[:, :, np.where(properties_all[:, 0] == "continuity_in_objective")][:, :, 0, :]
controls_objective_continuity_final = controls_objective_continuity[:, :, np.where(properties_objective_continuity[:, 2] == "final")][:, :, 0, :]
controls_objective_continuity_final_converged = controls_objective_continuity_final[:, :, objective_continuity_final_index_converged][:, :, 0, :]

controls_objective_spheres = controls_all[:, :, np.where(properties_all[:, 0] == "spheres_in_objective")][:, :, 0, :]
controls_objective_spheres_final = controls_objective_spheres[:, :, np.where(properties_objective_sphere[:, 2] == "final")][:, :, 0, :]
controls_objective_sphere_final_converged = controls_objective_spheres_final[:, :, objective_sphere_final_index_converged][:, :, 0, :]

why_penetrating_sol_high_cost_metrics(controls_objective_continuity_final_converged,
                                      properties_objective_continuity_final_converged[:, 10],
                                      properties_objective_continuity_final_converged[:, 6],
                                      nb_shooting,
                                      controls_objective_sphere_final_converged,
                                      properties_objective_sphere_final_converged[:, 6],
                                      idx_clusters_sphere_final
                                      )

