import logging
import pickle
import time
from IPython import embed
import numpy as np

from bioptim import OdeSolver, Solver, CostType, InitialGuess, InterpolationType

from prepare_ocp import prepare_ocp_constrained, prepare_ocp_unconstrained, prepare_ocp_objective_shpere, prepare_ocp_objective_continuity


def test_unconstrained(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    x_bounds,
    u_bounds,
    x_init,
    u_init,
    i,
    max_iteration_first,
    max_iteration_second,
    weight,
    weight_sphere,
    sol_dir,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    n_threads: int = 1,
    SOLVER_FLAG: str = "IPOPT",
):
    logging.info(f"Running unconstrained test {i} with weight={weight} and weight_sphere={weight_sphere}...")
    # start = time.time()

    ocp = prepare_ocp_unconstrained(
        biorbd_model_path,
        final_time,
        n_shooting,
        x_bounds,
        u_bounds,
        x_init,
        u_init,
        weight,
        weight_sphere,
        ode_solver=ode_solver,
        n_threads=n_threads)

    if SOLVER_FLAG == "IPOPT":
        solver = Solver.IPOPT()
        solver.set_maximum_iterations(max_iteration_first)
    elif SOLVER_FLAG == "SQP_method":
        solver = Solver.SQP_METHOD()
        solver.set_maximum_iterations(max_iteration_first)
        solver.set_qpsol("qrqp")

    sol1 = ocp.solve(solver)
    sol1.detailed_cost_values()

    del sol1.ocp
    sol1.weight = weight
    sol1.weight_sphere = weight_sphere
    sol1.max_iter_first = max_iteration_first
    sol1.max_iter_second = max_iteration_second
    sol1.case = i
    sol1.detailed_cost = sol1.detailed_cost

    # Initial guess
    x_init = np.vstack((sol1.states["q"], sol1.states["qdot"]))
    x_init = InitialGuess(x_init, interpolation=InterpolationType.EACH_FRAME)
    u_init = InitialGuess(sol1.controls["tau"][:, :-1], interpolation=InterpolationType.EACH_FRAME)
    final_time = sol1.parameters["time"][0][0]

    ocp = prepare_ocp_constrained(biorbd_model_path, final_time, n_shooting, x_bounds, u_bounds, x_init, u_init, n_threads=n_threads)

    if SOLVER_FLAG == "IPOPT":
        solver = Solver.IPOPT()
        solver.set_maximum_iterations(max_iteration_second)
    elif SOLVER_FLAG == "SQP_method":
        solver = Solver.SQP_METHOD()
        solver.set_maximum_iterations(max_iteration_second)
        solver.set_qpsol("qrqp")

    sol2 = ocp.solve(solver)
    sol2.detailed_cost_values()
    # stop = time.time()

    del sol2.ocp
    sol2.weight = weight
    sol2.weight_sphere = weight_sphere
    sol2.max_iter_first = max_iteration_first
    sol2.max_iter_second = max_iteration_second
    sol2.case = i
    sol2.total_time = sol1.real_time_to_optimize + sol2.real_time_to_optimize # stop - start
    sol2.detailed_cost = sol2.detailed_cost

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    filename = "unconstrained-initial-{sol_index}-{weight}-{weight_sphere}-{max_iters}-{timestamp}.pickle".format(
        sol_index=i, weight=weight, weight_sphere=weight_sphere, max_iters=max_iteration_first, timestamp=timestamp
    )
    try:
        with open(sol_dir + filename, "wb") as f:
            pickle.dump(sol1, f)
        logging.info(f"Saved solution {i} to '{sol_dir + filename}'.")
    except:
        logging.exception(f"Error when saving solution {i} to '{sol_dir + filename}'.")

    filename = "unconstrained-final-{sol_index}-{weight}-{weight_sphere}-{max_iters}-{timestamp}.pickle".format(
        sol_index=i, weight=weight, weight_sphere=weight_sphere, max_iters=max_iteration_first, timestamp=timestamp
    )
    try:
        with open(sol_dir + filename, "wb") as f:
            pickle.dump(sol2, f)
        logging.info(f"Saved solution {i} to '{sol_dir + filename}'.")
    except:
        logging.exception(f"Error when saving solution {i} to '{sol_dir + filename}'.")
        
        
def test_objective_sphere(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    x_bounds,
    u_bounds,
    x_init,
    u_init,
    i,
    max_iteration_first,
    max_iteration_second,
    weight,
    sol_dir,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    n_threads: int = 1,
	SOLVER_FLAG: str = "IPOPT",
):
    logging.info(f"Running objective test {i} with weight={weight} and max_iteration_first={max_iteration_first}...")
    # start = time.time()
    ocp = prepare_ocp_objective_shpere(
        biorbd_model_path,
        final_time,
        n_shooting,
        x_bounds,
        u_bounds,
        x_init,
        u_init,
        weight,
        ode_solver=ode_solver,
        n_threads=n_threads,
    )
    if SOLVER_FLAG == "IPOPT":
        solver = Solver.IPOPT()
        solver.set_maximum_iterations(max_iteration_first)
    elif SOLVER_FLAG == "SQP_method":
        solver = Solver.SQP_METHOD()
        solver.set_maximum_iterations(max_iteration_first)
        solver.set_qpsol("qrqp")

    sol1 = ocp.solve(solver)
    sol1.detailed_cost_values()

    del sol1.ocp
    sol1.weight = weight
    sol1.max_iter_first = max_iteration_first
    sol1.max_iter_second = max_iteration_second
    sol1.case = i
    sol1.detailed_cost = sol1.detailed_cost

    # Initial guess
    x_init = np.vstack((sol1.states["q"], sol1.states["qdot"]))
    x_init = InitialGuess(x_init, interpolation=InterpolationType.EACH_FRAME)
    u_init = InitialGuess(sol1.controls["tau"][:, :-1], interpolation=InterpolationType.EACH_FRAME)
    final_time = sol1.parameters["time"][0][0]

    ocp = prepare_ocp_constrained(biorbd_model_path, final_time, n_shooting, x_bounds, u_bounds, x_init, u_init, n_threads=n_threads)

    if SOLVER_FLAG == "IPOPT":
        solver = Solver.IPOPT()
        solver.set_maximum_iterations(max_iteration_second)
    elif SOLVER_FLAG == "SQP_method":
        solver = Solver.SQP_METHOD()
        solver.set_maximum_iterations(max_iteration_second)
        solver.set_qpsol("qrqp")

    sol2 = ocp.solve(solver)
    sol2.detailed_cost_values()
    # stop = time.time()

    del sol2.ocp
    sol2.weight = weight
    sol2.max_iter_first = max_iteration_first
    sol2.max_iter_second = max_iteration_second
    sol2.case = i
    sol2.total_time = sol1.real_time_to_optimize + sol2.real_time_to_optimize # stop - start
    sol2.detailed_cost = sol2.detailed_cost

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = "objective_sphere-initial-{sol_index}-{weight}-{max_iters}-{timestamp}.pickle".format(
        sol_index=i, weight=weight, max_iters=max_iteration_first, timestamp=timestamp
    )
    try:
        with open(sol_dir + filename, "wb") as f:
            pickle.dump(sol1, f)
        logging.info(f"Saved solution {i} to '{sol_dir + filename}'.")
    except:
        logging.exception(f"Error when saving solution {i} to '{sol_dir + filename}'.")

    filename = "objective_sphere-final-{sol_index}-{weight}-{max_iters}-{timestamp}.pickle".format(
        sol_index=i, weight=weight, max_iters=max_iteration_first, timestamp=timestamp
    )
    try:
        with open(sol_dir + filename, "wb") as f:
            pickle.dump(sol2, f)
        logging.info(f"Saved solution {i} to '{sol_dir + filename}'.")
    except:
        logging.exception(f"Error when saving solution {i} to '{sol_dir + filename}'.")

        
def test_objective_continuity(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    x_bounds,
    u_bounds,
    x_init,
    u_init,
    i,
    max_iteration_first,
    max_iteration_second,
    weight_sphere,
    sol_dir,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    n_threads: int = 1,
    SOLVER_FLAG: str = "IPOPT",
):
    logging.info(f"Running objective test {i} with weight_sphere={weight_sphere} and max_iteration_first={max_iteration_first}...")
    # start = time.time()
    ocp = prepare_ocp_objective_continuity(
        biorbd_model_path,
        final_time,
        n_shooting,
        x_bounds,
        u_bounds,
        x_init,
        u_init,
        weight_sphere,
        ode_solver=ode_solver,
        n_threads=n_threads,
    )
    if SOLVER_FLAG == "IPOPT":
        solver = Solver.IPOPT()
        solver.set_maximum_iterations(max_iteration_first)
    elif SOLVER_FLAG == "SQP_method":
        solver = Solver.SQP_METHOD()
        solver.set_maximum_iterations(max_iteration_first)
        solver.set_qpsol("qrqp")

    sol1 = ocp.solve(solver)
    sol1.detailed_cost_values()

    del sol1.ocp
    sol1.weight_sphere = weight_sphere
    sol1.max_iter_first = max_iteration_first
    sol1.max_iter_second = max_iteration_second
    sol1.case = i
    sol1.detailed_cost = sol1.detailed_cost

    # Initial guess
    x_init = np.vstack((sol1.states["q"], sol1.states["qdot"]))
    x_init = InitialGuess(x_init, interpolation=InterpolationType.EACH_FRAME)
    u_init = InitialGuess(sol1.controls["tau"][:, :-1], interpolation=InterpolationType.EACH_FRAME)
    final_time = sol1.parameters["time"][0][0]

    ocp = prepare_ocp_constrained(biorbd_model_path, final_time, n_shooting, x_bounds, u_bounds, x_init, u_init, n_threads=n_threads)
    
    if SOLVER_FLAG == "IPOPT":
        solver = Solver.IPOPT()
        solver.set_maximum_iterations(max_iteration_second)
    elif SOLVER_FLAG == "SQP_method":
        solver = Solver.SQP_METHOD()
        solver.set_maximum_iterations(max_iteration_second)
        solver.set_qpsol("qrqp")

    sol2 = ocp.solve(solver)
    sol2.detailed_cost_values()
    # stop = time.time()

    total_real_time_to_optimize_2 = sol2.real_time_to_optimize
    del sol2.ocp
    sol2.weight_sphere = weight_sphere
    sol2.max_iter_first = max_iteration_first
    sol2.max_iter_second = max_iteration_second
    sol2.case = i
    sol2.total_time = sol1.real_time_to_optimize + sol2.real_time_to_optimize # stop - start
    sol2.detailed_cost = sol2.detailed_cost

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = "objective_continuity-initial-{sol_index}-{weight_sphere}-{max_iters}-{timestamp}.pickle".format(
        sol_index=i, weight_sphere=weight_sphere, max_iters=max_iteration_first, timestamp=timestamp
    )
    try:
        with open(sol_dir + filename, "wb") as f:
            pickle.dump(sol1, f)
        logging.info(f"Saved solution {i} to '{sol_dir + filename}'.")
    except:
        logging.exception(f"Error when saving solution {i} to '{sol_dir + filename}'.")

    filename = "objective_continuity-final-{sol_index}-{weight_sphere}-{max_iters}-{timestamp}.pickle".format(
        sol_index=i, weight_sphere=weight_sphere, max_iters=max_iteration_first, timestamp=timestamp
    )
    try:
        with open(sol_dir + filename, "wb") as f:
            pickle.dump(sol2, f)
        logging.info(f"Saved solution {i} to '{sol_dir + filename}'.")
    except:
        logging.exception(f"Error when saving solution {i} to '{sol_dir + filename}'.")
        
def test_constraint(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    x_bounds,
    u_bounds,
    x_init,
    u_init,
    i,
    max_iteration,
    sol_dir,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    n_threads: int = 1,
    SOLVER_FLAG: str = "IPOPT",
):
    logging.info(f"Running constraint test {i}...")
    ocp = prepare_ocp_constrained(
        biorbd_model_path,
        final_time,
        n_shooting,
        x_bounds,
        u_bounds,
        x_init,
        u_init,
        ode_solver=ode_solver,
        n_threads=n_threads,
    )
    if SOLVER_FLAG == "IPOPT":
        solver = Solver.IPOPT()
        solver.set_maximum_iterations(max_iteration)
    elif SOLVER_FLAG == "SQP_method":
        solver = Solver.SQP_METHOD()
        solver.set_maximum_iterations(max_iteration)
        solver.set_qpsol("qrqp")

    sol = ocp.solve(solver)
    sol.detailed_cost_values()

    del sol.ocp
    sol.case = i
    sol.max_iter = max_iteration
    sol.total_time = sol.real_time_to_optimize
    sol.detailed_cost = sol.detailed_cost

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = "constraint-{sol_index}-{timestamp}.pickle".format(sol_index=i, timestamp=timestamp)
    try:
        with open(sol_dir + filename, "wb") as f:
            pickle.dump(sol, f)
        logging.info(f"Saved solution {i} to '{sol_dir + filename}'.")
    except:
        logging.exception(f"Error when saving solution {i} to '{sol_dir + filename}'.")

    logging.info("Done constraint tests.")



