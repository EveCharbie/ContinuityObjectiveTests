"""
TODO: General cleaning
An optimal control program consisting in a pendulum starting downward and ending upward while requiring
the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

There is a catch however: there are regions in which the weight of the pendulum cannot go.
The problem is solved in two passes. In the first pass, continuity is an objective rather then a constraint.
The goal of the first pass is to find quickly find a good initial guess. This initial guess is then given
to the second pass in which continuity is a constraint to find the optimal solution.

During the optimization process, the graphs are updated real-time. Finally, once it finished optimizing, it animates
the model using the optimal solution.
"""

import casadi as cas
import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    OptimalControlProgram,
    Node,
    DynamicsFcn,
    Dynamics,
    InterpolationType,
    InitialGuess,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintFcn,
    ConstraintList,
    OdeSolver,
    BiorbdInterface,
    Solution,
)


def out_of_sphere_constraint(all_pn, y, z):
    q = all_pn.nlp.states["q"].mx
    marker_q = all_pn.nlp.model.markers(q)[1].to_mx()

    distance = cas.sqrt((y - marker_q[1]) ** 2 + (z - marker_q[2]) ** 2)

    return BiorbdInterface.mx_to_cx("out_of_sphere_constraint", distance, all_pn.nlp.states["q"])

def out_of_sphere_objective(all_pn, y, z, limit):
    q = all_pn.nlp.states["q"].mx
    marker_q = all_pn.nlp.model.markers(q)[1].to_mx()

    distance = cas.if_else(cas.sqrt((y - marker_q[1]) ** 2 + (z - marker_q[2]) ** 2) - limit < 0,
                           (cas.sqrt((y - marker_q[1]) ** 2 + (z - marker_q[2]) ** 2) - limit)**2,
                           0)

    return BiorbdInterface.mx_to_cx("out_of_sphere_constraint", distance, all_pn.nlp.states["q"])


def prepare_ocp_unconstrained(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    x_bounds,
    u_bounds,
    x_init,
    u_init,
    weight,
    weight_sphere,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1, # 32,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolver = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=1, key="tau")
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, max_bound=2)
    objective_functions.add(out_of_sphere_objective, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=weight_sphere, quadratic=True, y=0.05, z=0, limit=0.35)
    objective_functions.add(out_of_sphere_objective, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=weight_sphere, quadratic=True, y=0.55, z=-0.85, limit=0.35)
    objective_functions.add(out_of_sphere_objective, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=weight_sphere, quadratic=True, y=0.75, z=0.2, limit=0.35)
    objective_functions.add(out_of_sphere_objective, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=weight_sphere, quadratic=True, y=-0.45, z=0, limit=0.35,)
    objective_functions.add(out_of_sphere_objective, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=weight_sphere, quadratic=True, y=1.4, z=0.5, limit=0.35)
    objective_functions.add(out_of_sphere_objective, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=weight_sphere, quadratic=True, y=2, z=1.2, limit=0.35)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
        state_continuity_weight=weight,
    )


def prepare_ocp_objective_shpere(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    x_bounds,
    u_bounds,
    x_init,
    u_init,
    weight,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1, # 32,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolver = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=1, key="tau")
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, max_bound=2)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="marker_2", second_marker="target_2")
    constraints.add(out_of_sphere_constraint, y=0.05, z=0, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere_constraint, y=0.55, z=-0.85, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere_constraint, y=0.75, z=0.2, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere_constraint, y=-0.45, z=0, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere_constraint, y=1.4, z=0.5, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere_constraint, y=2, z=1.2, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
        state_continuity_weight=weight,
    )

def prepare_ocp_objective_continuity(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    x_bounds,
    u_bounds,
    x_init,
    u_init,
    weight_sphere,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1, # 32,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolver = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=1, key="tau")
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, max_bound=2)
    objective_functions.add(out_of_sphere_objective, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=weight_sphere, quadratic=True, y=0.05, z=0, limit=0.35)
    objective_functions.add(out_of_sphere_objective, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=weight_sphere, quadratic=True, y=0.55, z=-0.85, limit=0.35)
    objective_functions.add(out_of_sphere_objective, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=weight_sphere, quadratic=True, y=0.75, z=0.2, limit=0.35)
    objective_functions.add(out_of_sphere_objective, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=weight_sphere, quadratic=True, y=-0.45, z=0, limit=0.35,)
    objective_functions.add(out_of_sphere_objective, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=weight_sphere, quadratic=True, y=1.4, z=0.5, limit=0.35)
    objective_functions.add(out_of_sphere_objective, custom_type=ObjectiveFcn.Mayer, node=Node.ALL, weight=weight_sphere, quadratic=True, y=2, z=1.2, limit=0.35)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def prepare_ocp_constrained(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    x_bounds,
    u_bounds,
    x_init,
    u_init,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1, # 32,
) -> OptimalControlProgram:

    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    ode_solver: OdeSolver = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=1, key="tau")
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, max_bound=2)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="marker_2", second_marker="target_2")
    constraints.add(out_of_sphere_constraint, y=0.05, z=0, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere_constraint, y=0.55, z=-0.85, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere_constraint, y=0.75, z=0.2, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere_constraint, y=-0.45, z=0, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere_constraint, y=1.4, z=0.5, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere_constraint, y=2, z=1.2, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
    )
