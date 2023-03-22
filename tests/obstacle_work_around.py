import argparse
import logging

import biorbd
from bioptim import Bounds, QAndQDotBounds, NoisedInitialGuess
import numpy as np

from tests import test_constraint, test_objective_sphere, test_objective_continuity, test_unconstrained

DEBUG_FLAG = False

if DEBUG_FLAG:
    class Arguments:
        def __init__(self):
            self.type = "constraint"
            self.idx_random = 0
            self.iters1 = 1
            self.iters2 = 10
            self.weight = 1000000
            self.weight_sphere = 100
            self.solver = "IPOPT"

    args = Arguments()

else:
    logging.basicConfig(
        filename="ObstacleWorkAround.log", level=logging.INFO, format="%(asctime)s:%(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser("Run test for a particular case.")
    parser.add_argument("type", action="store", help="type of test (constraint or objective)")
    parser.add_argument("idx_random", action="store", type=int, help="number of the test")
    parser.add_argument(
        "--iters1", action="store", required=True, type=int, help="maximum number of iterations allowed on first pass"
    )
    parser.add_argument(
        "--iters2", action="store", required=False, type=int, help="maximum number of iterations allowed on second pass"
    )
    parser.add_argument("--weight", action="store", required=False, type=float, help="weight of continuity objective")
    parser.add_argument("--weight_sphere", action="store", required=False, type=float, help="weight of sphere collision objective")
    parser.add_argument("--solver", action="store", required=False, type=str,
                        help="solver being used (SQP_method or IPOPT)")

    args = parser.parse_args()

def prepare_x_bounds(biorbd_model):
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, 0] = 0
    x_bounds[0, 2] = 0 ### might be removed
    x_bounds[1, 2] = np.pi

    return x_bounds


def prepare_u_bounds(biorbd_model):
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max = -300, 300
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively rotating

    return u_bounds


seed = 42
np.random.seed(seed)

biorbd_model_path = "../models/pendulum_maze.bioMod"

biorbd_model = biorbd.Model(biorbd_model_path)
nb_q = biorbd_model.nbQ()
nb_qdot = biorbd_model.nbQdot()
nb_tau = biorbd_model.nbGeneralizedTorque()

n_shooting = 500
final_time = 5
n_threads = 1

x_bounds = prepare_x_bounds(biorbd_model)
u_bounds = prepare_u_bounds(biorbd_model)

x_inits = [
    NoisedInitialGuess([0] * (nb_q + nb_qdot), bounds=x_bounds, noise_magnitude=0.001, n_shooting=n_shooting)
    for _ in range(100)
]
u_inits = [
    NoisedInitialGuess([0] * nb_tau, bounds=u_bounds, noise_magnitude=0.01, n_shooting=n_shooting - 1)
    for _ in range(100)
]

if args.type == "unconstrained":
    logging.info(
        f"Testing all objectives (no contraints) "
        f"seed={seed} "
        f"final_time={final_time} "
        f"n_shooting={n_shooting} "
        f"iters1={args.iters1} "
        f"weight={args.weight} "
        f"weight_sphere={args.weight_sphere} "
        f"n_threads={n_threads}"
        f"solver={args.solver}..."

    )

    if args.solver == "IPOPT":
        sol_dir = "../solutions_IPOPT/"
    if args.solver == "SQP_method":
        sol_dir = "../solutions_SQPmethod/"

    test_unconstrained(
        biorbd_model_path,
        final_time,
        n_shooting,
        x_bounds,
        u_bounds,
        x_inits[args.idx_random],
        u_inits[args.idx_random],
        args.idx_random,
        args.iters1,
        args.iters2,
        args.weight,
        args.weight_sphere,
        sol_dir,
        n_threads=n_threads,
        SOLVER_FLAG=args.solver,
    )

    logging.info("Done, Good Bye!")

elif args.type == "objective_sphere":
    logging.info(
        f"Testing continuity as objective, sphere as constraints "
        f"seed={seed} "
        f"final_time={final_time} "
        f"n_shooting={n_shooting} "
        f"iters1={args.iters1} "
        f"iters2={args.iters2} "
        f"weight={args.weight} "
        f"n_threads={n_threads}"
        f"solver={args.solver}..."

    )

    if args.solver == "IPOPT":
        sol_dir = "../solutions_IPOPT/"
    if args.solver == "SQP_method":
        sol_dir = "../solutions_SQPmethod/"

    test_objective_sphere(
        biorbd_model_path,
        final_time,
        n_shooting,
        x_bounds,
        u_bounds,
        x_inits[args.idx_random],
        u_inits[args.idx_random],
        args.idx_random,
        args.iters1,
        args.iters2,
        args.weight,
        sol_dir,
        n_threads=n_threads,
        SOLVER_FLAG=args.solver,
        )

    logging.info("Done, Good Bye!")

elif args.type == "objective_continuity":
    logging.info(
        f"Testing continuity as objective, sphere as constraints "
        f"seed={seed} "
        f"final_time={final_time} "
        f"n_shooting={n_shooting} "
        f"iters1={args.iters1} "
        f"iters2={args.iters2} "
        f"weight_sphere={args.weight_sphere} "
        f"n_threads={n_threads}"
        f"solver={args.solver}..."

    )

    if args.solver == "IPOPT":
        sol_dir = "../solutions_IPOPT/"
    if args.solver == "SQP_method":
        sol_dir = "../solutions_SQPmethod/"

    test_objective_continuity(
        biorbd_model_path,
        final_time,
        n_shooting,
        x_bounds,
        u_bounds,
        x_inits[args.idx_random],
        u_inits[args.idx_random],
        args.idx_random,
        args.iters1,
        args.iters2,
        args.weight_sphere,
        sol_dir,
        n_threads=n_threads,
        SOLVER_FLAG=args.solver,
        )

    logging.info("Done, Good Bye!")

elif args.type == "constraint":
    logging.info(
        f"Testing continuity constraint "
        f"seed={seed} "
        f"final_time={final_time} "
        f"n_shooting={n_shooting} "
        f"iters1={args.iters1} "
        f"n_threads={n_threads}"
        f"solver={args.solver}..."

    )

    if args.solver == "IPOPT":
        sol_dir = "../solutions_IPOPT/"
    if args.solver == "SQP_method":
        sol_dir = "../solutions_SQPmethod/"

    test_constraint(
        biorbd_model_path,
        final_time,
        n_shooting,
        x_bounds,
        u_bounds,
        x_inits[args.idx_random],
        u_inits[args.idx_random],
        args.idx_random,
        args.iters1,
        sol_dir,
        n_threads=n_threads,
        SOLVER_FLAG=args.solver,
    )

    logging.info("Done, Good Bye!")

else:
    print("Invalid type.")
