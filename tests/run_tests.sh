#!/bin/sh

for ((idx_random=0; idx_random<100; idx_random=idx_random+1)); do

    # run tests mixed_sphere - constrained
    weights=(1000 100000 1000000)
    max_iters=(100 1000 10000)
    for weight in ${weights[*]}; do
        for max_iter in ${max_iters[*]}; do
            python3 obstacle_work_around.py objective_sphere $idx_random --iters1 ${max_iter} --iters2 10000 --weight ${weight} --solver IPOPT
        done
    done

    # run tests mixed_continuity - constrained
    weight_spheres=(1000 100000 1000000)
    max_iters=(100 1000 10000)
    for weight_sphere in ${weight_spheres[*]}; do
        for max_iter in ${max_iters[*]}; do
            python3 obstacle_work_around.py objective_continuity $idx_random --iters1 ${max_iter} --iters2 10000 --weight_sphere ${weight_sphere} --solver IPOPT
        done
    done

    # run constrained tests
    python3 obstacle_work_around.py constraint  $idx_random --iters1 10000 --solver IPOPT

done


