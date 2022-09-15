#!/bin/sh


# run constraint tests DONE
for ((idx_random=0; idx_random<100; idx_random=idx_random+1)); do
    python3 obstacle_work_around.py constraint  $idx_random --iters1 10000
done

# run objective tests set weight
max_iters=(100 1000 10000)
for ((idx_random=0; idx_random<100; idx_random=idx_random+1)); do
    for max_iter in ${max_iters[*]}; do
        python3 obstacle_work_around.py objective $idx_random --var varit --iters1 ${max_iter} --iters2 10000 --weight 1000000
    done
done

# run objective tests set iters
weights=(1000 1000000 1000000000)
for ((idx_random=0; idx_random<100; idx_random=idx_random+1)); do
    for weight in ${weights[*]}; do
        python3 obstacle_work_around.py objective $idx_random --var varpoids --iters1 10000 --iters2 10000 --weight ${weight}
    done
done

# run objective tests optimal weight and iter
for ((idx_random=0; idx_random<100; idx_random=idx_random+1)); do
    python3 obstacle_work_around.py objective $idx_random --var varopt --iters1 1000 --iters2 10000 --weight 1000000000
done

# run tests without constraints for shpere/contionuity weights
weights=(1000 1000000 1000000000)
weight_spheres=(1000 1000000 1000000000)
for ((idx_random=0; idx_random<100; idx_random=idx_random+1)); do
    for weight in ${weights[*]}; do
        for weight_sphere in ${weight_spheres[*]}; do
            python3 obstacle_work_around.py unconstrained $idx_random --var varsphere --iters1 10000 --iter2 10000 --weight ${weight} --weight_sphere ${weight_sphere}
        done
    done
done