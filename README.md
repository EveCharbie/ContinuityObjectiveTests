This respository contains the code used to assess the two optimization step work-flow which will be presented at ECCOMAS 2023 and in the related conference paper entitled "Warm-starting multi-start procedure using penalties instead of constraints to find more optimal trajectories" (DOI to come). 

## Two optimization step work-flow

1. 100 noized initial guess where generated
2. The underconstrained OCP is solved (replacing either the continuity or the obstacles with penalty terms in the cost function)
3. The fully constrained OCP is solved with the solution from 2. as initial guess.

## Results
The proposed work-flow allowed to jump over the constraint barriers allowing the generation of more globally optimal solutions that a random intialization.

If you have any questions, please contact eve.charbonneau.1@umontreal.ca
