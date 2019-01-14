#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
python parameter_sweep_regression.py -c ./config/parameter_sweep_2/wine-quality-red.yaml           -ds wine-quality-red            -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm sweep-prior-var -nd &
python parameter_sweep_regression.py -c ./config/parameter_sweep_2/bostonHousing.yaml              -ds bostonHousing               -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm sweep-prior-var -nd &
#python parameter_sweep_regression.py -c ./config/parameter_sweep_1/energy.yaml                     -ds energy                      -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm sweep-2-prior-var &

export CUDA_VISIBLE_DEVICES=2
python parameter_sweep_regression.py -c ./config/parameter_sweep_2/concrete.yaml                   -ds concrete                    -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm sweep-prior-var -nd &
python parameter_sweep_regression.py -c ./config/parameter_sweep_2/yacht.yaml                      -ds yacht                       -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm sweep-prior-var -nd &

export CUDA_VISIBLE_DEVICES=3
python parameter_sweep_regression.py -c ./config/parameter_sweep_2/kin8nm.yaml                     -ds kin8nm                      -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm sweep-prior-var -nd &
# python parameter_sweep_regression.py -c ./config/naval-propulsion-plant.yaml     -ds naval-propulsion-plant      -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm sweep-1 &
#python parameter_sweep_regression.py -c ./config/parameter_sweep_2/power-plant.yaml                -ds power-plant                 -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm sweep-prior-var -nd &

export CUDA_VISIBLE_DEVICES=4
python parameter_sweep_regression.py -c ./config/parameter_sweep_2/protein-tertiary-structure.yaml -ds protein-tertiary-structure  -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm sweep-prior-var -nd &
#python experiment_parametrise_classification2.py -c ./config/mnist.yaml -ds mnist -ld /scratch/mjh252/logs  -dd /scratch/mjh252/data/UCL
wait
echo "All Finished"