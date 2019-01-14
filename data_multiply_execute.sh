#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
python data_multiply_experiment.py -c ./config/data_multiply/wine-quality-red.yaml           -ds wine-quality-red            -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm data-multiply -nd &
python data_multiply_experiment.py -c ./config/data_multiply/energy.yaml.yaml                -ds energy                      -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm data-multiply -nd &
python data_multiply_experiment.py -c ./config/data_multiply/bostonHousing.yaml              -ds bostonHousing               -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm data-multiply -nd &

export CUDA_VISIBLE_DEVICES=2
python data_multiply_experiment.py -c ./config/data_multiply/concrete.yaml                   -ds concrete                    -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm data-multiply -nd &
python data_multiply_experiment.py -c ./config/data_multiply/yacht.yaml                      -ds yacht                       -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm data-multiply -nd &

export CUDA_VISIBLE_DEVICES=3
python data_multiply_experiment.py -c ./config/data_multiply/kin8nm.yaml                     -ds kin8nm                      -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm data-multiply -nd &
# python parameter_sweep_regression.py -c ./config/naval-propulsion-plant.yaml     -ds naval-propulsion-plant      -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm sweep-1 &
python data_multiply_experiment.py -c ./config/data_multiply/power-plant.yaml                -ds power-plant                 -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm data-multiply -nd &

export CUDA_VISIBLE_DEVICES=4
python data_multiply_experiment.py -c ./config/data_multiply/protein-tertiary-structure.yaml -ds protein-tertiary-structure  -ld /scratch/mjh252/logs/pytorch -dd /scratch/mjh252/data/UCL/ -cm data-multiply -nd &
#python experiment_parametrise_classification2.py -c ./config/mnist.yaml -ds mnist -ld /scratch/mjh252/logs  -dd /scratch/mjh252/data/UCL

wait
echo "All Finished"