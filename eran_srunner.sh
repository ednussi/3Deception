#!/bin/bash
srun -c4 --mem=2048 --time=3-0 python3 runner.py -i 3deception-data/lilach/lilach.csv -a top -f all -pm global -lm 4v1_A -n regular -MR &
srun -c4 --mem=2048 --time=3-0 python3 runner.py -i 3deception-data/lilach/lilach.csv -a top -f all -pm global -lm 4v1_A -n zero-one -MR &
srun -c4 --mem=2048 --time=3-0 python3 runner.py -i 3deception-data/lilach/lilach.csv -a top -f all -pm global -lm 4v1_A -n one-one -MR &
srun -c4 --mem=2048 --time=3-0 python3 runner.py -i 3deception-data/lilach/lilach.csv -a top -f all -pm global -lm 4v1_A -n l1 -MR &
srun -c4 --mem=2048 --time=3-0 python3 runner.py -i 3deception-data/lilach/lilach.csv -a top -f all -pm global -lm 4v1_A -n l2 -MR &
srun -c4 --mem=2048 --time=3-0 python3 runner.py -i 3deception-data/lilach/lilach.csv -a top -f all -pm global -lm 4v1_A -n max -MR &