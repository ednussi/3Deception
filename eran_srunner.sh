#!/bin/bash
srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner.py -i questionnaire/data/lilach/lilach.csv -a top -f all -pm global -lm 4v1_T -MR &
srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner.py -i questionnaire/data/maayan/fs_shapes.1495385783.9178026.csv -a top -f all -pm global -lm 4v1_T -MR &
srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner.py -i questionnaire/data/inbar/fs_shapes.1495363909.5725057.csv -a top -f all -pm global -lm 4v1_T -MR &
srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner.py -i questionnaire/data/asaph/fs_shapes.1495371069.2698493.csv -a top -f all -pm global -lm 4v1_T -MR &
srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner.py -i questionnaire/data/arielt/fs_shapes.1495360021.8733559.csv -a top -f all -pm global -lm 4v1_T -MR &
srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner.py -i questionnaire/data/alon/fs_shapes.1495295074.8250973.csv -a top -f all -pm global -lm 4v1_T -MR &
srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner.py -i questionnaire/data/liav/fs_shapes.1495615047.832466.csv -a top -f all -pm global -lm 4v1_T -MR &
srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner.py -i questionnaire/data/omri/fs_shapes.1495611637.0109568.csv -a top -f all -pm global -lm 4v1_T -MR &
srun -c4 --mem=2048 --time=3-0 /cs/engproj/3deception/grisha/venv/bin/python3 runner.py -i questionnaire/data/yuval/fs_shapes.1495618426.2630248.csv -a top -f all -pm global -lm 4v1_T -MR &
