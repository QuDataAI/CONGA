# CONGA

Implementation of CONGA algorithm to solve 0/1 Knapsack problem tasks

## Installation

Istall torch with https://pytorch.org/get-started/locally/

Clone CONGA git repo:
```sh
git clone https://github.com/QuDataAI/CONGA
```

## Solve single task 

Parameters:
```sh
usage: src/solve.py [-h] [--solver SOLVER] --values VALUES
                             [VALUES ...] --weights WEIGHTS [WEIGHTS ...]
                             [--capacity CAPACITY]

optional arguments:
  -h, --help            show this help message and exit
  --solver SOLVER       Method for resolve task
  --values VALUES [VALUES ...]
                        List of item values
  --weights WEIGHTS [WEIGHTS ...]
                        List of item weights
  --capacity CAPACITY   Maximum capacity of knapsack
```

Example:
```sh
python src/solve.py  --values 9 11 13 15 --weights 6 5 9 7 --capacity 20
```

Output:
```sh
best value: 35.0
```

## Solve tasks from instances dataset 

Parameters:
```sh
usage: src/test_dataset.py [-h] [--solver SOLVER] [--path PATH]

optional arguments:
  -h, --help       show this help message and exit
  --solver SOLVER  Method for resolve task
  --path PATH      Path to dataset

```
Example:
```sh
python src/test_dataset.py --path data/instances_01_KP/large_scale
```

Output:
```sh
solver : CONGA
 idx pred:  value (dv)                   weigth (dw)         [epoch] | true: value      weight            n     cor t_bst  t_tot  |  file
   0    563647.00 (        0 = 0.0%)          0 (     49877) [  988] |      563647,       49877,      10000  -0.018 2.048s 12.166s |  knapPI_1_10000_1000_1
   1     54503.00 (        0 = 0.0%)          0 (      5002) [   56] |       54503,        5002,       1000   0.024 0.072s 7.377s |  knapPI_1_1000_1000_1
   2      9147.00 (        0 = 0.0%)        -10 (      1005) [   28] |        9147,         995,        100  -0.042 0.037s 7.851s |  knapPI_1_100_1000_1
   3    110625.00 (        0 = 0.0%)          0 (     10011) [  184] |      110625,       10011,       2000  -0.003 0.246s 7.593s |  knapPI_1_2000_1000_1
   4     11238.00 (        0 = 0.0%)        -21 (      1029) [   26] |       11238,        1008,        200   0.083 0.042s 7.719s |  knapPI_1_200_1000_1
   5    276457.00 (        0 = 0.0%)          0 (     25016) [  521] |      276457,       25016,       5000  -0.018 0.693s 7.862s |  knapPI_1_5000_1000_1
   6     28857.00 (        0 = 0.0%)          0 (      2543) [ 2045] |       28857,        2543,        500   0.005 2.562s 7.409s |  knapPI_1_500_1000_1
   7     90204.00 (        0 = 0.0%)          0 (     49877) [ 1697] |       90204,       49877,      10000   0.981 3.486s 12.179s |  knapPI_2_10000_1000_1
   8      9052.00 (        0 = 0.0%)          0 (      5002) [  619] |        9052,        5002,       1000   0.979 0.784s 7.797s |  knapPI_2_1000_1000_1
   9      1514.00 (        0 = 0.0%)         -4 (       999) [ 2075] |        1514,         995,        100   0.981 2.550s 7.324s |  knapPI_2_100_1000_1
  10     18051.00 (        0 = 0.0%)         -1 (     10012) [  646] |       18051,       10011,       2000   0.980 0.811s 7.770s |  knapPI_2_2000_1000_1
  11      1634.00 (        0 = 0.0%)         -2 (      1010) [  173] |        1634,        1008,        200   0.981 0.252s 7.366s |  knapPI_2_200_1000_1
  12     44356.00 (        0 = 0.0%)          0 (     25016) [ 1461] |       44356,       25016,       5000   0.980 2.002s 7.851s |  knapPI_2_5000_1000_1
  13      4566.00 (        0 = 0.0%)          0 (      2543) [  362] |        4566,        2543,        500   0.979 0.479s 7.546s |  knapPI_2_500_1000_1
  14    146919.00 (        0 = 0.0%)          0 (     49519) [  358] |      146919,       49519,      10000   1.000 0.791s 12.163s |  knapPI_3_10000_1000_1
  15     14390.00 (        0 = 0.0%)          0 (      4990) [  173] |       14390,        4990,       1000   1.000 0.239s 7.782s |  knapPI_3_1000_1000_1
  16      2397.00 (        0 = 0.0%)          0 (       997) [   71] |        2397,         997,        100   1.000 0.112s 7.683s |  knapPI_3_100_1000_1
  17     28919.00 (        0 = 0.0%)          0 (      9819) [  201] |       28919,        9819,       2000   1.000 0.298s 7.712s |  knapPI_3_2000_1000_1
  18      2697.00 (        0 = 0.0%)          0 (       997) [  106] |        2697,         997,        200   1.000 0.141s 7.864s |  knapPI_3_200_1000_1
  19     72505.00 (        0 = 0.0%)          0 (     24805) [  300] |       72505,       24805,       5000   1.000 0.414s 7.406s |  knapPI_3_5000_1000_1
  20      7117.00 (        0 = 0.0%)          0 (      2517) [  139] |        7117,        2517,        500   1.000 0.182s 7.925s |  knapPI_3_500_1000_1
avr err: 0.0000%  bst epoch: avr = 582 max = 2075
```
