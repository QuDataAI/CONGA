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
solver : conga
 idx pred:  value (dv)                   weigth (dw)         [epoch] | true: value      weight            n   t_bst t_tot  |  file
   0    563647.00 (        0 = 0.0%)          0 (     49877) [  626] |      563647,       49877,      10000  3.030s 6.535s |  knapPI_1_10000_1000_1
   1     54503.00 (        0 = 0.0%)          0 (      5002) [   70] |       54503,        5002,       1000  0.054s 1.271s |  knapPI_1_1000_1000_1
   2      9147.00 (        0 = 0.0%)        -10 (      1005) [   28] |        9147,         995,        100  0.019s 1.244s |  knapPI_1_100_1000_1
   3    110625.00 (        0 = 0.0%)          0 (     10011) [  124] |      110625,       10011,       2000  0.089s 1.360s |  knapPI_1_2000_1000_1
   4     11238.00 (        0 = 0.0%)        -21 (      1029) [   30] |       11238,        1008,        200  0.021s 1.286s |  knapPI_1_200_1000_1
   5    276457.00 (        0 = 0.0%)          0 (     25016) [  484] |      276457,       25016,       5000  0.688s 2.821s |  knapPI_1_5000_1000_1
   6     28857.00 (        0 = 0.0%)          0 (      2543) [   33] |       28857,        2543,        500  0.024s 1.291s |  knapPI_1_500_1000_1
   7     90202.00 (        2 = 0.0%)          0 (     49877) [ 1510] |       90204,       49877,      10000  3.856s 5.107s |  knapPI_2_10000_1000_1
   8      9052.00 (        0 = 0.0%)          0 (      5002) [  559] |        9052,        5002,       1000  0.370s 1.285s |  knapPI_2_1000_1000_1
   9      1513.00 (        1 = 0.1%)        -33 (      1028) [   81] |        1514,         995,        100  0.052s 1.263s |  knapPI_2_100_1000_1
  10     18051.00 (        0 = 0.0%)         -1 (     10012) [  695] |       18051,       10011,       2000  0.485s 1.385s |  knapPI_2_2000_1000_1
  11      1634.00 (        0 = 0.0%)         -2 (      1010) [  334] |        1634,        1008,        200  0.224s 1.283s |  knapPI_2_200_1000_1
  12     44355.00 (        1 = 0.0%)          0 (     25016) [ 1196] |       44356,       25016,       5000  1.691s 2.825s |  knapPI_2_5000_1000_1
  13      4566.00 (        0 = 0.0%)          0 (      2543) [  273] |        4566,        2543,        500  0.181s 1.344s |  knapPI_2_500_1000_1
  14    146919.00 (        0 = 0.0%)          0 (     49519) [  346] |      146919,       49519,      10000  0.893s 5.102s |  knapPI_3_10000_1000_1
  15     14390.00 (        0 = 0.0%)          0 (      4990) [  205] |       14390,        4990,       1000  0.142s 1.283s |  knapPI_3_1000_1000_1
  16      2397.00 (        0 = 0.0%)          0 (       997) [   76] |        2397,         997,        100  0.056s 1.427s |  knapPI_3_100_1000_1
  17     28919.00 (        0 = 0.0%)          0 (      9819) [  253] |       28919,        9819,       2000  0.223s 1.599s |  knapPI_3_2000_1000_1
  18      2697.00 (        0 = 0.0%)          0 (       997) [  139] |        2697,         997,        200  0.105s 1.401s |  knapPI_3_200_1000_1
  19     72505.00 (        0 = 0.0%)          0 (     24805) [  329] |       72505,       24805,       5000  0.475s 2.838s |  knapPI_3_5000_1000_1
  20      7117.00 (        0 = 0.0%)          0 (      2517) [  167] |        7117,        2517,        500  0.128s 1.412s |  knapPI_3_500_1000_1
avr err: 0.0034%  bst epoch: avr = 360 max = 1510
```