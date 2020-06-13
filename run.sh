#!/bin/bash

python ./train.py --ratio 1.0 --epoch 5
python ./train.py --pdbf --fibo 1 --ratio 1.0 --epoch 5


python ./train.py --ratio 0.5  --epoch 10
python ./train.py --pdbf --fibo 1 --ratio 0.5 --epoch 10


python ./train.py --ratio 0.25  --epoch 15
python ./train.py --pdbf --fibo 1 --ratio 0.25 --epoch 15


python ./train.py --ratio 0.125 --epoch 15
python ./train.py --pdbf --fibo 1 --ratio 0.125 --epoch 15


python ./train.py --ratio 0.0625  --epoch 20
python ./train.py --pdbf --fibo 1 --ratio 0.0625 --epoch 20


python ./train.py --ratio 0.0375  --epoch 25
python ./train.py --pdbf --fibo 1 --ratio 0.0375 --epoch 25


python ./train.py --ratio 0.01875  --epoch 30
python ./train.py --pdbf --fibo 1 --ratio 0.01875 --epoch 30
