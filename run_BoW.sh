#!/bin/bash


#python ./BoW.py --path "../data/pkled" --ratio 1.0 --epoch 20

python ./BoW.py --path "../data/pkled" --ratio 0.5 --epoch 20


python ./BoW.py --path "../data/pkled" --ratio 0.25 --epoch 25


python ./BoW.py --path "../data/pkled" --ratio 0.125 --epoch 25


python ./BoW.py --path "../data/pkled" --ratio 0.0625 --epoch 25


python ./BoW.py --path "../data/pkled" --ratio 0.0375 --epoch 25


python ./BoW.py --path "../data/pkled" --ratio 0.01875 --epoch 30
