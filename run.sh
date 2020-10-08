#!/bin/bash

rm -r /home/golf/code/data/CY101NPY
python data/make_data.py
python train.py --ratio 1.0 --epochs 5
python train.py --ratio 0.5 --epochs 10
python train.py --ratio 0.25 --epochs 20
python train.py --ratio 0.125 --epochs 50
python train.py --ratio 0.0625 --epochs 100
python train.py --ratio 0.03125 --epochs 100
python train.py --ratio 0.015625 --epochs 200


rm -r /home/golf/code/data/CY101NPY
python data/make_data.py
python train.py --ratio 1.0 --epochs 5
python train.py --ratio 0.5 --epochs 10
python train.py --ratio 0.25 --epochs 20
python train.py --ratio 0.125 --epochs 50
python train.py --ratio 0.0625 --epochs 100
python train.py --ratio 0.03125 --epochs 100
python train.py --ratio 0.015625 --epochs 200


rm -r /home/golf/code/data/CY101NPY
python data/make_data.py
python train.py --ratio 1.0 --epochs 5
python train.py --ratio 0.5 --epochs 10
python train.py --ratio 0.25 --epochs 20
python train.py --ratio 0.125 --epochs 50
python train.py --ratio 0.0625 --epochs 100
python train.py --ratio 0.03125 --epochs 100
python train.py --ratio 0.015625 --epochs 200


rm -r /home/golf/code/data/CY101NPY
python data/make_data.py
python train.py --ratio 1.0 --epochs 5
python train.py --ratio 0.5 --epochs 10
python train.py --ratio 0.25 --epochs 20
python train.py --ratio 0.125 --epochs 50
python train.py --ratio 0.0625 --epochs 100
python train.py --ratio 0.03125 --epochs 100
python train.py --ratio 0.015625 --epochs 200


rm -r /home/golf/code/data/CY101NPY
python data/make_data.py
python train.py --ratio 1.0 --epochs 5
python train.py --ratio 0.5 --epochs 10
python train.py --ratio 0.25 --epochs 20
python train.py --ratio 0.125 --epochs 50
python train.py --ratio 0.0625 --epochs 100
python train.py --ratio 0.03125 --epochs 100
python train.py --ratio 0.015625 --epochs 200
