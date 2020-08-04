for local testing

```
$ python protonets.py --train --epochs 3 --episodes 5 --n-train 1 --k-train 3 --q-train 5
```

for training

```
$ nohup python protonets.py --train --n-train 5 --k-train 60 --q-train 5 --n-eval 5 --k-eval 5 --q-eval 15 > train.log 2>&1 &
```