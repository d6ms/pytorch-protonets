for local testing

```
$ python protonets.py --train --epochs 3 --episodes 5 --n-train 1 --k-train 3 --q-train 5
```

for training

```
$ nohup python protonets.py --train --n-eval 1 --k-eval 20 > train.log 2>&1 &
```