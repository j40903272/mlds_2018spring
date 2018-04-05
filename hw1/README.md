# MLDS hw1

## 1-1

### Simulate a Function

#### Train
```bash
# function 1:
python3 1-1-1_model1.py
python3 1-1-1_model2.py
python3 1-1-1_model3.py
python3 1-1-1_ground_truth.py
# function 2:
python3 1-1-1_bonus_model1.py
python3 1-1-1_bonus_model2.py
python3 1-1-1_bonus_model3.py
```

#### Plot
```bash
# function 1:
python3 1-1-1_plot.py
# function 2:
python3 1-1-1_bonus_plot.py
```

### Train on Actual Tasks

#### Train
```bash
# CIFAR10
python3 cifar10_hidden1.py
python3 cifar10_hidden2.py
python3 cifar10_hidden4.py
# MNIST
python3 mnist_hidden1.py
python3 mnist_hidden2.py
python3 mnist_hidden4.py
```

#### Plot
```bash
python3 visualize_cifar10.py
python3 visualize_mnsit.py
```

## 1-2

### Visualize the optimization process
```bash
python3 visualize_optimization.py
```

### Observe gradient norm during training
```bash
python3 observe_gradient_norm.py
```

### What happens when gradient is almost zero?
```bash
python3 gradient_almost_zero.py
```

### Bonus
```bash
python3 bonus.py
```

## 1-3

### Can network fit random variables?
```bash
python3 1-3-1_1.py
python3 1-3-1_2.py
```

### Number of parameters v.s. Generalization

### Flatness v.s. Generalization

#### train
```bash
python3 batch_32.py
python3 batch_64.py
python3 batch_128.py
python3 batch_512.py
python3 batch_1024.py
python3 batch_2048.py
python3 batch_4096.py
```
#### part 1
```bash
python3 1-3-3_part1.py
```
#### part 2
```bash
python3 1-3-3_part2.py
```
