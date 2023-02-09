Introduction
---
The source code for paper **USDNL: Uncertainty-based Single Dropout in Noisy Label Learning**

How to use
---
The code is currently trained only on GPU and contains three different noise types of CIFAR dataset: `[CCN, OOD, IDN]`. You can specify the 
noise construction of noise dataset through `args.noise`.

- Demo
  - If you want to train our model in IDN CIFAR-10 dataset, you can modify `arg.noise`, `args.noise_type`, `args.noise_rate`, 
    `args.dataset`, and then run
      ```
      python main.py --args.noise idn --args.noise_type instance --args.noise_rate 0.2 --args.dataset cifar10
      ```