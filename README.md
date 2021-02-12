# NPS_ICML

## Installation
Our code was tested with python 3.6
Use `pip install -r requirements.txt` to install all dependencies.


## Arithmetic Task
Run the following command from the `synthetic` folder.
```
sh runner.sh num_rules rule_emb_dim embed_dim seed

num_rules: Number of rules to use. Should always be 3 as there are 3 operations: {addition, subtraction, multiplication}.
rule_emb_dim: Rule embedding dimension.
embed_dim: Dimension to which the numbers are encoded to.

```

To reproduce the experiments in the paper:
```
sh runner.sh 3 32 64 3
```

Expected output:

```
0 : {'addition': 16640, 'subtraction': 0, 'multiplication': 0}
2 : {'addition': 0, 'subtraction': 0, 'multiplication': 16501}
1 : {'addition': 0, 'subtraction': 16856, 'multiplication': 0}
```
Here we can see that the rules have been completely segregated. rule number 0 is solely used for the addition operation, rule number 1 is solely used for the multiplication operation, and  rule number 1 is solely used for the subtraction operation. You should also see a best eval mse of about:
```
best_eval_mse:tensor(0.0005, device='cuda:0')
```
Depending on the seed as well as the environment in which the code is run, the best eval mse may vary but should be somewhere around the above number. Below we show best eval mse for 3 different seeds:
```
(1) best_eval_mse:tensor(0.0005, device='cuda:0')
(2) best_eval_mse:tensor(0.0004, device='cuda:0')
(3) best_eval_mse:tensor(0.0006, device='cuda:0')
```

To evalaute the model on multiple sequence lengths:
```
sh eval_runner.sh num_rules rule_emb_dim embed_dim seed
```
This will run the saved model corresponding to the args provided.

To train and evaluate the model presented in the paper run:
```
sh runner.sh 3 32 64 3
sh eval_runner.sh 3 32 64 3
```

The output of eval_runner.sh should be as follows:
```
FINAL RESULTS ACROSS VARIOUS SEQUENCE LENGTHS
	SEQUENCE LENGTH	|	MSE
	      10        |	0.0003
	      20        |	0.0005
	      30        |	0.0008
	      40        |	0.0014
	      50        |	0.0018
```

## MNIST Transformation Task

Run the following command from the `MNIST` folder.

```
sh run.sh seed
```
Expected output:
After a few epochs a complete segregation of rules should be observed in the following manner:
```
rotate_left : {0: 0, 1: 0, 2: 0, 3: 4981}
translate_up : {0: 0, 1: 4950, 2: 0, 3: 0}
rotate_right : {0: 0, 1: 0, 2: 5030, 3: 0}
translate_down : {0: 5039, 1: 0, 2: 0, 3: 0}

```

The above snippet indicates that rule number 3 is solely being used for the rotate left transformation, rule number 1 is being used for translate up, rule number 2 is being used for rotate right, and rule number 0 is being used for translate down. Hence a complete segregation is observed.

