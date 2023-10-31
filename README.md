# Overview
Every delivery company solves the problem of matching couriers and orders. The black-box which takes new couriers and orders sets as input and then assigns one to another is named dispatch. The main purpose of the project is to create a neural network-based dispatch and fine-tune it via reinforcement learning to solve the assignment problem.

The main ideas are described below. Please, read `result.ipynb` to get more information about the approach.

# Steps
I wand to solve the problem in 3 steps:
1. Realize an algorithmic-based dispatch, which minimize the total estimate distance of arrival (ETA) between couriers and orders. This would be a good baseline.
2. Train a network-based dispatch to clone the behaviour of baseline dispatch. We would have a network-based dispatch which solves the problem quite well.
3. Fine-tune our network-based dispatch via reinforcement learning to maximize the number of completed orders.

# Currect results
The main metric, by which our algorithms can be compared is CR (complete rate) - the proportion of success (delivered) orders. At this moment the results are the following:
1. Algorithmic-based dispatch CR: 70%
2. Cloning-training dispatch CR: 68%
3. RL-tuned dispatch CR: 45%

# Description

## The content
1. The simulator
2. Baseline dispatch
3. Neural-network (NN) dispatch
4. Cloning training
5. RL fine tune


## The simulator
### Overview
Training our dispatch to solve the matching problem needs a simulator. The simulator models motion of couriers, clients behaviour (creating and cancalling orders) and e.c.t.

### Details of realization
In fact, the simulator operates with 3 objects: 
- courier (free courier which waits for a new orders)
- order (free (new) order which is not assigned yet)
- active route (an order-courier pair. The order is already in progress)
The state of simulator is a triplet of arrays (free couriers, free orders, active routes) which is called gamble-triple.
Orders are sampled from some distribution (Uniform as a baseline) with some intensity every iteration. Every order has a livetime, after which it would be cancelled if not assigned.
Assigned couriers move with a constant speed in a straight line. Unassigned couriers do not move.

### Files
You can find the realization of simulator in `src/simulator/simulator.py` file.
The realization of order, courier classes can be found in `src/objects` folder.
In `configs/simulator_settings.json` the model parameters can be configurated. For example, the time after which the order is cancelled if not assinged or the speed of our couriers. All experiments are carried out with a single set of settings.

## Baseline dispatch
### Overview
A dispatch is a black-box which takes a simulator state as an input and returns courier-order pairs (assignments) as an output.

### Details
The baseline dispatch which works fine is a Hungarian-dispatch. Free couriers and free orders sets can be represented as a bipartite graph. The weight of the edge between courier and order is considered as a distance between courier and source point of the order. To find the optimal matching on this graph, the hungarian algorithm can be used - it minimized the total weights of chosen edges.

## NN-dispatch
### Overview
The NN-dispatch contains a neural network, which encodes a simulator current state and returns probabilities of assignments (of given order to given courier). 
The detailed description of NN architerture can be found in `result.ipynb`. Here are the main ideas.

### Details
I've started with training a point-encoder on the prombem of distance prediction between 2 arbitary points.
Then, using this point-enocder we can get embeddings of couriers, orders and active-routes.
The encoded gamble-triple is in fact a tiplet of 3 sequences of embeddings - it is an input of our NN.
We want to assign every order of some courier (or no order) - it is a classification problem. That's why we want to get a matrix of probabilities of assignments (of size orders*couriers) as an output.

So, in fact we have a seq2seq problem. Transformers is a good approach to solve such problems: the backbobne of my NN are combined decoder-transformer layers.
Hyperparameters of the model are configured here: `configs/network_hyperparameters.json`

### Files
Encoders can be found in `src/networks` folder. The network realization lives in `src/networks/scoring_networks/net1.py` (see the last version).

## Cloning training
### Overview
The purpose of this step is to train NN-dispatch to clone baseline-dispatch behaviour.
Here we have a supervised learning problem: given an gamble-triple we need to predict the index of courier for every order (or -1 if not assigned). The target is the baseline-dipatch assignments. 

### Details
We will use cross entropy loss for these classification problem. 
Every iteration a random gamble-triple is sampled then the loss on it can be computed. The evaluation would be done on the simulator.

### Files
I've made 2 runs: the main training and the futher training.
Reports can be found here:
1. main: https://api.wandb.ai/links/limon8884/z8bgvw2o
2. futher training: https://api.wandb.ai/links/limon8884/pvfcqcap
Script: `training_cloning.py`
Training settings and hyperparameters are configured here: `configs/training_settings.json`

## RL-finetune
### Overview
The purpose of this step is to beat baseline score by fine-tune a pretrained NN-dispatch via RL.

### Details
I used PPO algorithm. The reward for the action is the number of completed orders on current iteration. The action is the assignments.

### Files
The first success run report: https://api.wandb.ai/links/limon8884/0o6f6e5t
Script: `training_RL.py`
Training settings and hyperparameters are configured here: `configs/rl_settings.json`
