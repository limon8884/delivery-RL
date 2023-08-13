## The problem
Every delivery company solves the problem of matching couriers and orders. The box which takes new couriers and orders sets as input and then assigns one to another is named dispatch. The main idea is to create a neural network-based dispatch and fine-tune it via reinforcement learning to solve the assignment problem.

The main ideas of the project are described below. Please, read `result.ipynb` to get more information about the approach.

## Plan
I wand to solve the problem in 3 steps:
1. Realize an algorithmic-based dispatch, which minimize the total estimate distance of arrival (ETA) between couriers and orders. This would be a good baseline.
2. Train a network-based dispatch to clone the behaviour of baseline dispatch. We would have a network which solves the problem quite well.
3. Fine-tune our network-based dispatch via reinforcement learning to maximize the number of completed orders.

## Currect results
The main metric, by which our algorithms can be compared is CR (complete rate) - the proportion of success (delivered) orders.
1. Algorithmic-based dispatch CR: 70%
2. Cloning-training dispatch CR: 68%
3. RL-tuned dispatch CR: 45%

## The content
1. Simulator of the world
2. Baseline dispatch
3. Neural-network (NN) dispatch architecture
4. RL-environmet
5. Training pipeline

## Description

### The simulator
Training our dispatch to solve the matching problem needs a simulator. The simulator models motion of couriers, clients behaviour (creating and cancalling orders) and e.c.t.
You can find the realization in `src/simulator/simulator.py` file.

In fact, the simulator operates with 3 objects: 
- courier (free courier which waits for a new orders)
- order (free (new) order which is not assigned yet)
- active route (an order-courier pair. The order is already in progress)
The state of simulator is a triplet of arrays (free couriers, free orders, active routes) which is called gamble-triple.
The realization of all these objects can be found in `src/objects` folder.

In `configs/simulator_settings.json` the model parameters can be configurated. For example, the time after which the order is cancelled if not assinged or the speed of our couriers. All experiments are carried out with a single set of settings.

### Baseline dispatch
A dispatch is a black-box which takes a simulator state as an input and returns courier-order pairs (assignments) as an output.

The baseline dispatch which works fine is a Hungarian-dispatch. Free couriers and free orders sets can be represented as a bipartite graph. The weight of the edge between courier and order is considered as a distance between courier and source point of the order. To find the optimal matching on this graph, the hungarian algorithm can be used - it minimized the total weights of chosen edges.

### NN-dispatch
The detailed description of NN architerture can be found in `result.ipynb`. Here are the main ideas.

I've started with training a point-encoder on the prombem of distance prediction between 2 arbitary points.
Then, using this point-enocder we can get embeddings of couriers, orders and active-routes.
The encoded gamble-triple is in fact a tiplet of 3 sequences of embeddings - it is an input of our NN.
We want to assign every order of some courier (or no order) - it is a classification problem. That's why we want to get a matrix of probabilities of assignments (of size orders*couriers) as an output.

So, in fact we have a seq2seq problem. Transformers is a good approach to solve such problems: the backbobne of my NN are combined decoder-transformer layers.


### RL-finetune
I've used a PPO algorithm to fine-tune a given model via RL.
The first success run report: https://api.wandb.ai/links/limon8884/0o6f6e5t
Script: `training_RL.py`

#### Cloning training
Here we have a supervised learning problem: given an gambletriple we need to predict the index of courier for every order (or -1 if not assigned). The target is the baseline-dipatch assignments. We will use cross entropy loss for these classification problem. 

Every iteration a random gamble-triple is sampled then the loss on it can be computed. The evaluation would be done on the simulator.
I've made 2 runs: the main training and the futher training.
Reports can be found here:
1. main: https://api.wandb.ai/links/limon8884/z8bgvw2o
2. futher training: https://api.wandb.ai/links/limon8884/pvfcqcap
Script: `training_cloning.py`
