# delivery-RL
I will try to use RL approachs to solve a dispath assignment promlem.
Please, read `result.ipynb` to get more information about the project.

## The problem
Every delivery company solves a problem of matching couriers and orders. The box which take new couriers and orders as input and then assign one to another is named dispatch.
The main idea is to create neural-network-based dispatch and the fine-tune it via reinforcement learning.

## The plan
I wand to solve the problem in 3 steps:
1. Realize an hungarian-algorithm-based dispatch, which minimize the total estimate distance of arrival (ETA) between couriers and orders. This would be a good baseline
2. Train a network-base dispatch to clone the behaviour of baseline-dispatch. We would have a network which solves the problem quite well.
3. Fine-tune our network-based dispatch via reinforcement learning to maximize the number of completed orders

## The content
1. Simulator of the world
2. Baseline dispatch
3. Neural-network (NN) dispatch architecture
4. RL-environmet
5. Training pipeline

## Description

### The simulator
To train our dispatch to solve matching task we need a simulator. The simulator models motion of couriers, creating and cancelling orders by clients e.c.t.
You can find in in `src/simulator/simulator.py` file.

In fact, the simulator operates with 3 objects: 
- courier (free courier which waits for new orders)
- order (free (new) order which is not assigned yet)
- active route (an order-courier pair. The order is in progress)
The state of simulator is a triplet of arrays (free couriers, free orders, active routes) which is called gamble-triple.
The realization of all these objects can be found in `src/objects` folder.

In `configs/simulator_settings.json` the model parameters can be configurated. For example, the time after which the order is cancelled if not assinged or the speed of our couriers.

### Baseline dispatch
A dispatch is a black-box which takes a simulator state as an input and returns courier-order pairs (assignments) as the output.

The baseline dispatch which works fine is a Hungarian-dispatch. Free couriers and free orders can be represented as a bipartite graph. The weight between courier and order is set to the distance between courier and source point of the order. To find the optimal matching on this graph, the hungarian algorithm can be used - it minimized the total weights of chosen edges.

### NN-dispatch
TBD

### RL-environment
TBD

### Training pipeline
The main metric, by which our algorithms can be compared is CR (complete rate) - the proportion of success (delivered) orders.

#### Cloning training
Here we have a supervised learning problem: given an gambletriple we need to predict the index of courier for every order (or -1 if not assigned). The target is the baseline-dipatch assignments. We will use cross entropy loss for these classification problem. 

Every iteration a random gable-triple is sampled then the loss on it can be computed. The evaluation would be done on the simulator.
I've made 2 runs: the main training and the futher training.
Reports can be found here:
1. main: https://api.wandb.ai/links/limon8884/z8bgvw2o
2. posttraining: https://api.wandb.ai/links/limon8884/pvfcqcap


