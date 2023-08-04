# delivery-RL
### Main idea
I will try to use RL approachs to solve a dispath assignment promlem.
Please, read `result.ipynb` to get more information of the work.
### Description
Every delivery company solves a problem of matching couriers and orders. The box which take new couriers and orders as input and then assign one to another is named dispatch.
The main idea is to create neural-network-based dispatch and the fine-tune it via reinforcement learning.
### Plan
1. Write a simulator and a default algorithm-based dispatch (without any NNs)
2. Create a NN architecture which is able to match couriers and orders somehow
3. Train in supervised way NN to clone the default dispatch behavior
4. Finetune with reinforcement learning NN to solve the problem better
5. Improve simulator with real data 
6. Make a production version of NN-based dispatch



