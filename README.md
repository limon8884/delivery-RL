### This is my master thesis work. For more information, please, look at `Thesis.pdf`
# Overview
Last-mile delivery is a rapidly evolving field. Every company specializing in
this area faces the challenge of assigning couriers to customer orders. The main
question is how to make these assignments most efficient.

In this thesis, I formally posed this problem and, using real-world simulations,
explored the possibilities of solving it with neural networks via reinforcement
learning. The resulting algorithm is compared with several heuristic algorithms
to evaluate its performance relative to a baseline.

This work has several outcomes. Firstly, I implemented an infrastructure
that allows for simulating couriers and orders; I developed the neural network
architecture; and I implemented an algorithm that trains models using reinforcement
learning. Secondly, I analyzed the model’s performance based on various parameters.
I tested different reward functions, approaches to working with geo coordinates,
and model sizes. Thirdly, I proved several theoretical statements that apply to
various similar problems. Specifically, I proved the optimality of one algorithm
under certain conditions and proposed a method for dealing with certain types of
sparse rewards.

The current conclusion of this work is that for the basic problem of assignment,
heuristic algorithms that rely on the distances between couriers and orders are
quite effective. Neural networks can learn to solve these problems at a similar
level but have not yet succeeded in improving the results.

Thus, this work provides only an initial understanding of the problem and
its possible solutions, leaving several questions for future research. For example,
the presented model did not cover the possibility of adding a new order to an
existing one, although the necessary functionality is implemented in the code.
This could be one of the areas where reinforcement learning might demonstrate
higher performance compared to heuristics.
