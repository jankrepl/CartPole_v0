# CartPole_v0
5 different techniques are used to solved the control problem of balancing a pole on a cart. Environment is taken from OpenAI gym
CartPole_v0.

* **Random Search** (Generic optimization)
* **Hill Climbing** (Generic optimization)
* **Q-learning** (Reinforcement learning)
* **Sarsa** (Reinforcement learning)
* **Monte Carlo** (Reinforcement learning)


## Code

### Running
```
python Main.py
```

### Dependencies
*  gym
*  itertools
*  math
*  matplotlib
*  networkx
*  numpy

## Detailed Description
### Problem Statement and Environment
The goal is to move the cart to the left and right in a way that the pole on top of it does not fall down. The states 
of the environment are composed of 4 elements - **cart position** (x), **cart speed** (xdot),
**pole angle** (theta) and **pole angular velocity** (thetadot). For each time step when the pole is still on the cart
we get a reward of 1. The problem is considered to be solved if for 100 consecutive
episodes the average reward is at least 195.


If we translate this problem into reinforcement learning terminology:
* action space is **0** (left) and **1** (right)
* state space is a set of all 4-element lists with all possible combinations of values of x, xdot, theta, thetadot

---
### Solution #1 - Random Search
The policy/action will be simply determined by the following decision rule:

```python
if w_0 * x + w_1 * xdot + w_2 * theta + w_3 * thetadot + b > 0: 
  return 1
else:
  return 0
```

Each episode we randomly generate the weights **w** = (w_0, w_1, w_2, w_3, b) and use the above decision rule to
move the cart. If we stumble upon a winner (**w** that leads to 200 steps) then we save it for later. From certain episode,
we switch to a "greedy mode" and only pick randomly the previous winning policies. If a policy wins again, we keep it in the 
winners list. If it does not, we remove it.



### Solution #2 - Hill Climbing
We use the same decision rule as in Random Search but instead of generating a new **w** each episode, we simply
wait until a good enough policy is found ( > 150 timesteps) and then apply the following logic until a winner is found.

```python
while reward(w_old) < 200
  w_new = w_old + random_noise
  if reward(w_new) > reward(w_old):
    w_old = w_new
  else:
    pass
```


---
For the following 3 methods it is necessary to discretize the sample space. To do this we select for each
element a list of threshold points and based on these we discretize the continuous observations. A sufficient discretization
for all three algorithms is for example the following:

```python
    threshold_points = [[1, 0, 1],  # x
                        [-0.5, 0, 0.5],  # xdot
                        [-math.radians(5), 0, math.radians(5)],  # theta
                        [-math.radians(5), 0, math.radians(5)]]  # thetadot
                      
```
It results in a sample space of 256 states.

### Solution #3 - Q learning
The Q value function update is the following and can be done online without the need to wait until the end of episode.

```python
Q(S_t,A_t) += alpha * (R_t + discount_factor * max Q(S_{t+1}, a) - Q(S_t,A_t))

```


### Solution #4 - Sarsa
The Sarsa update has the following form:
```python
Q(S_t,A_t) += alpha * (R_t + discount_factor * Q(S_{t+1}, A_{t+1}) - Q(S_t,A_t))

```


### Solution #5 - Monte Carlo

Monte Carlo approximates the Q function with the sample mean of observed returns
```
Q(S,A) = np.mean([observed returns starting with an state-action value (S,A)])
```
## Examples and evaluations
* [Random Search](https://gym.openai.com/evaluations/eval_UAQHInPMQR2cWVNJfmSwTg)
* [Hill Climbing](https://gym.openai.com/evaluations/eval_OyTa0n1vRUSqAPLJXrSBfA)
* [Q-learning](https://gym.openai.com/evaluations/eval_gT2ZDu1BT0iCB2LQkAJIlQ)
* [Sarsa](https://gym.openai.com/evaluations/eval_Yj0ikfI5Tw2K33LNATJezQ)
* [Monte Carlo](https://gym.openai.com/evaluations/eval_2QlLOTqlQgyAAWphHkGTQ)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Links & Resources
* https://gym.openai.com/envs/CartPole-v0
* http://kvfrans.com/simple-algoritms-for-solving-cartpole/ - amazing blog post about Random Search and Hill Climbing
