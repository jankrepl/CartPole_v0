"""
Implementation of 5 different techniques to solve the OpenAI CartPole_v0 problem.
    - Random Search 
    - Hill Climbing
    - Qlearning
    - Sarsa
    - Monte Carlo


"""

__author__ = "Jan Krepl"
__email__ = "jankrepl@yahoo.com"
__license__ = "MIT"


import math
import gym
from gym import wrappers
from foo import *


# MODEL OVERVIEW
model_dictionary = {0: 'Random Search', 1: 'Hill Climbing', 2: 'Qlearning', 3: 'Sarsa',
                    4: 'MC'}  # IDs of available models

# PARAMETERS
# Environment
number_of_episodes = 3000

# Recording
bool_record = False
output_folder = 'path'

# Submission
bool_submit = False
API_KEY = 'api_key'

# Results and visualizations
bool_visualize = True
my_results = Results()

# Model Choice
model_id = 2 # see model_dictionary above
my_solver = None  # Just initialized - defined below

# Model Specific Parameters
if model_id == 0:
    # Random Search
    lower_bound = -2
    upper_bound = 2
    greedy_switch = 2000

    my_solver = RandomSearch(number_of_episodes, lower_bound, upper_bound, greedy_switch)

elif model_id == 1:
    # Hill Climbing
    noise_lower_bound = -1
    noise_upper_bound = 1
    minimal_starting_return = 180  # we want to guarantee a good start - otherwise random weights

    my_solver = HillClimbing(number_of_episodes, noise_lower_bound, noise_upper_bound, minimal_starting_return)

elif model_id == 2:
    # Q Learning
    discount_rate = 1  # gamma
    threshold_points = [[-2.4, 1, 0, 1, 2.4],  # x
                        [-0.5, 0, 0.5],  # xdot
                        [-math.radians(15), -math.radians(6), 0, math.radians(6), math.radians(15)],  # theta
                        [-math.radians(5), 0, math.radians(5)]]  # thetadot
    initial_q_value = 0
    epsilon = 1
    epsilon_decay = (0.001 / epsilon) ** (1 / (number_of_episodes - 1000))
    alpha = 0.5  # How seriously we take the next episode

    my_solver = Qlearning(number_of_episodes, threshold_points, initial_q_value, discount_rate, epsilon, epsilon_decay,
                          alpha)
    print(len(my_solver.state_space))

elif model_id == 3:
    # SARSA
    discount_rate = 1  # gamma
    threshold_points = [[1, 0, 1],  # x
                        [-0.5, 0, 0.5],  # xdot
                        [-math.radians(5), 0, math.radians(5)],  # theta
                        [-math.radians(5), 0, math.radians(5)]]  # thetadot

    initial_q_value = 0
    epsilon = 1
    epsilon_decay = (0.001 / epsilon) ** (1 / (number_of_episodes - 800))
    alpha = 0.3  # How seriously we take the next episode

    my_solver = Sarsa(number_of_episodes, threshold_points, initial_q_value, discount_rate, epsilon, epsilon_decay,
                      alpha)

elif model_id == 4:
    # MC
    epsilon = 1
    epsilon_decay = (0.01 / epsilon) ** (1 / (number_of_episodes - 200))
    # ideally we would like to spend last approx. 200 episodes just exploiting
    # epsilon * decay ^ (eps - 200) = 0.01 -> decay = (0.01/epsilon)^(1/(eps - 200))

    initial_q_value = 0  # should be 0
    threshold_points = [[1, 0, 1],  # x
                        [-0.5, 0, 0.5],  # xdot
                        [-math.radians(5), 0, math.radians(5)],  # theta
                        [-math.radians(5), 0, math.radians(5)]]  # thetadot

    my_solver = MC(number_of_episodes, threshold_points, initial_q_value, epsilon, epsilon_decay)

# ALGORITHM
env = gym.make('CartPole-v0')

if bool_record:
    env = wrappers.Monitor(env, output_folder)

for i_episode in range(number_of_episodes):
    observation_old = env.reset()

    for t in range(201):
        # Render
        env.render()

        # Choose action
        action = my_solver.choose_action(observation_old)

        # Take action
        observation_new, reward, done, info = env.step(action)

        # Feed SAR into database
        my_solver.feed_new_SAR(i_episode, observation_old, action, reward, done)

        # End of timestep update
        my_solver.end_of_timestep_update(i_episode, t, done)

        # Move forward
        observation_old = observation_new

        # check end
        if done:
            print('In the ' + str(i_episode + 1) + '-th episode there were ' + str(t + 1) + ' timesteps')
            my_results.add_result(t)
            break

    # End of episode update
    my_solver.end_of_episode_update(i_episode)


env.close()

# UPLOAD SCORE
if bool_submit:
    gym.upload(output_folder, api_key=API_KEY)

# Visualize results
if bool_visualize:
    my_results.compute_empirical_quntiles(my_solver.episode_database, [0.25, 0.5, 0.75])
    my_results.plot_evolution()
