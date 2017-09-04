import itertools
import matplotlib.pyplot as plt
import numpy as np


class Solver:
    """
    General Solver serves a parent to specific techniques
    """

    def __init__(self, number_of_episodes):
        """
        Constructor of a parent class of all solution methods. In all algorithms, we take into account the number
        of episodes. For some strategies it is also necessary to use sequences of all SAR in a given episode -
        we save into episode_database

        :param number_of_episodes: total number of episodes
        :type number_of_episodes: int
        """

        self.number_of_episodes = number_of_episodes
        self.episode_database = [[] for i in range(number_of_episodes)]  # list of lists of tuples

        return

    def feed_new_SAR(self, episode, observation, action, reward, done):
        """
        Updates episode_database with a new SAR. For each episode there will be t_episode tuples (SAR) at the end.
        Note that the last observation S is on purpose not logged

        :param episode: The number of current episode
        :type episode: int
        :param observation: A list of 4 observations [cart_position, cart_speed, pole_position, pole_speed]
        :type observation: list
        :param action: 0 or 1
        :type action: int
        :param reward: 0 or 1 ... if S A lead to reward done=True then it is 0, otherwise 1
        :type reward: int
        """

        # If unhappy about the fact that last action gives reward 1, you can use the code below. But it makes sense
        # if done:
        #     reward = 0
        self.episode_database[episode].append(tuple((observation, action, reward)))
        # Examples:
        #   episode_database[3][10] = ([x,xdot,y,ydot],0,1) -> 3rd episode, 10th SAR
        return

    def choose_action(self, *argv):
        """
        Every child should override this since its the policy:D

        :return:
        :rtype:
        """
        pass
        return

    def end_of_timestep_update(self, *argv):
        """
        It is supposed to be either overridden by the child or do no action. Since different children have
        different implementations it is important to keep the number of arguments variable

        """

        pass
        return

    def end_of_episode_update(self, *argv):
        """
        It is supposed to be either overridden by the child or do no action. Since different children have
        different implementations it is important to keep the number of arguments variable

        """

        pass
        return


class RandomSearch(Solver):
    """
    Random Search Solver class that is a child to the Solver Class.

    Method description: For each episode, we sample weights randomly and then let the simulation run. If we
                        find a winner (200 timesteps without interruption) then we save it into a winner database.
                        For episodes after greedy switch we just create a tournament among the winning strategies
                        where we pick them randomly and require them to win again otherwise we delete them.

    """

    def __init__(self, number_of_episodes, lower_bound, upper_bound, greedy_switch):
        """
        Constructor that define all relevant variables/attributes

        :param number_of_episodes: overall number of episodes the aglorithm is running for
        :type number_of_episodes: int
        :param lower_bound: lower bound on all 5 weights - (w_0, w_1, w_2, w_3, b)
        :type lower_bound: float
        :param upper_bound: upper bound on all 5 weights - (w_0, w_1, w_2, w_3, b)
        :type upper_bound: float
        :param greedy_switch: an episode number from which onwards we will only use previous winning episodes
        :type greedy_switch: int
        """
        super(RandomSearch, self).__init__(number_of_episodes)  # Initialize parent

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.w_episode = (self.upper_bound - self.lower_bound) * np.random.rand(4, ) + self.lower_bound  # U(lb,ub)
        self.b_episode = (self.upper_bound - self.lower_bound) * np.random.rand() + self.lower_bound  # U(lb,ub)

        self.greedy_switch = greedy_switch
        self.winners_database = []  # database of strategies that managed to get 200 timesteps

    def choose_action(self, observation):
        """
        We use a simple decision criterion w_0 * o_0 + w_1 * o_1 +  w_2 * o_2 + w_3 * o_3 + b) > 0

        :param observation: current observation
        :type observation:
        :return: 0 or 1 - represents a movement of the cart to left or right
        :rtype: int
        """
        if np.inner(self.w_episode, observation) + self.b_episode > 0:
            return 1
        else:
            return 0

    def end_of_episode_update(self, episode):
        """
        Update performed at the end of each episode. We first check whether the current weights won the game and
        if yes we save them. Then we determine whether we are in exploration or exploitation mode.
        If exploration, we generate new weights randomly. If in exploitation, we run a random winning strategy
        and delete it from winner database. We keep doing this until there is only one winner left or the simulation
        ends (all episodes played out)

        :param episode: current episode
        :type episode: int
        """

        # Check if it is a winning strategy
        return_episode = len(self.episode_database[episode]) - 1
        if return_episode == 199:
            self.winners_database.append(tuple((self.w_episode, self.b_episode)))

        # Check if exploration or exploitation
        if episode >= self.greedy_switch:

            # Check how many winners are left -
            assert len(self.winners_database) > 0, 'No winners were found in the training phase'

            if len(self.winners_database) == 1:
                # There is only one winner left - we just run all the remaining episodes with this strategy
                print('We have identified the best strategy')

            else:
                # we simply delete a given strategy from winners and let it run in in the next episode
                print('There are ' + str(len(self.winners_database)) + ' possible winners')

                i = np.random.randint(0, len(self.winners_database))

                self.w_episode = self.winners_database[i][0]
                self.b_episode = self.winners_database[i][1]

                del self.winners_database[i]

        else:
            # Generate new random weights
            self.w_episode = (self.upper_bound - self.lower_bound) * np.random.rand(4, ) + self.lower_bound
            self.b_episode = (self.upper_bound - self.lower_bound) * np.random.rand() + self.lower_bound

        return


class HillClimbing(Solver):
    """
    Hill Climbing Solver class is a child to the Solver Class

    Method description: Initially, we are just trying random solutions until the minimal_starting_return requirement is
    met. When that happens we just keep the best weights and randomly add noise to them, if that improves the return we
    stick with them if not we go back to the previous ones.

    """

    def __init__(self, number_of_episodes, noise_lower_bound, noise_upper_bound, minimal_starting_return):
        super(HillClimbing, self).__init__(number_of_episodes)  # Initialize parent

        self.noise_lower_bound = noise_lower_bound
        self.noise_upper_bound = noise_upper_bound
        self.minimal_starting_return = minimal_starting_return

        self.w_episode = (self.noise_upper_bound - self.noise_lower_bound) * np.random.rand(
            4, ) + self.noise_lower_bound  # U(lb,ub)
        self.b_episode = (
                             self.noise_upper_bound - self.noise_lower_bound) * np.random.rand() + self.noise_lower_bound  # U(lb,ub)

        self.w_prev_episode = np.array([0, 0, 0, 0])
        self.b_prev_episode = 0
        self.return_prev = 0  # Avoid local maxima

    def choose_action(self, observation):
        """
        We use a simple decision criterion w_0 * o_0 + w_1 * o_1 +  w_2 * o_2 + w_3 * o_3 + b) > 0

        :param observation: current observation
        :type observation:
        :return: 0 or 1 - represents a movement of the cart to left or right
        :rtype: int
        """
        if np.inner(self.w_episode, observation) + self.b_episode > 0:
            return 1
        else:
            return 0

    def end_of_episode_update(self, episode):
        # Define important variables
        return_this_episode = len(self.episode_database[episode]) - 1
        initial_mode_bool = self.return_prev < self.minimal_starting_return and return_this_episode < self.minimal_starting_return
        new_high_bool = self.return_prev < return_this_episode

        # Check whether in initial mode - we haven't surpassed the self.minimal_starting_return yet
        if initial_mode_bool:
            # We are in the initial mode - goal is to avoid bad starting values
            # -> just generate new weights randomly
            self.w_episode = (self.noise_upper_bound - self.noise_lower_bound) * np.random.rand(
                4, ) + self.noise_lower_bound
            self.b_episode = (
                                 self.noise_upper_bound - self.noise_lower_bound) * np.random.rand() + self.noise_lower_bound
            return

        if self.return_prev < return_this_episode:
            # We found a new HIGH !!!!!!
            # Make current weights and return a benchmark for the next episodes
            self.return_prev = return_this_episode
            self.w_prev_episode = self.w_episode
            self.b_prev_episode = self.b_episode

            # Now check if a winner - if yes, we keep the same weight, if not we add random noise

            if return_this_episode == 199:
                # it is a winner - keep the same weights
                pass
            else:
                # it is the best result so far, but not a winner
                self.w_episode += (self.noise_upper_bound - self.noise_lower_bound) * np.random.rand(
                    4, ) + self.noise_lower_bound
                self.b_episode += (
                                      self.noise_upper_bound - self.noise_lower_bound) * np.random.rand() + self.noise_lower_bound
        else:
            # We did not beat our previous  best - lets return back to the previous best
            self.w_episode = self.w_prev_episode
            self.b_episode = self.b_prev_episode

        return


class Qlearning(Solver):
    def __init__(self, number_of_episodes, threshold_points, initial_q_value, discount_rate, epsilon, epsilon_decay,
                 alpha):
        """
        Constructor of the Q Learning solver

        :param number_of_episodes: number of episodes
        :type number_of_episodes: int
        :param threshold_points: points that divide each observation element into buckets/classes
        :type threshold_points: list of lists
        :param initial_q_value: entire q function is populated with this value at initialization
        :type initial_q_value: float
        :param discount_rate: in (0,1] - how we discount tomorrows value
        :type discount_rate: float
        :param epsilon: starting exploration parameter - on average epsilon * 100 % of time we explore,
        :type epsilon: float
        :param epsilon_decay: At the end of each each episode we put epsilon *= epsilon_decay -> we will end
                            up exploiting more in later periods
        :type epsilon_decay: float
        :param alpha: learning rate
        :type alpha: float
        """
        super(Qlearning, self).__init__(number_of_episodes)  # Initialize parent

        # Input attributes
        self.discount_rate = discount_rate
        self.threshold_points = threshold_points
        self.initial_q_value = initial_q_value
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha

        # Induced attributes
        self.state_space = self.__generate_state_space(self.threshold_points)  # list of 4-tuples - not really needed
        self.Q = self.__generate_Q(self.threshold_points, self.initial_q_value)  # 5 dimensional ndarray

    def __generate_state_space(self, threshold_points):
        """
        Generates all possible state spaces (4-tuples)

        :param threshold_points: points that devide each observation element into buckets/classes
        :type threshold_points: list of lists
        :return: list of all possible states <- state space
        :rtype: list of tuples
        """
        # Step 1 - For each state variable define possible outcomes - variable outcome array
        a_0 = np.arange(0, len(threshold_points[0]) + 1)
        a_1 = np.arange(0, len(threshold_points[1]) + 1)
        a_2 = np.arange(0, len(threshold_points[2]) + 1)
        a_3 = np.arange(0, len(threshold_points[3]) + 1)

        # Step 2 - Define a state space as a cartesian product of the 4 variable outcomes
        return list(itertools.product(a_0, a_1, a_2, a_3))

    def __generate_Q(self, threshold_points, initial_q_value):
        """
        Generates a Q functions as a ndarray

        :param threshold_points: point that partition state space for each observation element
        :type threshold_points: list of lists
        :param initial_q_value: the value we populate q function at initialization of the algorithm
        :type initial_q_value: float
        :return: Q function represented as a 5d matrix ... first 4 coordinates are for observations and 5th is the action
        :rtype: ndarray
        """
        return initial_q_value * np.ones((len(threshold_points[0]) + 1, len(threshold_points[1]) + 1,
                                          len(threshold_points[2]) + 1, len(threshold_points[3]) + 1, 2))

    def __discretize_observation(self, continuous_observation):
        '''
        Inputs a continuous observation and assigns to it the discretized version based on the threshold points

        :param continuous_observation: the 4 element observation generated by the environment
        :type continuous_observation: ndarray
        :return: dicretized 4-element observation
        :rtype: ndarray
        '''
        output = [0, 0, 0, 0]

        for element in range(4):
            order = 0
            last = True

            for points in self.threshold_points[element]:
                if continuous_observation[element] < points:
                    last = False
                    output[element] = order
                    break

                else:
                    order += 1

            if last:
                output[element] = len(self.threshold_points[element])

        return output  # it can be just list since we will convert it to tuple anyway

    def choose_action(self, observation):
        """
        Given an observation, we perform our action - POLICY

        :param observation: continuous observation
        :type observation: list
        :return: 1 or 0
        :rtype: int
        """
        # map observation to a discrete version
        disc_obs = self.__discretize_observation(observation)

        # decide whether exploration or exploitation mode

        explore_mode = np.random.rand() < self.epsilon

        if self.Q[tuple(disc_obs + [0])] > self.Q[tuple(disc_obs + [1])]:
            if explore_mode:
                return 1
            else:
                return 0
        elif self.Q[tuple(disc_obs + [0])] < self.Q[tuple(disc_obs + [1])]:
            if explore_mode:
                return 0
            else:
                return 1
        else:
            return np.random.choice([0, 1], 1)[0]

    def end_of_timestep_update(self, i_episode, timestep, done):
        if timestep == 0:
            return
        S_old, A_old, R = self.episode_database[i_episode][timestep - 1]

        S_new, A_new, _ = self.episode_database[i_episode][timestep]

        S_old_disc = self.__discretize_observation(S_old)
        S_new_disc = self.__discretize_observation(S_new)

        q_max = max([self.Q[tuple(S_new_disc + [a])] for a in range(2)])

        # UPDATE STEP
        self.Q[tuple(S_old_disc + [A_old])] += self.alpha * (
            R + self.discount_rate * q_max - self.Q[tuple(S_old_disc + [A_old])])

        # EXTREMELY IMPORTANT
        if done and timestep < 199:
            # we have to set the reward for the last-terminal observation manually, since the above update can sometimes
            # adjusts it -> WHY????
            # The same continuous state sometimes leads to done and sometimes not -> Because we discretized our
            # sample space - so it is really important to just manually make sure that its 0

            # Maybe a more elegant solution is to select threshold points such that the bins correspond
            # To the actual values that lead to failure (x /notin (-2.4,2.4) and theta /notin ()

            # It works like an induction, these terminal states will backpropagate to the earlier states
            print('AA')
            self.Q[tuple(S_new_disc + [A_new])] = 0

    def end_of_episode_update(self, *args):
        """
        Update of epsilon (greedy parameter) based on epsilon_decay

        :param args: just an empty argument
        :type args: none

        """
        self.epsilon *= self.epsilon_decay
        print('Epsilon is: ' + str(self.epsilon))

        # print('The epsilon is ' + str(self.epsilon))

    def print_Q(self):
        """
        Prints Q value of all states in the state space

        :return:
        :rtype:
        """
        actions = [0, 1]

        for s in self.state_space:
            for a in actions:
                print('State ' + str(list(s) + [a]) + ' has a q value of ' + str(self.Q[tuple(list(s) + [a])]))

    def print_highest_q_values(self, n):
        """
        Print n highest q values and the corresponding states

        :param n: number of values to print
        :type n: int


        """

        # Print n highest action values

        argument_space = [tuple(list(o) + [a]) for o in self.state_space for a in [0, 1]]
        value_space = [self.Q[arg] for arg in argument_space]

        I = sorted(range(len(argument_space)), key=lambda k: value_space[k])

        for i in range(1, n + 1):
            print('State ' + str(argument_space[I[-i]]) + ' has a q value of ' + str(value_space[I[-i]]))


class MC(Solver):
    """
    Monte Carlo solver. We keep a running database of simulated returns for all state-action pairs while evaluating
    a policy (initialize with random policy). At the end of the episode we redefine Q values and the policy we follow
    (epsilon greedy). We iterate this process.
    """

    def __init__(self, number_of_episodes, threshold_points, initial_q_value, epsilon, epsilon_decay):
        """
        Constructor of the Monte Carlo Solver

        :param number_of_episodes: number of episodes
        :type number_of_episodes: int
        :param threshold_points: points that divide each observation element into buckets/classes
        :type threshold_points: list of lists
        :param initial_q_value: entire q function is populated with this value at initialization
        :type initial_q_value: float
        :param epsilon: starting exploration parameter - on average epsilon * 100 % of time we explore,
        :type epsilon: float
        :param epsilon_decay: At the end of each each episode we put epsilon *= epsilon_decay -> we will end
                            up exploiting more in later periods
        :type epsilon_decay: float
        """
        super(MC, self).__init__(number_of_episodes)  # Initialize parent

        # Input attributes
        self.threshold_points = threshold_points
        self.initial_q_value = initial_q_value
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Induced attributes
        self.state_space = self.__generate_state_space(self.threshold_points)  # list of 4-tuples - not really needed
        self.Q = self.__generate_Q(self.threshold_points, self.initial_q_value)  # 5 dimensional ndarray
        self.return_database = self.__initialize_return_database()  # dict (state, action) -> list(all returns so far)

    def __generate_state_space(self, threshold_points):
        """
        Generates all possible state spaces (4-tuples)

        :param threshold_points: points that devide each observation element into buckets/classes
        :type threshold_points: list of lists
        :return: list of all possible states <- state space
        :rtype: list of tuples
        """
        # Step 1 - For each state variable define possible outcomes - variable outcome array
        a_0 = np.arange(0, len(threshold_points[0]) + 1)
        a_1 = np.arange(0, len(threshold_points[1]) + 1)
        a_2 = np.arange(0, len(threshold_points[2]) + 1)
        a_3 = np.arange(0, len(threshold_points[3]) + 1)

        print("The state space has " + str(len(a_0) * len(a_1) * len(a_2) * len(a_3)) + " elements")
        # Step 2 - Define a state space as a cartesian product of the 4 variable outcomes
        return list(itertools.product(a_0, a_1, a_2, a_3))

    def __generate_Q(self, threshold_points, initial_q_value):
        """
        Generates a Q functions as a ndarray

        :param threshold_points: point that partition state space for each observation element
        :type threshold_points: list of lists
        :param initial_q_value: the value we populate q function at initialization of the algorithm
        :type initial_q_value: float
        :return: Q function represented as a 5d matrix ... first 4 coordinates are for observations and 5th is the action
        :rtype: ndarray
        """
        return initial_q_value * np.ones((len(threshold_points[0]) + 1, len(threshold_points[1]) + 1,
                                          len(threshold_points[2]) + 1, len(threshold_points[3]) + 1, 2))

    def __initialize_return_database(self):
        """
        Initializing a return database - to each state-action pair we assign a an empty list and save this
        data structure as a dictionary

        :return: list of all returns for a given state-action pair
        :rtype: dict - tuple to list
        """
        return {tuple(list(s) + [a]): [] for s in self.state_space for a in [0, 1]}

    def __discretize_observation(self, continuous_observation):
        '''
        Inputs a continuous observation and assigns to it the discretized version based on the threshold points

        :param continuous_observation: the 4 element observation generated by the environment
        :type continuous_observation: ndarray
        :return: dicretized 4-element observation
        :rtype: ndarray
        '''
        output = [0, 0, 0, 0]

        for element in range(4):
            order = 0
            last = True

            for points in self.threshold_points[element]:
                if continuous_observation[element] < points:
                    last = False
                    output[element] = order
                    break

                else:
                    order += 1

            if last:
                output[element] = len(self.threshold_points[element])

        return output  # it can be just list since we will convert it to tuple anyway

    def choose_action(self, observation):
        """
        Simple epsilon greedy

        :param observation:
        :type observation:

        """
        # since we initialize Q with uniform value, the first episode the policy is just random

        # map observation to a discrete version
        disc_obs = self.__discretize_observation(observation)

        # decide whether exploration or exploitation mode

        explore_mode = np.random.rand() < self.epsilon

        if explore_mode:
            # EXPLORE MODE - just choose randomly
            # print('explore')
            return np.random.choice([0, 1], 1)[0]

        else:
            # print('EXPLOIT')
            # EXPLOIT MODE - choose the action with higher Q ... if equal select randomly
            if self.Q[tuple(disc_obs + [0])] > self.Q[tuple(disc_obs + [1])]:
                return 0
            elif self.Q[tuple(disc_obs + [0])] < self.Q[tuple(disc_obs + [1])]:
                return 1
            else:
                return np.random.choice([0, 1], 1)[0]

    def end_of_episode_update(self, episode, *argv):
        """
        4 majors things :
        1) Check whether state-action pair encountered for the first time
        2) Add a generated return for a given state-action pair
        3) Redefine Q as a mean of all past returns
        4) Update epsilon - with the help of epsilon_decay

        :param episode: the number of the episode
        :type episode: int
        :param argv: empty argument
        :type argv:
        """
        # Only condition is to take into account the first occurance of each SA pair -> keep a table

        have_seen_before = np.zeros((len(self.threshold_points[0]) + 1, len(self.threshold_points[1]) + 1,
                                     len(self.threshold_points[2]) + 1, len(self.threshold_points[3]) + 1, 2))

        timesteps = len(self.episode_database[episode])

        # Update return database
        for index, SAR in enumerate(self.episode_database[episode]):
            S_disc = self.__discretize_observation(SAR[0])
            index_tuple = tuple(S_disc + [SAR[1]])
            if have_seen_before[index_tuple] == 0:
                have_seen_before[index_tuple] = 1  # visited

                self.return_database[index_tuple].append(timesteps - index)
                # Update Q values right away
                self.Q[index_tuple] = np.mean(self.return_database[index_tuple])

        # Update epsilon
        self.epsilon *= self.epsilon_decay
        print('Epsilon is: ' + str(self.epsilon))


class Sarsa(Solver):
    def __init__(self, number_of_episodes, threshold_points, initial_q_value, discount_rate, epsilon, epsilon_decay,
                 alpha):
        """
        Constructor of the Q Learning solver

        :param number_of_episodes: number of episodes
        :type number_of_episodes: int
        :param threshold_points: points that divide each observation element into buckets/classes
        :type threshold_points: list of lists
        :param initial_q_value: entire q function is populated with this value at initialization
        :type initial_q_value: float
        :param discount_rate: in (0,1] - how we discount tomorrows value
        :type discount_rate: float
        :param epsilon: starting exploration parameter - on average epsilon * 100 % of time we explore,
        :type epsilon: float
        :param epsilon_decay: At the end of each each episode we put epsilon *= epsilon_decay -> we will end
                            up exploiting more in later periods
        :type epsilon_decay: float
        :param alpha: learning rate
        :type alpha: float
        """
        super(Sarsa, self).__init__(number_of_episodes)  # Initialize parent

        # Input attributes
        self.discount_rate = discount_rate
        self.threshold_points = threshold_points
        self.initial_q_value = initial_q_value
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha

        # Induced attributes
        self.state_space = self.__generate_state_space(self.threshold_points)  # list of 4-tuples - not really needed
        self.Q = self.__generate_Q(self.threshold_points, self.initial_q_value)  # 5 dimensional ndarray

    def __generate_state_space(self, threshold_points):
        """
        Generates all possible state spaces (4-tuples)

        :param threshold_points: points that devide each observation element into buckets/classes
        :type threshold_points: list of lists
        :return: list of all possible states <- state space
        :rtype: list of tuples
        """
        # Step 1 - For each state variable define possible outcomes - variable outcome array
        a_0 = np.arange(0, len(threshold_points[0]) + 1)
        a_1 = np.arange(0, len(threshold_points[1]) + 1)
        a_2 = np.arange(0, len(threshold_points[2]) + 1)
        a_3 = np.arange(0, len(threshold_points[3]) + 1)

        # Step 2 - Define a state space as a cartesian product of the 4 variable outcomes
        return list(itertools.product(a_0, a_1, a_2, a_3))

    def __generate_Q(self, threshold_points, initial_q_value):
        """
        Generates a Q functions as a ndarray

        :param threshold_points: point that partition state space for each observation element
        :type threshold_points: list of lists
        :param initial_q_value: the value we populate q function at initialization of the algorithm
        :type initial_q_value: float
        :return: Q function represented as a 5d matrix ... first 4 coordinates are for observations and 5th is the action
        :rtype: ndarray
        """
        return initial_q_value * np.ones((len(threshold_points[0]) + 1, len(threshold_points[1]) + 1,
                                          len(threshold_points[2]) + 1, len(threshold_points[3]) + 1, 2))

    def __discretize_observation(self, continuous_observation):
        '''
        Inputs a continuous observation and assigns to it the discretized version based on the threshold points

        :param continuous_observation: the 4 element observation generated by the environment
        :type continuous_observation: ndarray
        :return: dicretized 4-element observation
        :rtype: ndarray
        '''
        output = [0, 0, 0, 0]

        for element in range(4):
            order = 0
            last = True

            for points in self.threshold_points[element]:
                if continuous_observation[element] < points:
                    last = False
                    output[element] = order
                    break

                else:
                    order += 1

            if last:
                output[element] = len(self.threshold_points[element])

        return output  # it can be just list since we will convert it to tuple anyway

    def choose_action(self, observation):
        """
        Given an observation, we perform our action - POLICY. The exact version

        :param observation: continuous observation
        :type observation: list
        :return: 1 or 0
        :rtype: int
        """
        # map observation to a discrete version
        disc_obs = self.__discretize_observation(observation)

        # decide whether exploration or exploitation mode

        explore_mode = np.random.rand() < self.epsilon

        if self.Q[tuple(disc_obs + [0])] > self.Q[tuple(disc_obs + [1])]:
            if explore_mode:
                return 1
            else:
                return 0
        elif self.Q[tuple(disc_obs + [0])] < self.Q[tuple(disc_obs + [1])]:
            if explore_mode:
                return 0
            else:
                return 1
        else:
            return np.random.choice([0, 1], 1)[0]

    def end_of_timestep_update(self, i_episode, timestep, done):
        """
        Main SARSA update step

        :param i_episode: number of the episode
        :type i_episode: int
        :param timestep: number of the timestep
        :type timestep: int
        :param done: indicates whether game over
        :type done: bool
        """
        if timestep == 0:
            return
        S_old, A_old, R = self.episode_database[i_episode][timestep - 1]

        S_new, A_new, _ = self.episode_database[i_episode][timestep]

        S_old_disc = self.__discretize_observation(S_old)
        S_new_disc = self.__discretize_observation(S_new)

        self.Q[tuple(S_old_disc + [A_old])] += self.alpha * (
            R + self.discount_rate * self.Q[tuple(S_new_disc + [A_new])] - self.Q[tuple(S_old_disc + [A_old])])

        # EXTREMELY IMPORTANT
        if done and timestep < 199:
            # we have to set the reward for the last-terminal observation manually, since the above update can sometimes
            # adjusts it -> WHY????
            # The same continuous state sometimes leads to done and sometimes not -> Because we discretized our
            # sample space - so it is really important to just manually make sure that its 0

            # Maybe a more elegant solution is to select threshold points such that the bins correspond
            # To the actual values that lead to failure (x /notin (-2.4,2.4) and theta /notin ()

            # It works like an induction, these terminal states will backpropagate to the earlier states
            self.Q[tuple(S_new_disc + [A_new])] = 0

    def end_of_episode_update(self, *arg):
        """
        Only used to update epsilon


        :param arg: unused parameter
        :type arg:
        """
        # EPSILON UPDATE
        self.epsilon *= self.epsilon_decay
        print('Epsilon is: ' + str(self.epsilon))


class Results:
    def __init__(self):
        self.timesteps_per_episode_database = []

    def add_result(self, res):
        self.timesteps_per_episode_database.append(res)

    def plot_evolution(self):
        """
        Plots evolution of timesteps with increasing episodes - IMPORTANT: stops execution because of plt.show()

        """
        plt.figure()
        plt.title('Evolution')
        plt.plot(self.timesteps_per_episode_database)

        plt.show()

    def compute_empirical_quntiles(self, episode_database, quantiles):
        """
        Prints empirical quantiles of all observations

        :param episode_database: an attribute from solver - already fully populated when calling this function
        :type episode_database: list of lists
        :param quantiles: list of quantiles that we want to compute for each of the 4 observation elements
        :type quantiles: list

        """
        S = [episode_database[e][t][0] for e in range(len(episode_database)) for t in range(len(episode_database[e]))]
        s_0 = [i[0] for i in S]
        s_1 = [i[1] for i in S]
        s_2 = [i[2] for i in S]
        s_3 = [i[3] for i in S]

        for q in quantiles:
            print("x - percentile: " + str(q) + " is: " + str(np.percentile(s_0, q)))
            print("xdot - percentile: " + str(q) + " is: " + str(np.percentile(s_1, q)))
            print("theta - percentile: " + str(q) + " is: " + str(np.percentile(s_2, q)))
            print("thetadot - percentile: " + str(q) + " is: " + str(np.percentile(s_3, q)))
            print()


