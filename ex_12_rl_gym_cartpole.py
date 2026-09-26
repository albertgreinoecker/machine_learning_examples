import gymnasium as gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Modified hyperparameters
ENV_NAME = "CartPole-v1"
EPISODES = 500

# Increase max steps per episode so the environment won't truncate too early
# Gymnasium allows setting max_episode_steps when creating the env:
env = gym.make(ENV_NAME, max_episode_steps=1000)

STATE_SIZE = 4
ACTION_SIZE = 2
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 5000         # Larger replay buffer
EPSILON_DECAY = 0.99       # Slower decay to explore longer
EPSILON_MIN = 0.01

# Lower negative reward if done or truncated; from -10 to -1
FAILURE_PENALTY = -1

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0
        self.model = self._build_model()

    def _build_model(self):
        """
        Build a simple neural network with Keras to approximate Q-values.
        Increased layer size to 32 -> 32 for more capacity.
        """
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience in replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action based on an epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """
        Sample a batch from memory and train the neural network.
        """
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, next_state, done in minibatch:
            # Current Q-values
            target = self.model.predict(state, verbose=0)[0]

            if done:
                # If episode ended, the target is just the immediate reward
                target[action] = reward
            else:
                # Otherwise, add discounted future reward
                t = self.model.predict(next_state, verbose=0)[0]
                target[action] = reward + DISCOUNT_FACTOR * np.amax(t)

            # Train the network
            self.model.fit(state, np.array([target]), epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

def train_dqn():
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

    for episode in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, STATE_SIZE])

        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Penalize failure less severely (-1 instead of -10).
            if done or truncated:
                reward = FAILURE_PENALTY

            next_state = np.reshape(next_state, [1, STATE_SIZE])

            # Store experience
            agent.remember(state, action, reward, next_state, done or truncated)

            # Train from memory
            agent.replay()

            state = next_state
            total_reward += reward

        print(
            f"Episode: {episode+1}/{EPISODES}, "
            f"Score: {int(total_reward)}, "
            f"Epsilon: {agent.epsilon:.2f}"
        )

    env.close()

if __name__ == "__main__":
    train_dqn()
