from __future__ import print_function

import signal
from functools import partial
import numpy as np
from flappy_bird_wrapper import GameState
from flappy_bird_wrapper import SCREENHEIGHT, SCREENWIDTH
from preprocessor import preprocess_screen
from sklearn.preprocessing import scale

from keras.models import Sequential
from keras.layers import Convolution2D, Activation, Flatten, Dense, Dropout, MaxPooling2D

import matplotlib.pyplot as plt


gamma = 0.99  # discount factor for reward
batch_size = 10  # every how many episodes to do a param update?


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def avg_eps(r, eps):
    return np.sum(r) / eps

def conv_model():
    model = Sequential()
    model.add(Convolution2D(10, 8, 8, init='glorot_normal', subsample=(4, 4),
                            input_shape=(4, 94, 94)))
    model.add(Activation('relu'))
    model.add(Convolution2D(20, 4, 4, init='glorot_normal', subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model


def post_process(signum, frame, avg_rs, avg_scrs, num_eps):
    ep_arr = np.arange(num_eps, step=10)
    plt.plot(ep_arr, avg_rs, ep_arr, avg_scrs)
    plt.show()



def fit(model, ep_batch=10):
    game = GameState()
    ep_count = 0
    avg_scores, avg_rewards = [], []
    while True:

        # Get the first state
        image_data, reward, terminal = game.frame_step(1)
        screen = preprocess_screen(image_data, (SCREENWIDTH, SCREENHEIGHT))
        state_t = [screen for _ in xrange(4)]
        inputs, actions, r, p = [np.array(state_t)], [], [], []

        # Run ep_batch number of episodes
        for i in xrange(ep_batch):
            # Keep on moving until we lose
            while not terminal:
                p.append(model.predict(inputs[-1]))  # Get the predictions
                actions.append(1 if np.random.rand() < p else 0)  # Sample from proba of jumping
                image_data, terminal, reward = game.frame_step(actions[-1])  # Apply the action
                # Add the newest state and remove the earliest state
                state_t = state_t[1:] + [(preprocess_screen(image_data, (SCREENWIDTH, SCREENHEIGHT)))]
                inputs.append(np.array(state_t))
                r.append(reward)
        r = np.array(r)
        disc_rs = scale(discount_rewards(r))  # Calculate the discounted rewards  and scale them
        y = np.array(actions) * disc_rs - (disc_rs - 1) * np.array(p)  # Formula for labels so gradients work
        model.fit(np.array(inputs), y, nb_epoch=1)

        # Increment the number of games we've played
        ep_count += ep_batch

        # Calculate the avg reward and score per episode as a metric
        avg_reward = avg_eps(r, ep_batch)
        avg_score = avg_eps(r > 0, ep_batch)

        print('EPISODES PLAYED: %s' % ep_count)
        print('AVERAGE REWARD: %s' % avg_reward)
        print('AVERAGE SCORE: %s' % avg_score)

        # Store them so we can graph them
        avg_rewards.append(avg_reward)
        avg_scores.append(avg_score)

        # Allow us to break out of training at any time and show our results
        signal.signal(signal.SIGINT, partial(post_process, avg_rs=avg_rewards, avg_scrs=avg_scores, num_eps=ep_count))


def main():
    model = conv_model()
    fit(model)

if __name__ == '__main__':
    main()
