from __future__ import print_function

import signal
from functools import partial
import numpy as np
from flappy_bird_wrapper import GameState
from flappy_bird_wrapper import SCREENHEIGHT, SCREENWIDTH
from preprocessor import preprocess_screen
from sklearn.preprocessing import scale
from scipy.special import expit

from keras.models import Sequential
from keras.layers import Convolution2D, Activation, Flatten, Dense, Input, Dropout
from keras.initializations import normal, identity
from keras.regularizers import WeightRegularizer
from keras.optimizers import SGD , Adam


import matplotlib.pyplot as plt

from preprocessor import RESIZE

FRAMES = 4


gamma = 0.99  # discount factor for reward
batch_size = 10  # every how many episodes to do a param update?


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0.01:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def avg_eps(r, eps):
    return np.sum(r) / eps

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    sum_arr = np.exp(x[:, 0]) + np.exp(x[:, 1])
    x[:, 0] = np.exp(x[:, 0]) / sum_arr
    x[:, 1] = np.exp(x[:, 0]) / sum_arr
    return x


def conv_model():
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4),
                            input_shape=(FRAMES, RESIZE[1], RESIZE[0])))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # model.add(Dropout(.5))
    # model.add(Dense(100, activation='relu',  W_regularizer=WeightRegularizer(l2=.01)))
    # model.add(Dropout(.5))
    # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def simple_model():
    model = Sequential()
    # model.add(Input(shape=(FRAMES, RESIZE[1], RESIZE[0])))
    # model.add(Flatten())
    model.add(Dense(200, input_dim=FRAMES*RESIZE[0]*RESIZE[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model

def model2():
    print("Now we build the model")
    model = Sequential()
    model.add(
        Convolution2D(32, 8, 8, subsample=(4, 4), # init=lambda shape, name: normal(shape, scale=0.01, name=name),
                      border_mode='same', input_shape=(4, RESIZE[1], RESIZE[0])))
    model.add(Activation('relu'))
    model.add(
        Convolution2D(64, 4, 4, subsample=(2, 2), # init=lambda shape, name: normal(shape, scale=0.01, name=name),
                      border_mode='same'))
    model.add(Activation('relu'))
    model.add(
        Convolution2D(64, 3, 3, subsample=(1, 1), # init=lambda shape, name: normal(shape, scale=0.01, name=name),
                      border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(2, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('softmax'))

    adam = Adam(lr=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    print("We finish building the model")
    return model


def post_process(signum, frame, avg_rs, avg_scrs, num_eps):
    ep_arr = np.arange(num_eps, step=10)
    plt.plot(ep_arr, avg_rs, ep_arr, avg_scrs)
    plt.show()


def fit(model, ep_batch=1):
    game = GameState()
    ep_count = 0
    avg_scores, avg_rewards = [], []

    while True:
        inputs, actions, r, p = [], [], [], []
        # Run ep_batch number of episodes
        for i in xrange(ep_batch):
            # Get the first state
            image_data, terminal, reward = game.frame_step(0)
            screen = preprocess_screen(image_data, (SCREENHEIGHT, SCREENWIDTH))
            state_t = [screen for _ in xrange(FRAMES)]
            epinputs, epactions, epr, epp = [], [], [], []
            # Keep on moving until we lose
            while not terminal:
                curr_state = epinputs[-1] if len(epinputs) else np.array(state_t).reshape((1, FRAMES, RESIZE[1], RESIZE[0]))
                epp.append(model.predict(curr_state)[0, 1])  # Get the predictions
                epactions.append(1 if np.random.rand() < epp[-1] else 0)  # Sample from proba of jumping
                # epactions.append(int(np.around(epp[-1])))
                image_data, terminal, reward = game.frame_step(epactions[-1])  # Apply the action
                # Add the newest state and remove the earliest state
                state_t = state_t[1:] + [(preprocess_screen(image_data, (SCREENHEIGHT, SCREENWIDTH)))]
                epinputs.append(np.array(state_t).reshape((1, FRAMES, RESIZE[1], RESIZE[0])))
                epr.append(reward)

            inputs += epinputs
            actions += epactions
            r += epr
            p += epp

            # print(epp)
            # print(epactions)

        arr_r = np.array(r)
        disc_rs = scale(discount_rewards(arr_r))  # Calculate the discounted rewards  and scale them
        disc2_rs = np.zeros(disc_rs.shape + (2,))
        arr_actions = np.array(actions)
        disc2_rs[arr_actions == 0, 0] = disc_rs[arr_actions == 0]
        disc2_rs[arr_actions == 1, 1] = disc_rs[arr_actions == 1]
        # y = np.random.rand(*np.array(p).shape)
        # y[np.array(actions) == 1] = 1
        # y[np.array(actions) == 1] = (np.array(p) * disc_rs)[np.array(actions) == 1]
        # y = np.array(actions) * disc_rs - (disc_rs - 1) * np.array(p)  # Formula for labels so gradients work
        # disc_rs[np.array(actions) == 0] *= -1.
        y = np.zeros(disc2_rs.shape)
        y[:, 0] = (disc2_rs[:, 0] + (1 - disc2_rs[:, 0]) * (1 - np.array(p)))
        y[:, 1] = (disc2_rs[:, 1] + (1 - disc2_rs[:, 1]) * (np.array(p)))
        # y = softmax(y)

        # y[np.array(actions) == 0] *= -1

        # print("Probabilities", np.around(p, 4))
        # print("Labels: ", np.around(y, 2))

        model.fit(np.vstack(inputs), y, nb_epoch=1, shuffle=False, verbose=0)

        # Increment the number of games we've played
        ep_count += ep_batch

        # Calculate the avg reward and score per episode as a metric
        avg_reward = avg_eps(r, ep_batch)
        avg_score = avg_eps(r > 0, ep_batch)
        if ep_count % 100 == 0:
            print('EPISODES PLAYED: %s' % ep_count)
            print('AVERAGE REWARD: %s' % avg_reward)
            print('AVERAGE SCORE: %s' % avg_score)

        # Save the model every 100 episodes
        if ep_count % 100 == 0:
            model.save_weights('models/{0}eps_{1}r_{2}scr_weights.h5'.format(ep_count, avg_reward, avg_score))

        # Store them so we can graph them
        avg_rewards.append(avg_reward)
        avg_scores.append(avg_score)

    # Allow us to break out of training at any time and show our results
    signal.signal(signal.SIGINT, partial(post_process, avg_rs=avg_rewards, avg_scrs=avg_scores, num_eps=ep_count))


def main():
    model = model2()
    model.load_weights('models/15500eps_0.7r_1scr_weights.h5')
    fit(model)

if __name__ == '__main__':
    main()
