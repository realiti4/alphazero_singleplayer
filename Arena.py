import logging

from tqdm import tqdm

log = logging.getLogger(__name__)



# Single-player MCTS
import numpy as np
import matplotlib
# matplotlib.use('TKAgg')     # Agg not displaying plot
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Plotting
def plot_end_result(plot_price, plot_action):
    time = np.arange(len(plot_price))
    # colors {0: 'b', 1: 'r'}
    print(plot_action)

    # fig = plt.figure()

    plt.plot(plot_price, zorder=0)
    plt.scatter(x=time,y=plot_price,c=['b' if x == 1 else 'r' for x in plot_action])
    
    # plt.show()  # TODO save plots instead of showing since it is not working in mp
    plt.savefig('recent.png')

    print('debug')

# plot_end_result([4, 10, 40, 20, 18], [1, 1, 0, 0, 1])

def simple_evaluation(game, mcts):
    trainExamples = []
    # board = game.getInitBoard()
    canonicalBoard = game.reset(key=0)
    episodeStep = 0

    plot_price = []
    plot_action = []

    while True:
        # canonicalBoard = game.getCanonicalForm(board, 1)

        pi = mcts.getActionProb(canonicalBoard, temp=0)
        trainExamples.append([canonicalBoard, pi])

        action = np.argmax(pi)

        plot_price.append(game.current_price)
        plot_action.append(action)

        canonicalBoard, _, _, info = game.step(action)

        r = game.getGameEnded()

        if r != 0:
            print(r)
            print(info)
            plot_end_result(plot_price, plot_action)
            return info['achievement']


    for i in range(10000):
        with torch.no_grad():
            _, action, _ = model(state.unsqueeze(1))

        state, reward, done, info = env_evaluate.step(action.item())
        state = torch.from_numpy(state).float().to(device)

        if render:
            env_evaluate.render()

        if done:
            rewards.append(info['reward'])
            achievements.append(info['achievement'])
            break

