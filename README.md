An implementation of DeepMind's AlphaGo. Original paper from Google DeepMind: https://www.nature.com/articles/nature16961


1. Start by cloning the git repository.

2. Start training. Run train.py like so:
**python3 train.py --policy --value**
Specify both "--policy" and "--value" to train both networks since we need both to be trained for our Monte Carlo tree search algorithm to work. After training (which might take a while), the parameters will be saved to 'policy_network_model.pth' and 'value_network_model.pth'). Training has to reach the complete end for the parameters to be saved (do not stop while it's in the middle of training).

5. Play the game by running:
**python3 mcts.py**

