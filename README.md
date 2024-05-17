An implementation of DeepMind's AlphaGo. Original paper from Google DeepMind: https://www.nature.com/articles/nature16961


1. Start by cloning the git repo and creating a folder named "sgffiles". Download some past professional Go games from https://u-go.net/gamerecords/ and copy those .sgf files in this folder. You can add as many as you like but it might take a long time to train without a GPU. I used 100 SGF files which took about 2-3 hours of training time in total. I would recommend sticking to around 100 or less.

2. Start training. Run train.py like so: python3 train.py --policy --value
Specify both "--policy" and "--value" to train both networks since we need both to be trained for our Monte Carlo tree search algorithm to work. After training (which might take a while), the parameters will be saved to 'policy_network_model.pth' and 'value_network_model.pth')

3. Play the game by running: python3 mcts.py

