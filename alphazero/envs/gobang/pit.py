import argparse
import json
from pyximport import install
from numpy import get_include
install(setup_args={'include_dirs': get_include()})

from alphazero.Arena import Arena
from alphazero.GenericPlayers import *
from alphazero.NNetWrapper import NNetWrapper as NNet
from alphazero.Coach import get_args


"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
if __name__ == '__main__':
    from alphazero.envs.gobang.gobang import Game, display
    from alphazero.envs.gobang.train import args
    from alphazero.envs.gobang.GobangPlayers import HumanGobangPlayer
    import random

    parser = argparse.ArgumentParser(description='Run the AlphaZero Gobang pit.')
    parser.add_argument('--opponent_type', type=str, choices=['self', 'human'], default='self',
                        help='Type of player: "self" for neural network, "human" for human player')
    parser.add_argument('--checkpoint_folder', type=str, required=False,
                        help='Full path to the first checkpoint folder')
    parser.add_argument('--checkpoint', type=str, required=False,
                        help='Filename of checkpoint')
    parser.add_argument('--args_path', type=str, required=False,
                    help='Path to the args.json file')
    script_args = parser.parse_args()

    if script_args.args_path is not None:
        with open(script_args.args_path, "r") as f:
            args = json.load(f)
            args = get_args(args)
    batched_arena = False
    args.numMCTSSims = 400
    args._num_players = 2
    #args.arena_batch_size = 64
    args.temp_scaling_fn = lambda x,y,z:0.2
    #args2.temp_scaling_fn = args.temp_scaling_fn
    #args.cuda = False
    args.add_root_noise = args.add_root_temp = False

    # nnet players
    nn1 = NNet(Game, args)
    if script_args.checkpoint_folder:
        nn1.load_checkpoint(script_args.checkpoint_folder, script_args.checkpoint)
    else:
        nn1.load_checkpoint('./checkpoint/' + args.run_name, 'mcts-100-iteration-0145.pkl')
    player1 = NNPlayer(nn1, args=args, verbose=True)

    if script_args.opponent_type == 'self':
        nn2 = NNet(Game, args)
        if script_args.checkpoint_folder:
            nn2.load_checkpoint(script_args.checkpoint_folder, script_args.checkpoint)
        else:
            nn2.load_checkpoint('./checkpoint/' + args.run_name, 'iteration-0118.pkl')
        player2 = NNPlayer(nn2, args=args, verbose=True)
    elif script_args.opponent_type == 'human':
        player2 = HumanGobangPlayer(Game, args=args, verbose=True)

    #player1 = nn1.process
    #player2 = nn2.process

    #player2 = RandomPlayer()
    #player2 = GreedyTaflPlayer()
    #player2 = RandomPlayer()
    #player2 = OneStepLookaheadConnect4Player()
    #player2 = RawMCTSPlayer(Game, args)
    #player2 = HumanFastaflPlayer()

    players = [player1, player2]
    #random.shuffle(players)

    arena = Arena(players, Game, use_batched_mcts=batched_arena, args=args, display=display)
    if batched_arena:
        wins, draws, winrates = arena.play_games(args.arenaCompare)
        for i in range(len(wins)):
            print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
        print('draws: ', draws)
    else:
        arena.play_game(verbose=True)

