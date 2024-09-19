
from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.minusvsplus.minsvsplus import MinusPlusGame

from alphazero.GenericPlayers import RawMCTSPlayer

args = get_args(run_name='minusvsplus', workers=11)

if __name__ == "__main__":
    nnet = nn(MinusPlusGame, args)
    c = Coach(MinusPlusGame, nnet, args)
    c.learn()
