import socket
import os
from typing import Callable, Optional

from alphazero.GenericPlayers import BasePlayer
from alphazero.Game import GameState
from alphazero.utils import dotdict
import traceback
import io
import sys


class HumanGobangPlayer(BasePlayer):
    def play(self, state: GameState) -> int:
        valid = state.valid_moves()
        """
        for i in range(len(valid)):
            if valid[i]:
                print(int(i / state._board.n), int(i % state._board.n))
        """

        while True:
            a = input('Enter a move: ')
            a_split = [int(x) for x in a.split(' ')]
            if len(a_split) == 2:
                x, y = a_split
                a = state._board.n * x + y if x != -1 else state._board.n ** 2
                if valid[a]:
                    break
                else:
                    print('Invalid move entered.')
            else:
                print('Unexpected move format. Expected: x y')

        return a


class GreedyGobangPlayer(BasePlayer):
    def play(self, state: GameState) -> int:
        valids = state.valid_moves()
        candidates = []

        for a in range(state.action_size()):
            if not valids[a]: continue

            next_state = state.clone()
            next_state.play_action(a)
            candidates += [(int(next_state.win_state()[next_state.player]), a)]

        candidates.sort()
        return candidates[0][1]

class UnixSocketGobangPlayer(BasePlayer):

    def __init__(self, game_cls: GameState = None, args: dotdict = None, verbose: bool = False, display: Callable[[GameState, Optional[int]], None] = None):
        """
        Initializes the player class.
        :param game: An instance of the GobanGame class.
        """
        self.socket_path = "/tmp/gobang.sock"  # Unix socket path
        self.display = display
        self.setup_socket()
        super().__init__(game_cls, args, verbose)

    def setup_socket(self):
        """
        Sets up the Unix socket for communication.
        """
        # Remove old socket if exists
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        # Create the Unix socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(1)  # Allow only 1 connection at a time

        print(f"Socket setup complete. Listening at {self.socket_path}")

    def receive_move(self, conn):
        """
        Receives a move from the client.
        :param conn: The connection object from the client.
        """
        data = conn.recv(1024).decode('utf-8')  # Receive the move as a string
        return data

    def play(self, state: GameState) -> int:
        """
        Main game loop to handle receiving moves and sending board states.
        """
        while True:
            print("Waiting for a connection...")
            conn, _ = self.server_socket.accept()  # Accept the client connection
            print("Connection accepted.")
            
            try:
                while True:
                    valid = state.valid_moves()

                    # Capture the output of the display function
                    old_stdout = sys.stdout
                    new_stdout = io.StringIO()
                    sys.stdout = new_stdout
                    self.display(state)
                    # Reset stdout
                    sys.stdout = old_stdout
                    board_string = new_stdout.getvalue()
                    conn.sendall(board_string.encode('utf-8'))

                    print("Waiting for a connection to receive the move...")
                    conn, _ = self.server_socket.accept()  # Accept the client connection
                    print("Connection accepted.")

                    # Receive move from the client
                    valid_action = None
                    move = self.receive_move(conn)
                    print(f"Received move: {move}")
                    move_split = [int(x) for x in move.split(',')]
                    if len(move_split) == 2:
                        x, y = move_split
                        action = state._board.n * x + y if x != -1 else state._board.n ** 2
                        if valid[action]:
                            valid_action = action
                            break
                        else:
                            # TODO: how to send message back to client?
                            print('Invalid move entered.')
                            break
                    else:
                        print('Unexpected move format. Expected: x,y')
                        break
                    # board_string = self.display(state)
                    # conn.sendall(board_string.encode('utf-8'))

                conn.close()
                return valid_action
            
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
            finally:
                conn.close()
