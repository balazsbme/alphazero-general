import numpy as np
import random

from alphazero.Game import GameState

BOARD_SIZE = 6 # 6x6 grid
NUMBER_OF_INITIAL_PIECES_PER_PLAYER = 12
MAX_MOVES_PER_PIECE = 10

class MinusPlusGame(GameState):
    def __init__(self):
        self.board_size = BOARD_SIZE
        initial_board = np.zeros((self.board_size, self.board_size), dtype=int)  # initialize empty board
        super().__init__(initial_board)  # Call the parent constructor with the initial board

        self.max_turns = 80
        self.score = [0, 0]  # Player 0's score, Player 1's score
        self.last_move_type = [None, None]  # Track the last move type for each player
        self._setup_board()
        self._valid_moves_cache = None  # Initialize the cache

    @staticmethod
    def observation_size():
        """Return the size of the observation space."""
        return (1, BOARD_SIZE, BOARD_SIZE)

    @staticmethod
    def num_players():
        """Return the number of players."""
        return 2
    
    @staticmethod
    def action_size():
        """Return the maximum possible number of valid moves."""
        # Each piece can move in multiple directions:
        # - 1 forward move
        # - 5 sideways moves (whole width of the board)
        # - 2 diagonal forward moves (jump)
        # - 2 diagonal forward moves (attack/addition)
        return NUMBER_OF_INITIAL_PIECES_PER_PLAYER * MAX_MOVES_PER_PIECE
    
    def __eq__(self, other):
        """Check if two game states are equal."""
        return np.array_equal(self._board, other.board) and self.score == other.score and self.turns == other.turns and self.player == other.player

    def observation(self) -> np.ndarray:
        """Return the current observation of the game."""
        # TODO: add obsrevation also the current points, maybe the turn count (Should it return symmetries??!!)
        return self._board

    def win_state(self):
        """Determine the win state of the game."""
        result = [False] * (self.num_players() + 1)  # Include an extra element for the tie
        if self.is_game_over():
            winner = self.winner()
            if winner == 0:
                result[0] = True  # Positive player wins
            elif winner == 1:
                result[1] = True  # Negative player wins
            else:
                result[2] = True  # Tie
        return np.array(result, dtype=np.uint8)
    
    def clone(self):
        """Create a deep copy of the game state."""
        cloned_game = MinusPlusGame()
        cloned_game._board = np.copy(self._board)
        cloned_game.score = self.score[:]
        cloned_game._turns = self.turns
        cloned_game._player = self.player
        cloned_game.last_move_type = self.last_move_type[:]
        return cloned_game

    def _setup_board(self):
        # Set up the board according to the new initial placement rules
        values = [6, 5, 4, 3, 2, 1]
        
        # Player 0's positive pieces on rows 0 and 1
        self._board[0, :] = values
        self._board[1, :] = values[::-1]
        
        # Player 1's negative pieces on rows 4 and 5
        self._board[4, :] = [-v for v in values]
        self._board[5, :] = [-v for v in values[::-1]]

    def get_board(self):
        """Return the current state of the board."""
        return self._board

    def get_current_player(self):
        """Return the current player."""
        return self.player

    # TODO: add symmetries implementation for 180 degree rotation + swap colors (if needed...)

    def print_board(self):
        """Print the board with a frame and the number indicators for rows."""
        # Top numbering, now reversed to match the expected format
        print("    " + "  ".join(str(self.board_size - i) for i in range(self.board_size)))

        print("  ┌" + "───" * self.board_size + "─┐")  # Adjusted frame length

        for i, row in enumerate(self._board):
            row_str = " ".join(f"{x:+2d}" if x != 0 else "  " for x in row)  # Ensure empty fields are "  "
            print(f"{i + 1} │ {row_str} │")  # Adjusted row numbering from 1 to 6

        print("  └" + "───" * self.board_size + "─┘")  # Adjusted frame length

        # Bottom numbering (reversed as in the expected format)
        print("    " + "  ".join(str(i + 1) for i in range(self.board_size)))
    
    def get_valid_moves(self):
        if self._valid_moves_cache is not None:
            return self._valid_moves_cache
        # Determine valid moves for the current player
        moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self._board[x, y] < 0 and self.player == 1 or self._board[x, y] > 0 and self.player == 0:
                    piece_moves = self._get_moves(x, y, self.player)
                    assert len(piece_moves) == MAX_MOVES_PER_PIECE, f"Expected {MAX_MOVES_PER_PIECE} moves, but got {len(piece_moves)}"
                    moves.extend(piece_moves)
        # Count the pieces, if it is less than the initial number of pieces, extend with a full None list to the moves list
        current_piece_count = np.count_nonzero(self._board < 0) if self.player == 1 else np.count_nonzero(self._board > 0)
        if current_piece_count < NUMBER_OF_INITIAL_PIECES_PER_PLAYER:
            additional_moves_needed = (NUMBER_OF_INITIAL_PIECES_PER_PLAYER - current_piece_count) * MAX_MOVES_PER_PIECE
            moves.extend([None] * additional_moves_needed)
        assert len(moves) == self.action_size(), f"Expected {self.action_size()} moves, but got {len(moves)}"
        self._valid_moves_cache = moves
        return moves

    def valid_moves(self):
        moves = self.get_valid_moves()
        # Convert the moves to a numpy array with always fixed length to action size.
        return np.array([0 if move is None else 1 for move in moves], dtype=np.uint8)

    def _get_moves(self, x, y, player):
        # Return possible moves for a piece located at (x, y)
        piece = self._board[x, y]
        moves = [None] * MAX_MOVES_PER_PIECE
        index = 0
        last_row = self.board_size - 1 if player == 0 else 0
        direction = 1 if self.player == 0 else -1  # Player 0 moves down, Player 1 moves up

        # Forward move: no jumps allowed, no additions/subtractions
        if 0 <= x + direction < self.board_size and self._board[x + direction, y] == 0:
            moves[index] = (x, y, x + direction, y, 'forward')
        index += 1

        # Sideways moves: no jumps, whole width allowed, check for consecutive lateral move restriction
        if self.last_move_type[player] != 'sideways':
            for dy in [-1, 1]:
                first_block = False
                new_y = y + dy
                while 0 <= new_y < self.board_size:
                    if self._board[x, new_y] == 0 and not first_block:
                        moves[index] = (x, y, x, new_y, 'sideways')
                    else:
                        first_block = True
                    new_y += dy
                    index += 1
        else:
            index += self.board_size - 1

        # Diagonal forward moves with same sign (jump over one piece of same sign)
        for dx, dy in [(direction, -1), (direction, 1)]:
            if 0 <= x + 2*dx < self.board_size and 0 <= y + 2*dy < self.board_size:
                mid_piece = self._board[x + dx, y + dy]
                target_piece = self._board[x + 2*dx, y + 2*dy]
                if mid_piece * piece > 0 and target_piece == 0:  # Jump over same sign piece
                    moves[index] = (x, y, x + 2*dx, y + 2*dy, 'jump')
                index += 1
            else:
                index += 1

        # Diagonal forward attack or addition (same sign), but NOT allowed if moving into the last row
        for dx, dy in [(direction, -1), (direction, 1)]:
            if 0 <= x + dx < self.board_size and 0 <= y + dy < self.board_size:
                target_piece = self._board[x + dx, y + dy]
                if target_piece * piece < 0 and x + dx != last_row:  # Opponent piece: addition attack, not in last row
                    new_value = piece + target_piece
                    if abs(new_value) <= 6:
                        moves[index] = (x, y, x + dx, y + dy, 'attack')
                    index += 1
                elif target_piece * piece > 0:  # Same sign piece: addition
                    new_value = piece + target_piece
                    if abs(new_value) <= 6:  # Only valid if result stays within the range
                        moves[index] = (x, y, x + dx, y + dy, 'addition')
                    index += 1
                else:
                    index += 1
            else:
                index += 1

        return moves

    def play_action(self, action):
        self.play_move(self.get_valid_moves()[action])

    def play_move(self, move):
        x, y, nx, ny, move_type = move
        piece = self._board[x, y]
        target = self._board[nx, ny]

        # Check for move restrictions: prevent consecutive sideways moves for the same player
        if move_type == 'sideways' and self.last_move_type[self.player] == 'sideways':
            print("Cannot perform consecutive sideways moves.")
            return

        if move_type == 'forward' or move_type == 'sideways':
            # Forward and sideways moves are simple: no interaction, just move the piece
            self._board[nx, ny] = piece
        elif move_type == 'jump':
            # Jump over a same-sign piece, move to an empty square
            self._board[nx, ny] = piece
        elif move_type == 'attack':
            # Opponent piece, perform addition
            result = piece + target
            # if result == 0:
            #     print(f"Produced a 0 as a result of attack, TO BE HANDLED!")
            if abs(result) <= 6:
                self._board[nx, ny] = result
                # Scoring: attacker gets points
                self.score[self.player] += min(abs(piece), abs(target))
            else:
                print(f"Move invalid: attack results in {result}, which is outside of the allowed range.")
                return 
        elif move_type == 'addition':
            # Same-sign piece, perform addition
            result = piece + target
            if abs(result) <= 6:
                self._board[nx, ny] = result
            else:
                print(f"Move invalid: addition results in {result}, which is outside of the allowed range.")
                return 

        # Empty the original cell
        self._board[x, y] = 0
        self._valid_moves_cache = None  # Reset the cache

        # Track the move type to enforce lateral move restriction
        self.last_move_type[self.player] = move_type

        # Check if the piece reached the last row
        if (self.player == 0 and nx == self.board_size - 1) or (self.player == 1 and nx == 0):
            points = abs(piece)
            self.score[self.player] += points
            self._return_to_first_row(nx, ny, piece)

        self._update_turn()

    def _return_to_first_row(self, nx, ny, piece):
        # Determine the corresponding column based on the value of the piece
        value = abs(piece)
        first_row = 0 if piece > 0 else self.board_size - 1

        if self._board[first_row, value - 1] == 0:
            self._board[first_row, value - 1] = piece
        else:
            # Perform an addition if the piece in the first row is of the same sign
            existing_piece = self._board[first_row, value - 1]
            if piece * existing_piece > 0 and abs(piece + existing_piece) <= 6:
                self._board[first_row, value - 1] = piece + existing_piece
        self._board[nx, ny] = 0

    def is_game_over(self):
        # Game is over if no moves are left for either player, or max score is reached
        if any(score >= 24 for score in self.score):
            return True
        valid_moves = self.get_valid_moves()
        return self.turns >= self.max_turns or all(move is None for move in valid_moves)

    def winner(self):
        # Determine the winner based on the score
        if self.score[0] >= 24:
            return 0  # Positive player wins by score
        elif self.score[1] >= 24:
            return 1  # Negative player wins by score
        elif self.score[0] > self.score[1]:
            return 0
        elif self.score[1] > self.score[0]:
            return 1
        valid_moves = self.get_valid_moves()
        if all(move is None for move in valid_moves):
            # Handle tie cases based on player moves
            other_player = 1 if self.player == 0 else 0
            return -1 if self.score[self.player] >= self.score[other_player] else other_player
        return -1  # Tie

if __name__ == "__main__":
    game = MinusPlusGame()
    game.print_board()

    while not game.is_game_over():
        # Get the current player's valid moves
        player = game.get_current_player()
        moves = game.get_valid_moves()

        # Print the valid moves before making a move
        print(f"Valid moves for Player {player}: {[move for move in moves if move is not None]}")

        if not moves:
            print(f"Player {player} has no valid moves. The game ends in a draw.")
            break

        # Randomly select a move from available moves that is not None
        move = random.choice([m for m in moves if m is not None])
        print(f"Selected move: {move}")
        game.play_move(move)
        game.print_board()

        print(f"Turn {game.turns}, Player {player} moved. Current Scores: {game.score}")

    # Print the result of the game
    if game.is_game_over():
        winner = game.winner()
        if winner == -1:
            print("The game ends in a tie.")
        else:
            print(f"Player {winner} wins!")