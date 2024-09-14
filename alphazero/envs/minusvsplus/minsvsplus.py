import numpy as np
import random

class MinusPlusGame:
    def __init__(self):
        self.board_size = 6  # 6x6 grid
        self.num_players = 2
        self.max_turns = 80
        self.score = [0, 0]  # Player 0's score, Player 1's score
        self.turns = 0
        self.player_turn = 0  # 0 for positive player, 1 for negative player
        self.last_move_type = None  # Track the last move type to prevent consecutive lateral moves
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)  # initialize empty board
        self._setup_board()

    def _setup_board(self):
        # Set up the board according to the initial placement rules
        values = [1, 2, 3, 4, 5, 6]
        
        # Player 0's positive pieces on rows 0 and 1
        self.board[0, :] = values
        self.board[1, :] = values[::-1]
        
        # Player 1's negative pieces on rows 4 and 5
        self.board[4, :] = [-v for v in values[::-1]]
        self.board[5, :] = [-v for v in values]

    def print_board(self):
        # Display the current state of the board
        print("Current board:")
        for row in self.board:
            print(' '.join(f'{cell:+2d}' if cell != 0 else ' 0 ' if cell == -9 else '  ' for cell in row))  # 0 represents zero-value piece, empty fields as "  "
        print()

    def valid_moves(self, player):
        # Determine valid moves for the current player
        moves = []
        direction = 1 if player == 0 else -1  # Player 0 moves down, Player 1 moves up
        for x in range(self.board_size):
            for y in range(self.board_size):
                piece = self.board[x, y]
                if player == 0 and piece > 0:  # Positive player pieces
                    moves.extend(self._get_moves(x, y, direction))
                elif player == 1 and piece < 0:  # Negative player pieces
                    moves.extend(self._get_moves(x, y, direction))
        return moves

    def _get_moves(self, x, y, direction):
        # Return possible moves for a piece located at (x, y)
        piece = self.board[x, y]
        moves = []

        # Forward move: no jumps allowed, no additions/subtractions
        if 0 <= x + direction < self.board_size and self.board[x + direction, y] == 0:
            moves.append((x, y, x + direction, y, 'forward'))

        # Sideways moves: no jumps, whole width allowed
        for dy in [-1, 1]:
            new_y = y
            while 0 <= new_y + dy < self.board_size and self.board[x, new_y + dy] == 0:
                new_y += dy
                moves.append((x, y, x, new_y, 'sideways'))

        # Diagonal forward moves with same sign (jump over one piece of same sign)
        for dx, dy in [(direction, -1), (direction, 1)]:
            if 0 <= x + 2*dx < self.board_size and 0 <= y + 2*dy < self.board_size:
                mid_piece = self.board[x + dx, y + dy]
                target_piece = self.board[x + 2*dx, y + 2*dy]
                if mid_piece * piece > 0 and target_piece == 0:  # Jump over same sign piece
                    moves.append((x, y, x + 2*dx, y + 2*dy, 'jump'))

        # Diagonal forward attack or addition (same sign): add/subtract based on opponent
        for dx, dy in [(direction, -1), (direction, 1)]:
            if 0 <= x + dx < self.board_size and 0 <= y + dy < self.board_size:
                target_piece = self.board[x + dx, y + dy]
                if piece * target_piece < 0:  # Opponent piece: subtraction
                    moves.append((x, y, x + dx, y + dy, 'attack'))
                elif piece * target_piece > 0:  # Same sign piece: addition
                    new_value = piece + target_piece
                    if abs(new_value) <= 6:
                        moves.append((x, y, x + dx, y + dy, 'addition'))

        return moves

    def play_action(self, move):
        x, y, nx, ny, move_type = move
        piece = self.board[x, y]
        target = self.board[nx, ny]

        # Check for move restrictions: prevent consecutive sideways moves
        if move_type == 'sideways' and self.last_move_type == 'sideways':
            print("Cannot perform consecutive sideways moves.")
            return False

        if move_type == 'forward' or move_type == 'sideways':
            # Forward and sideways moves are simple: no interaction, just move the piece
            self.board[nx, ny] = piece
        elif move_type == 'jump':
            # Jump over a same-sign piece, move to an empty square
            self.board[nx, ny] = piece
        elif move_type == 'attack':
            # Opponent piece, perform subtraction
            result = piece - target
            if result == 0:
                # Replace both pieces with a zero-value piece
                self.board[nx, ny] = -9  # Represent zero-value pieces with -9
            else:
                self.board[nx, ny] = result
                # Scoring: attacker gets points
                self.score[self.player_turn] += min(abs(piece), abs(target))
        elif move_type == 'addition':
            # Same-sign piece, perform addition
            result = piece + target
            if abs(result) <= 6:
                self.board[nx, ny] = result
            else:
                print(f"Move invalid: addition results in {result}, which is outside of the allowed range.")
                return False

        # Empty the original cell
        self.board[x, y] = 0

        # Handle zero-value pieces (cannot be moved or jumped over)
        if target == -9:
            print("Zero-value pieces cannot be moved or jumped.")

        # Track the move type to enforce lateral move restriction
        self.last_move_type = move_type

        # Check if the piece reached the last row
        if (self.player_turn == 0 and nx == self.board_size - 1) or (self.player_turn == 1 and nx == 0):
            # Place the piece back in the first row
            self._return_to_first_row(nx, ny, piece)

        # Switch to the next player's turn
        self.player_turn = (self.player_turn + 1) % self.num_players
        self.turns += 1
        return True

    def _return_to_first_row(self, nx, ny, piece):
        # Handle returning a piece to the first row after reaching the last row
        mod_value = abs(piece)
        if self.player_turn == 0:
            row = 0
        else:
            row = self.board_size - 1

        # Check if the first row's square is occupied
        if self.board[row, mod_value - 1] == 0:
            self.board[row, mod_value - 1] = piece
        else:
            # Add values if possible
            result = self.board[row, mod_value - 1] + piece
            if abs(result) <= 6:
                self.board[row, mod_value - 1] = result
            else:
                # Otherwise, remove the piece from the board
                self.board[row, mod_value - 1] = 0

    def is_game_over(self):
        # The game ends when a player reaches a score of 24 or the max turns are reached
        return self.turns >= self.max_turns or any(score >= 24 for score in self.score)

    def winner(self):
        if self.score[0] >= 24:
            return 0  # Positive player wins
        elif self.score[1] >= 24:
            return 1  # Negative player wins
        elif self.turns >= self.max_turns:
            return 0 if self.score[0] > self.score[1] else 1 if self.score[1] > self.score[0] else -1  # -1 for tie
        return None

# Simulate a full game by selecting random valid moves
if __name__ == "__main__":
    game = MinusPlusGame()
    game.print_board()

    while not game.is_game_over():
        # Get the current player's valid moves
        player = game.player_turn
        moves = game.valid_moves(player)

        # Print the valid moves before making a move
        print(f"Valid moves for Player {player}: {moves}")

        if not moves:
            print(f"Player {player} has no valid moves. The game ends in a draw.")
            break

        # Randomly select a move from available moves
        move = random.choice(moves)
        print(f"Selected move: {move}")
        game.play_action(move)
        game.print_board()

        print(f"Turn {game.turns}, Player {player} moved. Current Scores: {game.score}")

    # Print the result of the game
    if game.is_game_over():
        winner = game.winner()
        if winner == -1:
            print("The game ends in a tie.")
        else:
            print(f"Player {winner} wins!")
