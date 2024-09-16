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
        self.last_move_type = [None, None]  # Track the last move type for each player
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)  # initialize empty board
        self._setup_board()

    def _setup_board(self):
        # Set up the board according to the new initial placement rules
        values = [6, 5, 4, 3, 2, 1]
        
        # Player 0's positive pieces on rows 0 and 1
        self.board[0, :] = values
        self.board[1, :] = values[::-1]
        
        # Player 1's negative pieces on rows 4 and 5
        self.board[4, :] = [-v for v in values]
        self.board[5, :] = [-v for v in values[::-1]]

    def print_board(self):
        """Print the board with a frame and the number indicators for rows."""
        # Top numbering, now reversed to match the expected format
        print("    " + "  ".join(str(self.board_size - i) for i in range(self.board_size)))

        print("  ┌" + "───" * self.board_size + "─┐")  # Adjusted frame length

        for i, row in enumerate(self.board):
            row_str = " ".join(f"{x:+2d}" if x != 0 else "  " for x in row)  # Ensure empty fields are "  "
            print(f"{i + 1} │ {row_str} │")  # Adjusted row numbering from 1 to 6

        print("  └" + "───" * self.board_size + "─┘")  # Adjusted frame length

        # Bottom numbering (reversed as in the expected format)
        print("    " + "  ".join(str(i + 1) for i in range(self.board_size)))


    def valid_moves(self, player):
        # Determine valid moves for the current player
        moves = []
        direction = 1 if player == 0 else -1  # Player 0 moves down, Player 1 moves up
        for x in range(self.board_size):
            for y in range(self.board_size):
                piece = self.board[x, y]
                if player == 0 and piece > 0:  # Positive player pieces
                    moves.extend(self._get_moves(x, y, direction, player))
                elif player == 1 and piece < 0:  # Negative player pieces
                    moves.extend(self._get_moves(x, y, direction, player))
        return moves

    def _get_moves(self, x, y, direction, player):
        # Return possible moves for a piece located at (x, y)
        piece = self.board[x, y]
        moves = []

        # Forward move: no jumps allowed, no additions/subtractions
        if 0 <= x + direction < self.board_size and self.board[x + direction, y] == 0:
            moves.append((x, y, x + direction, y, 'forward'))

        # Sideways moves: no jumps, whole width allowed, check for consecutive lateral move restriction
        if self.last_move_type[player] != 'sideways':
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

        # Diagonal forward attack or addition (same sign): add based on opponent or same piece
        for dx, dy in [(direction, -1), (direction, 1)]:
            if 0 <= x + dx < self.board_size and 0 <= y + dy < self.board_size:
                target_piece = self.board[x + dx, y + dy]
                if target_piece * piece < 0:  # Opponent piece: addition attack
                    new_value = piece + target_piece
                    if abs(new_value) <= 6:
                        moves.append((x, y, x + dx, y + dy, 'attack'))
                elif target_piece * piece > 0:  # Same sign piece: addition
                    new_value = piece + target_piece
                    if abs(new_value) <= 6:  # Only valid if result stays within the range
                        moves.append((x, y, x + dx, y + dy, 'addition'))

        return moves

    def play_action(self, move):
        x, y, nx, ny, move_type = move
        piece = self.board[x, y]
        target = self.board[nx, ny]

        # Check for move restrictions: prevent consecutive sideways moves for the same player
        if move_type == 'sideways' and self.last_move_type[self.player_turn] == 'sideways':
            print("Cannot perform consecutive sideways moves.")
            return False

        if move_type == 'forward' or move_type == 'sideways':
            # Forward and sideways moves are simple: no interaction, just move the piece
            self.board[nx, ny] = piece
        elif move_type == 'jump':
            # Jump over a same-sign piece, move to an empty square
            self.board[nx, ny] = piece
        elif move_type == 'attack':
            # Opponent piece, perform addition (not subtraction)
            result = piece + target
            if abs(result) <= 6:
                self.board[nx, ny] = result
                # Scoring: attacker gets points
                self.score[self.player_turn] += min(abs(piece), abs(target))
            else:
                print(f"Move invalid: attack results in {result}, which is outside of the allowed range.")
                return False
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

        # Track the move type to enforce lateral move restriction
        self.last_move_type[self.player_turn] = move_type

        # Check if the piece reached the last row
        if (self.player_turn == 0 and nx == self.board_size - 1) or (self.player_turn == 1 and nx == 0):
            points = abs(piece)
            self.score[self.player_turn] += points
            self._return_to_first_row(nx, ny, piece)

        # Switch to the next player's turn
        self.player_turn = (self.player_turn + 1) % 2
        self.turns += 1

        return True

    def _return_to_first_row(self, nx, ny, piece):
        # Determine the corresponding column based on the value of the piece
        value = abs(piece)
        first_row = 0 if piece > 0 else self.board_size - 1

        if self.board[first_row, value - 1] == 0:
            self.board[first_row, value - 1] = piece
        else:
            # Perform an addition if the piece in the first row is of the same sign
            existing_piece = self.board[first_row, value - 1]
            if piece * existing_piece > 0 and abs(piece + existing_piece) <= 6:
                self.board[first_row, value - 1] = piece + existing_piece

    def is_game_over(self):
        # Game is over if no moves are left for either player, or max score is reached
        if any(score >= 24 for score in self.score):
            return True
        return self.turns >= self.max_turns or all(not self.valid_moves(p) for p in range(self.num_players))

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
        else:
            return -1  # Tie

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
