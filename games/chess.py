import chess
import numpy as np


class Chess:
    """Класс игры Шахматы с python-chess"""

    EMPTY = 0
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6

    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.shape_obs = 19
        self.action_size = 64 * 64

        self._piece_to_chess = {
            self.PAWN: chess.PAWN, self.KNIGHT: chess.KNIGHT,
            self.BISHOP: chess.BISHOP, self.ROOK: chess.ROOK,
            self.QUEEN: chess.QUEEN, self.KING: chess.KING,
        }
        self._chess_to_piece = {v: k for k, v in self._piece_to_chess.items()}

    def __repr__(self):
        return "Chess"

    def get_initial_state(self):
        pieces = np.zeros((8, 8), dtype=np.int8)
        pieces[0] = [-self.ROOK, -self.KNIGHT, -self.BISHOP, -self.QUEEN,
                     -self.KING, -self.BISHOP, -self.KNIGHT, -self.ROOK]
        pieces[1] = [-self.PAWN] * 8
        pieces[6] = [self.PAWN] * 8
        pieces[7] = [self.ROOK, self.KNIGHT, self.BISHOP, self.QUEEN,
                     self.KING, self.BISHOP, self.KNIGHT, self.ROOK]
        return {
            'pieces': pieces,
            'ep_square': None,
            'castling_rights': chess.BB_A1 | chess.BB_H1 | chess.BB_A8 | chess.BB_H8,
            'halfmove_clock': 0
        }

    def _state_to_board(self, state, player=1):
        pieces = state['pieces'] if isinstance(state, dict) else state
        board = chess.Board(fen=None)
        board.clear()
        for row in range(8):
            for col in range(8):
                piece_val = pieces[row, col]
                if piece_val != 0:
                    piece_type = self._piece_to_chess[abs(piece_val)]
                    color = chess.WHITE if piece_val > 0 else chess.BLACK
                    square = chess.square(col, 7 - row)
                    board.set_piece_at(square, chess.Piece(piece_type, color))
        board.turn = chess.WHITE if player == 1 else chess.BLACK
        if isinstance(state, dict):
            board.ep_square = state['ep_square']
            board.castling_rights = state['castling_rights']
            board.halfmove_clock = state['halfmove_clock']
        return board

    def _board_to_state(self, board):
        pieces = np.zeros((8, 8), dtype=np.int8)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                value = self._chess_to_piece[piece.piece_type]
                if piece.color == chess.BLACK:
                    value = -value
                pieces[row, col] = value
        return {
            'pieces': pieces,
            'ep_square': board.ep_square,
            'castling_rights': board.castling_rights,
            'halfmove_clock': board.halfmove_clock
        }

    def _action_to_move(self, action, board):
        from_pos = action // 64
        to_pos = action % 64
        from_row, from_col = from_pos // 8, from_pos % 8
        to_row, to_col = to_pos // 8, to_pos % 8
        from_square = chess.square(from_col, 7 - from_row)
        to_square = chess.square(to_col, 7 - to_row)
        promotion = None
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_square)
            if (piece.color == chess.WHITE and to_rank == 7) or \
               (piece.color == chess.BLACK and to_rank == 0):
                promotion = chess.QUEEN
        return chess.Move(from_square, to_square, promotion=promotion)

    def _move_to_action(self, move):
        from_row = 7 - chess.square_rank(move.from_square)
        from_col = chess.square_file(move.from_square)
        to_row = 7 - chess.square_rank(move.to_square)
        to_col = chess.square_file(move.to_square)
        return (from_row * 8 + from_col) * 64 + (to_row * 8 + to_col)

    def get_next_state(self, state, action, player):
        board = self._state_to_board(state, player)
        move = self._action_to_move(action, board)
        if move not in board.legal_moves:
            move_promo = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if move_promo in board.legal_moves:
                move = move_promo
            else:
                raise ValueError(f"Illegal move: {move}")
        board.push(move)
        return self._board_to_state(board)

    def get_valid_moves(self, state, player=1):
        pieces = state['pieces'] if isinstance(state, dict) else state
        if len(pieces.shape) == 3:
            pieces = pieces[0]
        board = self._state_to_board(state, player=1)
        valid_moves = np.zeros(self.action_size, dtype=np.uint8)
        for move in board.legal_moves:
            if move.promotion and move.promotion != chess.QUEEN:
                continue
            action = self._move_to_action(move)
            valid_moves[action] = 1
        return valid_moves

    def check_win(self, state, action):
        if action is None:
            return False
        opponent_state = self.change_perspective(state, -1)
        board = self._state_to_board(opponent_state, player=1)
        return board.is_checkmate()

    def get_value_and_terminated(self, state, action):
        if action is not None and self.check_win(state, action):
            return 1, True
        opponent_state = self.change_perspective(state, -1)
        board = self._state_to_board(opponent_state, player=1)
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
            return 0, True
        return 0, False

    def flip_action(self, action):
        from_pos = action // 64
        to_pos = action % 64
        from_row, from_col = from_pos // 8, from_pos % 8
        to_row, to_col = to_pos // 8, to_pos % 8
        new_from_row = 7 - from_row
        new_to_row = 7 - to_row
        new_from_pos = new_from_row * 8 + from_col
        new_to_pos = new_to_row * 8 + to_col
        return new_from_pos * 64 + new_to_pos

    def change_perspective(self, state, player):
        if player == 1:
            return {k: v for k, v in state.items()} if isinstance(state, dict) else state.copy()
        pieces = np.flip(state['pieces'], axis=0) * -1 if isinstance(state, dict) else np.flip(state, axis=0) * -1
        if not isinstance(state, dict):
            return pieces
        ep_square = None
        if state['ep_square'] is not None:
            file = chess.square_file(state['ep_square'])
            rank = chess.square_rank(state['ep_square'])
            new_rank = 7 - rank
            ep_square = chess.square(file, new_rank)
        castling = state['castling_rights']
        new_castling = 0
        if castling & chess.BB_H1:
            new_castling |= chess.BB_H8
        if castling & chess.BB_A1:
            new_castling |= chess.BB_A8
        if castling & chess.BB_H8:
            new_castling |= chess.BB_H1
        if castling & chess.BB_A8:
            new_castling |= chess.BB_A1
        return {
            'pieces': pieces,
            'ep_square': ep_square,
            'castling_rights': new_castling,
            'halfmove_clock': state['halfmove_clock']
        }

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def get_encoded_state(self, state):
        pieces = state['pieces'] if isinstance(state, dict) else state
        encoded_state = np.stack((
            (pieces == self.PAWN).astype(np.float32),
            (pieces == self.KNIGHT).astype(np.float32),
            (pieces == self.BISHOP).astype(np.float32),
            (pieces == self.ROOK).astype(np.float32),
            (pieces == self.QUEEN).astype(np.float32),
            (pieces == self.KING).astype(np.float32),
            (pieces == -self.PAWN).astype(np.float32),
            (pieces == -self.KNIGHT).astype(np.float32),
            (pieces == -self.BISHOP).astype(np.float32),
            (pieces == -self.ROOK).astype(np.float32),
            (pieces == -self.QUEEN).astype(np.float32),
            (pieces == -self.KING).astype(np.float32),
            (pieces == 0).astype(np.float32)
        ))
        if isinstance(state, dict):
            castling = state['castling_rights']
            wk = np.full((8, 8), 1.0 if castling & chess.BB_H1 else 0.0)
            wq = np.full((8, 8), 1.0 if castling & chess.BB_A1 else 0.0)
            bk = np.full((8, 8), 1.0 if castling & chess.BB_H8 else 0.0)
            bq = np.full((8, 8), 1.0 if castling & chess.BB_A8 else 0.0)
            ep_plane = np.zeros((8, 8))
            if state['ep_square'] is not None:
                file = chess.square_file(state['ep_square'])
                ep_plane[:, file] = 1.0
            halfmove_plane = np.full((8, 8), state['halfmove_clock'] / 100.0)
            additional = np.stack((wk, wq, bk, bq, ep_plane, halfmove_plane))
            encoded_state = np.concatenate((encoded_state, additional))
        if len(pieces.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        return encoded_state

    def action_to_coords(self, action):
        from_pos = action // 64
        to_pos = action % 64
        return (from_pos // 8, from_pos % 8, to_pos // 8, to_pos % 8)

    def coords_to_action(self, from_row, from_col, to_row, to_col):
        return (from_row * 8 + from_col) * 64 + (to_row * 8 + to_col)

    def move_to_algebraic(self, action):
        from_row, from_col, to_row, to_col = self.action_to_coords(action)
        files = 'abcdefgh'
        ranks = '87654321'
        return f"{files[from_col]}{ranks[from_row]}{files[to_col]}{ranks[to_row]}"

    def algebraic_to_move(self, notation):
        files = 'abcdefgh'
        ranks = '87654321'
        from_col = files.index(notation[0])
        from_row = ranks.index(notation[1])
        to_col = files.index(notation[2])
        to_row = ranks.index(notation[3])
        return self.coords_to_action(from_row, from_col, to_row, to_col)

    def print_board(self, state):
        pieces = state['pieces'] if isinstance(state, dict) else state
        symbols = {
            0: '.', self.PAWN: 'P', self.KNIGHT: 'N', self.BISHOP: 'B',
            self.ROOK: 'R', self.QUEEN: 'Q', self.KING: 'K',
            -self.PAWN: 'p', self.KNIGHT: 'n', self.BISHOP: 'b',
            -self.ROOK: 'r', self.QUEEN: 'q', self.KING: 'k'
        }
        print("  a b c d e f g h")
        for row in range(8):
            print(f"{8 - row} ", end="")
            for col in range(8):
                print(symbols.get(pieces[row, col], '?') + " ", end="")
            print(f"{8 - row}")
        print("  a b c d e f g h\n")

    def state_to_fen(self, state, player=1):
        return self._state_to_board(state, player).fen()