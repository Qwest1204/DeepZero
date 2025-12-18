import chess
import numpy as np


class Chess:
    """
    Класс игры Шахматы с использованием python-chess.
    """

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
        self.shape_obs = 13
        self.action_size = 64 * 64

        self._piece_to_chess = {
            self.PAWN: chess.PAWN,
            self.KNIGHT: chess.KNIGHT,
            self.BISHOP: chess.BISHOP,
            self.ROOK: chess.ROOK,
            self.QUEEN: chess.QUEEN,
            self.KING: chess.KING,
        }
        self._chess_to_piece = {v: k for k, v in self._piece_to_chess.items()}

    def __repr__(self):
        return "Chess"

    def get_initial_state(self):
        """Начальное состояние: белые внизу (положительные), чёрные вверху (отрицательные)"""
        state = np.zeros((8, 8), dtype=np.int8)
        state[0] = [-self.ROOK, -self.KNIGHT, -self.BISHOP, -self.QUEEN,
                    -self.KING, -self.BISHOP, -self.KNIGHT, -self.ROOK]
        state[1] = [-self.PAWN] * 8
        state[6] = [self.PAWN] * 8
        state[7] = [self.ROOK, self.KNIGHT, self.BISHOP, self.QUEEN,
                    self.KING, self.BISHOP, self.KNIGHT, self.ROOK]
        return state

    # ==================== Конвертация ====================

    def _state_to_board(self, state, player=1):
        """
        Конвертирует numpy state в chess.Board.
        player указывает кто сейчас ходит.
        """
        board = chess.Board(fen=None)
        board.clear()

        for row in range(8):
            for col in range(8):
                piece_val = state[row, col]
                if piece_val != 0:
                    piece_type = self._piece_to_chess[abs(piece_val)]
                    color = chess.WHITE if piece_val > 0 else chess.BLACK
                    square = chess.square(col, 7 - row)
                    board.set_piece_at(square, chess.Piece(piece_type, color))

        # Устанавливаем чей ход
        board.turn = chess.WHITE if player == 1 else chess.BLACK

        # Права рокировки
        board.castling_rights = chess.BB_EMPTY

        # Белые рокировки
        if state[7, 4] == self.KING:  # Король на e1
            if state[7, 7] == self.ROOK:  # Ладья на h1
                board.castling_rights |= chess.BB_H1
            if state[7, 0] == self.ROOK:  # Ладья на a1
                board.castling_rights |= chess.BB_A1

        # Чёрные рокировки
        if state[0, 4] == -self.KING:  # Король на e8
            if state[0, 7] == -self.ROOK:  # Ладья на h8
                board.castling_rights |= chess.BB_H8
            if state[0, 0] == -self.ROOK:  # Ладья на a8
                board.castling_rights |= chess.BB_A8

        return board

    def _board_to_state(self, board):
        """Конвертирует chess.Board в numpy state"""
        state = np.zeros((8, 8), dtype=np.int8)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                value = self._chess_to_piece[piece.piece_type]
                if piece.color == chess.BLACK:
                    value = -value
                state[row, col] = value

        return state

    def _action_to_move(self, action, board):
        """Конвертирует action в chess.Move"""
        from_pos = action // 64
        to_pos = action % 64
        from_row, from_col = from_pos // 8, from_pos % 8
        to_row, to_col = to_pos // 8, to_pos % 8

        from_square = chess.square(from_col, 7 - from_row)
        to_square = chess.square(to_col, 7 - to_row)

        # Проверяем превращение пешки
        promotion = None
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_square)
            if (piece.color == chess.WHITE and to_rank == 7) or \
                    (piece.color == chess.BLACK and to_rank == 0):
                promotion = chess.QUEEN

        return chess.Move(from_square, to_square, promotion=promotion)

    def _move_to_action(self, move):
        """Конвертирует chess.Move в action"""
        from_row = 7 - chess.square_rank(move.from_square)
        from_col = chess.square_file(move.from_square)
        to_row = 7 - chess.square_rank(move.to_square)
        to_col = chess.square_file(move.to_square)
        return (from_row * 8 + from_col) * 64 + (to_row * 8 + to_col)

    # ==================== Основные методы ====================

    def get_next_state(self, state, action, player):
        """
        Применяет action к state.

        ВАЖНО: action должен быть в координатах оригинального state,
        то есть если player=-1, action должен быть уже перевёрнут через flip_action.
        """
        board = self._state_to_board(state, player)
        move = self._action_to_move(action, board)

        # Проверяем легальность
        if move not in board.legal_moves:
            # Пробуем с превращением
            move_with_promo = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if move_with_promo in board.legal_moves:
                move = move_with_promo
            else:
                # Для отладки
                print(f"ERROR: Illegal move {move} for player {player}")
                print(f"Board:\n{board}")
                print(f"Legal moves: {list(board.legal_moves)}")
                raise ValueError(f"Illegal move: {move}")

        board.push(move)
        return self._board_to_state(board)

    def get_valid_moves(self, state, player=1):
        """
        Возвращает маску валидных ходов.

        state - состояние с перспективы player (после change_perspective)
        Для MCTS это neutral_state где текущий игрок положительный.
        """
        if len(state.shape) == 3:
            state = state[0]

        # Для neutral state текущий игрок всегда положительный = WHITE
        board = self._state_to_board(state, player=1)
        valid_moves = np.zeros(self.action_size, dtype=np.uint8)

        for move in board.legal_moves:
            if move.promotion and move.promotion != chess.QUEEN:
                continue
            action = self._move_to_action(move)
            valid_moves[action] = 1

        return valid_moves

    def check_win(self, state, action):
        """Проверяет победу после хода (мат противнику)"""
        if action is None:
            return False

        # После хода проверяем состояние с точки зрения противника
        opponent_state = self.change_perspective(state, -1)
        board = self._state_to_board(opponent_state, player=1)

        return board.is_checkmate()

    def get_value_and_terminated(self, state, action):
        """
        Проверяет окончание игры.
        Вызывается ПОСЛЕ применения хода.
        state - результат get_next_state
        """
        # Проверяем мат противнику
        if action is not None and self.check_win(state, action):
            return 1, True

        # Проверяем состояние для следующего хода
        # После get_next_state ходит противник, но state ещё не перевёрнут
        # Нужно проверить может ли противник ходить
        opponent_state = self.change_perspective(state, -1)
        board = self._state_to_board(opponent_state, player=1)

        if board.is_stalemate():
            return 0, True

        if board.is_insufficient_material():
            return 0, True

        # Правило 50 ходов (опционально)
        if board.is_fifty_moves():
            return 0, True

        # Троекратное повторение (сложнее отследить без истории)

        return 0, False

    def flip_action(self, action):
        """Переворачивает action (зеркально по вертикали)"""
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
        """Меняет перспективу: переворачивает доску и знаки фигур"""
        if player == -1:
            return np.flip(state, axis=0) * -1
        return state.copy()

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def get_encoded_state(self, state):
        """Кодирует состояние для нейронной сети (13 каналов)"""
        encoded_state = np.stack(
            (
                (state == self.PAWN).astype(np.float32),
                (state == self.KNIGHT).astype(np.float32),
                (state == self.BISHOP).astype(np.float32),
                (state == self.ROOK).astype(np.float32),
                (state == self.QUEEN).astype(np.float32),
                (state == self.KING).astype(np.float32),
                (state == -self.PAWN).astype(np.float32),
                (state == -self.KNIGHT).astype(np.float32),
                (state == -self.BISHOP).astype(np.float32),
                (state == -self.ROOK).astype(np.float32),
                (state == -self.QUEEN).astype(np.float32),
                (state == -self.KING).astype(np.float32),
                (state == 0).astype(np.float32)
            )
        )

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

    # ==================== Вспомогательные методы ====================

    def action_to_coords(self, action):
        from_pos = action // 64
        to_pos = action % 64
        return (from_pos // 8, from_pos % 8, to_pos // 8, to_pos % 8)

    def coords_to_action(self, from_row, from_col, to_row, to_col):
        from_pos = from_row * 8 + from_col
        to_pos = to_row * 8 + to_col
        return from_pos * 64 + to_pos

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
        symbols = {
            0: '.',
            self.PAWN: 'P', self.KNIGHT: 'N', self.BISHOP: 'B',
            self.ROOK: 'R', self.QUEEN: 'Q', self.KING: 'K',
            -self.PAWN: 'p', -self.KNIGHT: 'n', -self.BISHOP: 'b',
            -self.ROOK: 'r', -self.QUEEN: 'q', -self.KING: 'k'
        }

        print("  a b c d e f g h")
        for row in range(8):
            print(f"{8 - row} ", end="")
            for col in range(8):
                print(symbols.get(state[row, col], '?') + " ", end="")
            print(f"{8 - row}")
        print("  a b c d e f g h\n")

    def state_to_fen(self, state, player=1):
        board = self._state_to_board(state, player)
        return board.fen()

    def fen_to_state(self, fen):
        board = chess.Board(fen)
        return self._board_to_state(board)