import chess
import numpy as np


class Chess:
    """
    Класс игры Шахматы с использованием python-chess для валидации ходов.
    Интерфейс совместим с AlphaZero-style обучением.
    """

    # Константы для фигур
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
        self.shape_obs = 13  # 6 своих + 6 чужих + пустые

        # Действие: from_pos * 64 + to_pos (превращение всегда в ферзя)
        self.action_size = 64 * 64

        # Маппинг между нашими константами и python-chess
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
        """
        Возвращает начальное состояние доски 8x8.
        Белые (положительные) внизу (ряды 6-7), чёрные (отрицательные) вверху (0-1).
        """
        state = np.zeros((self.row_count, self.column_count), dtype=np.int8)

        # Чёрные (противник, вверху)
        state[0] = [-self.ROOK, -self.KNIGHT, -self.BISHOP, -self.QUEEN,
                    -self.KING, -self.BISHOP, -self.KNIGHT, -self.ROOK]
        state[1] = [-self.PAWN] * 8

        # Белые (игрок, внизу)
        state[6] = [self.PAWN] * 8
        state[7] = [self.ROOK, self.KNIGHT, self.BISHOP, self.QUEEN,
                    self.KING, self.BISHOP, self.KNIGHT, self.ROOK]

        return state

    # ==================== Методы конвертации ====================

    def _state_to_board(self, state):
        """Конвертирует numpy state в chess.Board"""
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

        # Положительные фигуры = WHITE, они ходят
        board.turn = chess.WHITE

        # Права рокировки на основе позиций фигур
        # (упрощение: если фигуры на местах, рокировка разрешена)
        board.castling_rights = chess.BB_EMPTY

        # Для текущего игрока (положительные = WHITE)
        if state[7, 4] == self.KING:
            if state[7, 7] == self.ROOK:
                board.castling_rights |= chess.BB_H1
            if state[7, 0] == self.ROOK:
                board.castling_rights |= chess.BB_A1

        # Для противника (отрицательные = BLACK)
        if state[0, 4] == -self.KING:
            if state[0, 7] == -self.ROOK:
                board.castling_rights |= chess.BB_H8
            if state[0, 0] == -self.ROOK:
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

    def _coords_to_square(self, row, col):
        """Конвертирует координаты (row, col) в chess.Square"""
        return chess.square(col, 7 - row)

    def _square_to_coords(self, square):
        """Конвертирует chess.Square в координаты (row, col)"""
        return (7 - chess.square_rank(square), chess.square_file(square))

    def _action_to_move(self, action, board):
        """Конвертирует action в chess.Move"""
        from_pos = action // 64
        to_pos = action % 64
        from_row, from_col = from_pos // 8, from_pos % 8
        to_row, to_col = to_pos // 8, to_pos % 8

        from_square = self._coords_to_square(from_row, from_col)
        to_square = self._coords_to_square(to_row, to_col)

        # Проверяем превращение пешки
        promotion = None
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_square)
            if (piece.color == chess.WHITE and to_rank == 7) or \
               (piece.color == chess.BLACK and to_rank == 0):
                promotion = chess.QUEEN  # Всегда в ферзя

        return chess.Move(from_square, to_square, promotion=promotion)

    def _move_to_action(self, move):
        """Конвертирует chess.Move в action"""
        from_row, from_col = self._square_to_coords(move.from_square)
        to_row, to_col = self._square_to_coords(move.to_square)
        from_pos = from_row * 8 + from_col
        to_pos = to_row * 8 + to_col
        return from_pos * 64 + to_pos

    # ==================== Основные методы игры ====================

    def get_valid_moves(self, state):
        """
        Возвращает маску допустимых ходов размером action_size.
        Использует python-chess для полной валидации (шах, рокировка и т.д.).
        """
        if len(state.shape) == 3:
            state = state[0]

        board = self._state_to_board(state)
        valid_moves = np.zeros(self.action_size, dtype=np.uint8)

        for move in board.legal_moves:
            # Пропускаем превращения не в ферзя
            if move.promotion and move.promotion != chess.QUEEN:
                continue
            action = self._move_to_action(move)
            valid_moves[action] = 1

        return valid_moves

    def get_next_state(self, state, action, player):
        """Применяет действие и возвращает новое состояние"""
        board = self._state_to_board(state)
        move = self._action_to_move(action, board)
        board.push(move)
        return self._board_to_state(board)

    def check_win(self, state, action):
        """
        Проверяет победу после хода.
        Победа если противник получил мат.
        """
        if action is None:
            return False

        # Проверяем состояние противника
        opponent_state = self.change_perspective(state, -1)
        board = self._state_to_board(opponent_state)

        return board.is_checkmate()

    def get_value_and_terminated(self, state, action):
        """
        Возвращает (value, terminated).
        value = 1 при победе (мат), 0 при ничьей.
        """
        # Проверяем мат противнику
        if self.check_win(state, action):
            return 1, True

        # Проверяем состояние противника для пата и других ничьих
        opponent_state = self.change_perspective(state, -1)
        board = self._state_to_board(opponent_state)

        # Пат
        if board.is_stalemate():
            return 0, True

        # Недостаток материала
        if board.is_insufficient_material():
            return 0, True

        # Дополнительная проверка: только два короля
        pieces = [p for p in board.piece_map().values()]
        if len(pieces) == 2:
            return 0, True

        return 0, False

    def flip_action(self, action):
        """Переворачивает action после change_perspective"""
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
        """Меняет перспективу доски для игрока"""
        if player == -1:
            return np.flip(state, axis=0) * -1
        return state.copy()

    def get_opponent(self, player):
        """Возвращает противника"""
        return -player

    def get_opponent_value(self, value):
        """Возвращает значение с точки зрения противника"""
        return -value

    def get_encoded_state(self, state):
        """
        Кодирует состояние для нейронной сети (13 каналов).
        """
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
        """Декодирует action в (from_row, from_col, to_row, to_col)"""
        from_pos = action // 64
        to_pos = action % 64
        return (from_pos // 8, from_pos % 8, to_pos // 8, to_pos % 8)

    def coords_to_action(self, from_row, from_col, to_row, to_col):
        """Кодирует координаты в action"""
        from_pos = from_row * 8 + from_col
        to_pos = to_row * 8 + to_col
        return from_pos * 64 + to_pos

    def print_board(self, state):
        """Красивый вывод доски в консоль"""
        symbols = {
            0: '.',
            self.PAWN: 'P', self.KNIGHT: 'N', self.BISHOP: 'B',
            self.ROOK: 'R', self.QUEEN: 'Q', self.KING: 'K',
            -self.PAWN: 'p', -self.KNIGHT: 'n', -self.BISHOP: 'b',
            -self.ROOK: 'r', -self.QUEEN: 'q', -self.KING: 'k'
        }

        print("  a b c d e f g h")
        for row in range(self.row_count):
            print(f"{8 - row} ", end="")
            for col in range(self.column_count):
                print(symbols.get(state[row, col], '?') + " ", end="")
            print(f"{8 - row}")
        print("  a b c d e f g h")
        print()

    def move_to_algebraic(self, action):
        """Конвертирует action в алгебраическую нотацию (e2e4)"""
        from_row, from_col, to_row, to_col = self.action_to_coords(action)
        files = 'abcdefgh'
        ranks = '87654321'
        return f"{files[from_col]}{ranks[from_row]}{files[to_col]}{ranks[to_row]}"

    def algebraic_to_move(self, notation):
        """Конвертирует алгебраическую нотацию в action"""
        files = 'abcdefgh'
        ranks = '87654321'
        from_col = files.index(notation[0])
        from_row = ranks.index(notation[1])
        to_col = files.index(notation[2])
        to_row = ranks.index(notation[3])
        return self.coords_to_action(from_row, from_col, to_row, to_col)

    # ==================== Дополнительные методы для FEN ====================

    def state_to_fen(self, state):
        """Конвертирует state в FEN строку (для отладки/экспорта)"""
        board = self._state_to_board(state)
        return board.fen()

    def fen_to_state(self, fen):
        """Конвертирует FEN строку в state"""
        board = chess.Board(fen)
        return self._board_to_state(board)

    def is_in_check(self, state):
        """Проверяет, находится ли текущий игрок под шахом"""
        board = self._state_to_board(state)
        return board.is_check()