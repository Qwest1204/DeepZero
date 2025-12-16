import numpy as np


class Chess:
    """
    Класс игры Шахматы
    Структура аналогична Checkers, TicTacToe и ConnectFour
    """

    # Константы для фигур
    EMPTY = 0
    PAWN = 1      # Пешка
    KNIGHT = 2   # Конь
    BISHOP = 3   # Слон
    ROOK = 4     # Ладья
    QUEEN = 5    # Ферзь
    KING = 6     # Король

    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.shape_obs = 13  # 6 своих фигур + 6 чужих + пустые

        # Действие кодируется как: from_pos * 64 + to_pos
        # Превращение пешки: всегда в ферзя (упрощение)
        self.action_size = 64 * 64  # 4096 возможных действий

        # Направления движения для фигур
        self.ROOK_DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.BISHOP_DIRECTIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        self.QUEEN_DIRECTIONS = self.ROOK_DIRECTIONS + self.BISHOP_DIRECTIONS
        self.KNIGHT_MOVES = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        self.KING_MOVES = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0), (1, 1)
        ]

    def __repr__(self):
        return "Chess"

    def get_initial_state(self):
        """
        Возвращает начальное состояние доски 8x8
        Белые (положительные) внизу (ряды 6-7), чёрные (отрицательные) вверху (ряды 0-1)
        """
        state = np.zeros((self.row_count, self.column_count), dtype=np.int8)

        # Чёрные фигуры (противник, вверху)
        state[0] = [-self.ROOK, -self.KNIGHT, -self.BISHOP, -self.QUEEN,
                    -self.KING, -self.BISHOP, -self.KNIGHT, -self.ROOK]
        state[1] = [-self.PAWN] * 8

        # Белые фигуры (игрок, внизу)
        state[6] = [self.PAWN] * 8
        state[7] = [self.ROOK, self.KNIGHT, self.BISHOP, self.QUEEN,
                    self.KING, self.BISHOP, self.KNIGHT, self.ROOK]

        return state

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

    def get_next_state(self, state, action, player):
        """
        Применяет действие к состоянию и возвращает новое состояние.
        Состояние должно быть с перспективы player (после change_perspective).
        """
        state = state.copy()

        from_pos = action // 64
        to_pos = action % 64

        from_row, from_col = from_pos // 8, from_pos % 8
        to_row, to_col = to_pos // 8, to_pos % 8

        piece = state[from_row, from_col]
        state[from_row, from_col] = 0

        # Превращение пешки (достигла верхнего края)
        if piece == self.PAWN and to_row == 0:
            state[to_row, to_col] = self.QUEEN  # Всегда превращаем в ферзя
        # Рокировка (король двигается на 2 клетки)
        elif abs(piece) == self.KING and abs(to_col - from_col) == 2:
            state[to_row, to_col] = piece
            # Перемещаем ладью
            if to_col > from_col:  # Короткая рокировка (вправо)
                state[to_row, 5] = state[to_row, 7]
                state[to_row, 7] = 0
            else:  # Длинная рокировка (влево)
                state[to_row, 3] = state[to_row, 0]
                state[to_row, 0] = 0
        else:
            state[to_row, to_col] = piece

        return state

    def get_valid_moves(self, state):
        """
        Возвращает маску допустимых ходов размером action_size.
        Предполагается что state с перспективы текущего игрока (player=1).
        Фильтрует ходы, которые оставляют короля под шахом.
        """
        if len(state.shape) == 3:
            state = state[0]

        valid_moves = np.zeros(self.action_size, dtype=np.uint8)

        for row in range(self.row_count):
            for col in range(self.column_count):
                piece = state[row, col]

                # Ищем только свои фигуры (положительные)
                if piece <= 0:
                    continue

                # Получаем все псевдо-легальные ходы для фигуры
                moves = self._get_piece_moves(state, row, col, piece)

                for to_row, to_col in moves:
                    action = self.coords_to_action(row, col, to_row, to_col)

                    # Проверяем, не оставляет ли ход короля под шахом
                    if not self._leaves_king_in_check(state, row, col, to_row, to_col):
                        valid_moves[action] = 1

        return valid_moves

    def _get_piece_moves(self, state, row, col, piece):
        """Возвращает список псевдо-легальных ходов для фигуры (без проверки шаха)"""
        moves = []

        if piece == self.PAWN:
            moves = self._get_pawn_moves(state, row, col)
        elif piece == self.KNIGHT:
            moves = self._get_knight_moves(state, row, col)
        elif piece == self.BISHOP:
            moves = self._get_sliding_moves(state, row, col, self.BISHOP_DIRECTIONS)
        elif piece == self.ROOK:
            moves = self._get_sliding_moves(state, row, col, self.ROOK_DIRECTIONS)
        elif piece == self.QUEEN:
            moves = self._get_sliding_moves(state, row, col, self.QUEEN_DIRECTIONS)
        elif piece == self.KING:
            moves = self._get_king_moves(state, row, col)

        return moves

    def _get_pawn_moves(self, state, row, col):
        """Ходы пешки (игрок всегда идёт вверх, row уменьшается)"""
        moves = []

        # Ход вперёд на 1
        if row > 0 and state[row - 1, col] == 0:
            moves.append((row - 1, col))

            # Ход вперёд на 2 с начальной позиции
            if row == 6 and state[row - 2, col] == 0:
                moves.append((row - 2, col))

        # Взятие по диагонали
        for dc in [-1, 1]:
            new_col = col + dc
            if 0 <= new_col < 8 and row > 0:
                target = state[row - 1, new_col]
                if target < 0:  # Фигура противника
                    moves.append((row - 1, new_col))

        return moves

    def _get_knight_moves(self, state, row, col):
        """Ходы коня"""
        moves = []
        for dr, dc in self.KNIGHT_MOVES:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_position(new_row, new_col):
                target = state[new_row, new_col]
                if target <= 0:  # Пусто или фигура противника
                    moves.append((new_row, new_col))
        return moves

    def _get_sliding_moves(self, state, row, col, directions):
        """Ходы скользящих фигур (слон, ладья, ферзь)"""
        moves = []
        for dr, dc in directions:
            for dist in range(1, 8):
                new_row, new_col = row + dr * dist, col + dc * dist
                if not self._is_valid_position(new_row, new_col):
                    break
                target = state[new_row, new_col]
                if target == 0:
                    moves.append((new_row, new_col))
                elif target < 0:  # Фигура противника - можно взять
                    moves.append((new_row, new_col))
                    break
                else:  # Своя фигура - стоп
                    break
        return moves

    def _get_king_moves(self, state, row, col):
        """Ходы короля включая рокировку"""
        moves = []

        # Обычные ходы
        for dr, dc in self.KING_MOVES:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_position(new_row, new_col):
                target = state[new_row, new_col]
                if target <= 0:  # Пусто или фигура противника
                    moves.append((new_row, new_col))

        # Рокировка (упрощённая проверка - король на начальной позиции)
        if row == 7 and col == 4:
            # Короткая рокировка
            if (state[7, 5] == 0 and state[7, 6] == 0 and
                state[7, 7] == self.ROOK):
                # Проверяем, что король не под шахом и не проходит через шах
                if (not self._is_square_attacked(state, 7, 4) and
                    not self._is_square_attacked(state, 7, 5) and
                    not self._is_square_attacked(state, 7, 6)):
                    moves.append((7, 6))

            # Длинная рокировка
            if (state[7, 3] == 0 and state[7, 2] == 0 and
                state[7, 1] == 0 and state[7, 0] == self.ROOK):
                if (not self._is_square_attacked(state, 7, 4) and
                    not self._is_square_attacked(state, 7, 3) and
                    not self._is_square_attacked(state, 7, 2)):
                    moves.append((7, 2))

        return moves

    def _is_square_attacked(self, state, row, col):
        """Проверяет, атакуется ли клетка противником"""

        # Атака пешкой (противник идёт вниз, значит атакует сверху)
        for dc in [-1, 1]:
            r, c = row - 1, col + dc
            if self._is_valid_position(r, c) and state[r, c] == -self.PAWN:
                return True

        # Атака конём
        for dr, dc in self.KNIGHT_MOVES:
            r, c = row + dr, col + dc
            if self._is_valid_position(r, c) and state[r, c] == -self.KNIGHT:
                return True

        # Атака королём
        for dr, dc in self.KING_MOVES:
            r, c = row + dr, col + dc
            if self._is_valid_position(r, c) and state[r, c] == -self.KING:
                return True

        # Атака по диагонали (слон, ферзь)
        for dr, dc in self.BISHOP_DIRECTIONS:
            for dist in range(1, 8):
                r, c = row + dr * dist, col + dc * dist
                if not self._is_valid_position(r, c):
                    break
                piece = state[r, c]
                if piece == -self.BISHOP or piece == -self.QUEEN:
                    return True
                if piece != 0:
                    break

        # Атака по прямой (ладья, ферзь)
        for dr, dc in self.ROOK_DIRECTIONS:
            for dist in range(1, 8):
                r, c = row + dr * dist, col + dc * dist
                if not self._is_valid_position(r, c):
                    break
                piece = state[r, c]
                if piece == -self.ROOK or piece == -self.QUEEN:
                    return True
                if piece != 0:
                    break

        return False

    def _find_king(self, state, player_sign=1):
        """Находит позицию короля"""
        king_value = self.KING * player_sign
        positions = np.where(state == king_value)
        if len(positions[0]) > 0:
            return positions[0][0], positions[1][0]
        return None

    def _leaves_king_in_check(self, state, from_row, from_col, to_row, to_col):
        """Проверяет, оставляет ли ход своего короля под шахом"""
        # Симулируем ход
        temp_state = state.copy()
        piece = temp_state[from_row, from_col]
        temp_state[from_row, from_col] = 0
        temp_state[to_row, to_col] = piece

        # Рокировка - двигаем ладью тоже
        if abs(piece) == self.KING and abs(to_col - from_col) == 2:
            if to_col > from_col:  # Короткая
                temp_state[to_row, 5] = temp_state[to_row, 7]
                temp_state[to_row, 7] = 0
            else:  # Длинная
                temp_state[to_row, 3] = temp_state[to_row, 0]
                temp_state[to_row, 0] = 0

        # Находим короля
        king_pos = self._find_king(temp_state, 1)
        if king_pos is None:
            return True

        return self._is_square_attacked(temp_state, king_pos[0], king_pos[1])

    def _is_in_check(self, state):
        """Проверяет, находится ли текущий игрок под шахом"""
        king_pos = self._find_king(state, 1)
        if king_pos is None:
            return True
        return self._is_square_attacked(state, king_pos[0], king_pos[1])

    def check_win(self, state, action):
        """
        Проверяет победу после хода.
        Победа если противник получил мат (под шахом и нет ходов).
        """
        if action is None:
            return False

        # Меняем перспективу чтобы проверить состояние противника
        opponent_state = self.change_perspective(state, -1)

        # Проверяем есть ли у противника допустимые ходы
        opponent_moves = self.get_valid_moves(opponent_state)
        if np.sum(opponent_moves) == 0:
            # Нет ходов - это мат или пат
            if self._is_in_check(opponent_state):
                return True  # Мат - победа!

        return False

    def get_value_and_terminated(self, state, action):
        """
        Возвращает (value, terminated).
        value = 1 при победе (мат), 0 при ничьей (пат) или продолжении.
        """
        if self.check_win(state, action):
            return 1, True

        # Проверяем пат (нет ходов, но не под шахом)
        valid_moves = self.get_valid_moves(state)
        if np.sum(valid_moves) == 0:
            if not self._is_in_check(state):
                return 0, True  # Пат - ничья
            else:
                # Текущий игрок в мате - но это значит предыдущий победил
                # Это не должно происходить если логика правильная
                return -1, True

        # Проверка на недостаток материала (упрощённая)
        # Король против короля
        pieces = state[state != 0]
        if len(pieces) == 2:  # Только два короля
            return 0, True

        return 0, False

    def get_opponent(self, player):
        """Возвращает противника"""
        return -player

    def get_opponent_value(self, value):
        """Возвращает значение с точки зрения противника"""
        return -value

    def change_perspective(self, state, player):
        """
        Меняет перспективу доски для игрока.
        Переворачивает доску и инвертирует знаки фигур.
        """
        if player == -1:
            return np.flip(state, axis=0) * -1
        return state.copy()

    def get_encoded_state(self, state):
        """
        Кодирует состояние для нейронной сети.
        Возвращает 13 каналов:
        - 6 каналов для своих фигур (P, N, B, R, Q, K)
        - 6 каналов для фигур противника
        - 1 канал для пустых клеток
        """
        encoded_state = np.stack(
            (
                (state == self.PAWN).astype(np.float32),    # Мои пешки
                (state == self.KNIGHT).astype(np.float32),  # Мои кони
                (state == self.BISHOP).astype(np.float32),  # Мои слоны
                (state == self.ROOK).astype(np.float32),    # Мои ладьи
                (state == self.QUEEN).astype(np.float32),   # Мой ферзь
                (state == self.KING).astype(np.float32),    # Мой король
                (state == -self.PAWN).astype(np.float32),   # Пешки противника
                (state == -self.KNIGHT).astype(np.float32), # Кони противника
                (state == -self.BISHOP).astype(np.float32), # Слоны противника
                (state == -self.ROOK).astype(np.float32),   # Ладьи противника
                (state == -self.QUEEN).astype(np.float32),  # Ферзь противника
                (state == -self.KING).astype(np.float32),   # Король противника
                (state == 0).astype(np.float32)             # Пустые клетки
            )
        )

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

    def _is_valid_position(self, row, col):
        """Проверяет, находится ли позиция в пределах доски"""
        return 0 <= row < self.row_count and 0 <= col < self.column_count

    def action_to_coords(self, action):
        """Декодирует действие в координаты (from_row, from_col, to_row, to_col)"""
        from_pos = action // 64
        to_pos = action % 64
        return (from_pos // 8, from_pos % 8, to_pos // 8, to_pos % 8)

    def coords_to_action(self, from_row, from_col, to_row, to_col):
        """Кодирует координаты в действие"""
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
        """Конвертирует action в алгебраическую нотацию"""
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
