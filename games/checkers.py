import numpy as np

class Checkers:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.shape_obs = 5
        # Действие кодируется как: from_pos * 64 + to_pos
        # from_pos = row * 8 + col, to_pos = row * 8 + col
        self.action_size = 64 * 64  # 4096 возможных действий
        # Фигуры: 1 = шашка текущего игрока, 2 = дамка текущего игрока
        #        -1 = шашка противника, -2 = дамка противника

    def __repr__(self):
        return "Checkers"

    def get_initial_state(self):
        """Возвращает начальное состояние доски 8x8"""
        state = np.zeros((self.row_count, self.column_count), dtype=np.int8)

        # Расставляем шашки только на тёмных клетках (row + col) % 2 == 1
        for row in range(self.row_count):
            for col in range(self.column_count):
                if (row + col) % 2 == 1:
                    if row < 3:
                        state[row, col] = -1  # Шашки противника (вверху)
                    elif row > 4:
                        state[row, col] = 1   # Шашки игрока (внизу)

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

        # Проверяем взятие (прыжок через 2 клетки)
        if abs(to_row - from_row) == 2:
            captured_row = (from_row + to_row) // 2
            captured_col = (from_col + to_col) // 2
            state[captured_row, captured_col] = 0

        # Превращение в дамку при достижении верхнего края (row == 0)
        # Игрок 1 всегда двигается вверх
        if piece == 1 and to_row == 0:
            state[to_row, to_col] = 2  # Дамка
        else:
            state[to_row, to_col] = piece

        return state

    def get_valid_moves(self, state):
        """
        Возвращает маску допустимых ходов размером action_size.
        Если есть взятия - возвращает только взятия (взятие обязательно).
        Предполагается что state с перспективы текущего игрока (player=1).

        Поддерживает только одиночное состояние (2D array).
        """
        # Защита от batch входа - берём только первое состояние если это batch
        if len(state.shape) == 3:
            state = state[0]

        valid_moves = np.zeros(self.action_size, dtype=np.uint8)
        captures = []
        regular_moves = []

        for row in range(self.row_count):
            for col in range(self.column_count):
                piece = state[row, col]

                # Ищем фигуры текущего игрока (1 = шашка, 2 = дамка)
                if piece != 1 and piece != 2:
                    continue

                is_king = piece == 2

                # Направления движения: шашка вверх, дамка в обе стороны
                if is_king:
                    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                else:
                    directions = [(-1, -1), (-1, 1)]  # Только вверх

                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc

                    if not self._is_valid_position(new_row, new_col):
                        continue

                    target = state[new_row, new_col]

                    if target == 0:
                        # Обычный ход
                        from_pos = row * 8 + col
                        to_pos = new_row * 8 + new_col
                        action = from_pos * 64 + to_pos
                        regular_moves.append(action)

                    elif target == -1 or target == -2:
                        # Возможное взятие (фигура противника)
                        jump_row, jump_col = new_row + dr, new_col + dc
                        if self._is_valid_position(jump_row, jump_col) and state[jump_row, jump_col] == 0:
                            from_pos = row * 8 + col
                            to_pos = jump_row * 8 + jump_col
                            action = from_pos * 64 + to_pos
                            captures.append(action)

                # Для обычной шашки проверяем взятие назад
                if not is_king:
                    for dr, dc in [(1, -1), (1, 1)]:  # Назад
                        new_row, new_col = row + dr, col + dc

                        if not self._is_valid_position(new_row, new_col):
                            continue

                        target = state[new_row, new_col]

                        if target == -1 or target == -2:
                            jump_row, jump_col = new_row + dr, new_col + dc
                            if self._is_valid_position(jump_row, jump_col) and state[jump_row, jump_col] == 0:
                                from_pos = row * 8 + col
                                to_pos = jump_row * 8 + jump_col
                                action = from_pos * 64 + to_pos
                                captures.append(action)

        # Взятие обязательно
        moves_to_use = captures if len(captures) > 0 else regular_moves

        for action in moves_to_use:
            valid_moves[action] = 1

        return valid_moves

    def check_win(self, state, action):
        """
        Проверяет победу после хода.
        Победа если у противника нет фигур или нет допустимых ходов.
        """
        if action is None:
            return False

        # Проверяем, есть ли фигуры у противника
        opponent_pieces = np.sum((state == -1) | (state == -2))
        if opponent_pieces == 0:
            return True

        # Проверяем, есть ли допустимые ходы у противника
        # Меняем перспективу и проверяем ходы
        opponent_state = self.change_perspective(state, -1)
        opponent_moves = self.get_valid_moves(opponent_state)
        if np.sum(opponent_moves) == 0:
            return True

        return False

    def get_value_and_terminated(self, state, action):
        """
        Возвращает (value, terminated).
        value = 1 при победе текущего игрока, 0 при ничьей или продолжении.
        """
        if self.check_win(state, action):
            return 1, True

        # Проверка на ничью (нет ходов у текущего игрока)
        if np.sum(self.get_valid_moves(state)) == 0:
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
        После вызова текущий игрок всегда представлен как player=1.
        Доска переворачивается и значения инвертируются.
        """
        if player == -1:
            # Переворачиваем доску и меняем знаки фигур
            return np.flip(state, axis=0) * -1
        return state.copy()

    def get_encoded_state(self, state):
        """
        Кодирует состояние для нейронной сети.
        Возвращает 5 каналов:
        - Шашки текущего игрока (1)
        - Дамки текущего игрока (2)
        - Шашки противника (-1)
        - Дамки противника (-2)
        - Пустые клетки (0)
        """
        encoded_state = np.stack(
            (
                (state == 1).astype(np.float32),   # Мои шашки
                (state == 2).astype(np.float32),   # Мои дамки
                (state == -1).astype(np.float32),  # Шашки противника
                (state == -2).astype(np.float32),  # Дамки противника
                (state == 0).astype(np.float32)    # Пустые клетки
            )
        )

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

    def _is_valid_position(self, row, col):
        """Проверяет, находится ли позиция в пределах доски"""
        return 0 <= row < self.row_count and 0 <= col < self.column_count

    def has_additional_captures(self, state, row, col):
        """Проверяет, есть ли дополнительные взятия для шашки после хода"""
        piece = state[row, col]
        if piece != 1 and piece != 2:
            return False

        # Все направления для взятия
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if not self._is_valid_position(new_row, new_col):
                continue

            target = state[new_row, new_col]
            if target == -1 or target == -2:
                jump_row, jump_col = new_row + dr, new_col + dc
                if self._is_valid_position(jump_row, jump_col) and state[jump_row, jump_col] == 0:
                    return True

        return False

    def get_captures_for_piece(self, state, row, col):
        """Возвращает список взятий для конкретной шашки"""
        captures = []
        piece = state[row, col]

        if piece != 1 and piece != 2:
            return captures

        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if not self._is_valid_position(new_row, new_col):
                continue

            target = state[new_row, new_col]
            if target == -1 or target == -2:
                jump_row, jump_col = new_row + dr, new_col + dc
                if self._is_valid_position(jump_row, jump_col) and state[jump_row, jump_col] == 0:
                    from_pos = row * 8 + col
                    to_pos = jump_row * 8 + jump_col
                    action = from_pos * 64 + to_pos
                    captures.append(action)

        return captures

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
            1: 'w',   # Моя шашка
            2: 'W',   # Моя дамка
            -1: 'b',  # Шашка противника
            -2: 'B'   # Дамка противника
        }

        print("  0 1 2 3 4 5 6 7")
        for row in range(self.row_count):
            print(f"{row} ", end="")
            for col in range(self.column_count):
                print(symbols[state[row, col]] + " ", end="")
            print()
        print()