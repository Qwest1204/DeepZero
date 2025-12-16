import gradio as gr
import numpy as np
import torch
import time
import copy


# Импортируйте ваши классы (раскомментируйте и укажите правильные пути)
# from checkers import Checkers
# from model import ResNet
# from mcts import MCTS

# ============= ЗАГЛУШКА ДЛЯ ТЕСТИРОВАНИЯ =============
# Удалите этот блок и раскомментируйте импорты выше

class Checkers:
    """Заглушка класса Checkers для демонстрации"""

    def __init__(self):
        self.action_size = 32 * 4 * 2  # Примерный размер

    def get_initial_state(self):
        # 8x8 доска: 1 = белые, -1 = чёрные, 2/-2 = дамки
        board = np.zeros((8, 8), dtype=np.int8)
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = -1  # чёрные
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = 1  # белые
        return board

    def get_valid_moves(self, state):
        """Возвращает список допустимых ходов"""
        moves = np.zeros(self.action_size)
        valid_actions = self._get_valid_actions(state, 1)
        for action in valid_actions:
            if action < self.action_size:
                moves[action] = 1
        return moves

    def _get_valid_actions(self, state, player):
        """Получить все допустимые действия для игрока"""
        actions = []
        captures = []

        for row in range(8):
            for col in range(8):
                piece = state[row][col]
                if (player == 1 and piece in [1, 2]) or (player == -1 and piece in [-1, -2]):
                    piece_actions, piece_captures = self._get_piece_moves(state, row, col, piece)
                    actions.extend(piece_actions)
                    captures.extend(piece_captures)

        # Если есть взятия, только они допустимы
        if captures:
            return captures
        return actions

    def _get_piece_moves(self, state, row, col, piece):
        """Получить ходы для конкретной шашки"""
        actions = []
        captures = []
        is_king = abs(piece) == 2

        if is_king:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        elif piece > 0:  # белые идут вверх
            directions = [(-1, -1), (-1, 1)]
        else:  # чёрные идут вниз
            directions = [(1, -1), (1, 1)]

        for dr, dc in directions:
            # Обычный ход
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if state[new_row][new_col] == 0:
                    action = self._encode_action(row, col, new_row, new_col)
                    actions.append(action)

            # Взятие
            jump_row, jump_col = row + 2 * dr, col + 2 * dc
            if 0 <= jump_row < 8 and 0 <= jump_col < 8:
                mid_row, mid_col = row + dr, col + dc
                mid_piece = state[mid_row][mid_col]
                if mid_piece != 0 and np.sign(mid_piece) != np.sign(piece):
                    if state[jump_row][jump_col] == 0:
                        action = self._encode_action(row, col, jump_row, jump_col)
                        captures.append(action)

        return actions, captures

    def _encode_action(self, from_row, from_col, to_row, to_col):
        """Кодирует ход в число"""
        from_idx = from_row * 8 + from_col
        to_idx = to_row * 8 + to_col
        return from_idx * 64 + to_idx

    def _decode_action(self, action):
        """Декодирует число в ход"""
        from_idx = action // 64
        to_idx = action % 64
        from_row, from_col = from_idx // 8, from_idx % 8
        to_row, to_col = to_idx // 8, to_idx % 8
        return from_row, from_col, to_row, to_col

    def get_next_state(self, state, action, player):
        """Применяет ход к состоянию"""
        new_state = state.copy()
        from_row, from_col, to_row, to_col = self._decode_action(action)

        piece = new_state[from_row][from_col]
        new_state[from_row][from_col] = 0
        new_state[to_row][to_col] = piece

        # Взятие
        if abs(to_row - from_row) == 2:
            mid_row = (from_row + to_row) // 2
            mid_col = (from_col + to_col) // 2
            new_state[mid_row][mid_col] = 0

        # Превращение в дамку
        if piece == 1 and to_row == 0:
            new_state[to_row][to_col] = 2
        elif piece == -1 and to_row == 7:
            new_state[to_row][to_col] = -2

        return new_state

    def get_value_and_terminated(self, state, action):
        """Проверяет окончание игры"""
        white_pieces = np.sum((state == 1) | (state == 2))
        black_pieces = np.sum((state == -1) | (state == -2))

        if white_pieces == 0:
            return -1, True
        if black_pieces == 0:
            return 1, True

        # Проверяем есть ли ходы
        white_moves = len(self._get_valid_actions(state, 1))
        black_moves = len(self._get_valid_actions(state, -1))

        if white_moves == 0:
            return -1, True
        if black_moves == 0:
            return 1, True

        return 0, False

    def change_perspective(self, state, player):
        """Меняет перспективу доски"""
        if player == -1:
            return np.flip(state) * -1
        return state.copy()

    def flip_action(self, action):
        """Переворачивает действие для чёрного игрока"""
        from_row, from_col, to_row, to_col = self._decode_action(action)
        from_row, from_col = 7 - from_row, 7 - from_col
        to_row, to_col = 7 - to_row, 7 - to_col
        return self._encode_action(from_row, from_col, to_row, to_col)

    def get_opponent(self, player):
        return -player


class SimpleMCTS:
    """Простой MCTS для демонстрации (замените на ваш MCTS)"""

    def __init__(self, game, args, model=None):
        self.game = game
        self.args = args
        self.model = model

    def search(self, state):
        """Возвращает вероятности ходов и оценку"""
        valid_moves = self.game.get_valid_moves(state)
        probs = valid_moves / (valid_moves.sum() + 1e-8)

        # Простая оценка
        white = np.sum((state == 1)) + 2 * np.sum((state == 2))
        black = np.sum((state == -1)) + 2 * np.sum((state == -2))
        value = (white - black) / (white + black + 1e-8)

        return probs, value


# ============= КОНЕЦ ЗАГЛУШКИ =============


class CheckersGame:
    """Класс для управления игрой в Gradio"""

    def __init__(self):
        self.game = Checkers()
        self.device = torch.device("cpu")

        self.args = {
            'C': 2,
            'num_searches': 400,
            'num_iterations': 10,
            'num_parallel_games': 200,
            'batch_size': 128,
            'num_selfPlay_iterations': 1000,
            'num_epochs': 10,
            'temperature': 1.0,
            'dirichlet_epsilon': 0.0,
            'dirichlet_alpha': 0.3
        }

        # Инициализация моделей (раскомментируйте для реальных моделей)
        # self.model_white = ResNet(self.game, 24, 256, device=self.device)
        # self.model_white.load_state_dict(torch.load("weights/model_3_Checkers.pt", map_location=self.device))
        # self.model_white.eval()

        # self.model_black = ResNet(self.game, 24, 256, device=self.device)
        # self.model_black.load_state_dict(torch.load("weights/model_3_Checkers.pt", map_location=self.device))
        # self.model_black.eval()

        # Используем простой MCTS для демонстрации
        self.mcts_white = SimpleMCTS(self.game, self.args, None)
        self.mcts_black = SimpleMCTS(self.game, self.args, None)

        # Для реальных моделей:
        # self.mcts_white = MCTS(self.game, self.args, self.model_white)
        # self.mcts_black = MCTS(self.game, self.args, self.model_black)

        self.reset_game()

    def reset_game(self):
        """Сброс игры"""
        self.state = self.game.get_initial_state()
        self.player = 1
        self.game_over = False
        self.winner = None
        self.selected_cell = None
        self.valid_moves_for_selected = []
        self.move_history = []
        self.message = "Ход белых. Выберите шашку."

    def state_to_html(self):
        """Преобразует состояние доски в HTML"""
        pieces = {
            0: '',
            1: '⚪',  # белая шашка
            -1: '⚫',  # чёрная шашка
            2: '👑',  # белая дамка
            -2: '🖤'  # чёрная дамка (корона)
        }

        html = '''
        <style>
            .board { 
                display: grid; 
                grid-template-columns: repeat(8, 60px); 
                gap: 0; 
                border: 4px solid #5d4037;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            .cell { 
                width: 60px; 
                height: 60px; 
                display: flex; 
                align-items: center; 
                justify-content: center; 
                font-size: 36px;
                cursor: pointer;
                transition: all 0.2s;
            }
            .cell:hover { transform: scale(1.05); }
            .light { background: linear-gradient(145deg, #f5deb3, #e8d4a8); }
            .dark { background: linear-gradient(145deg, #5d4e37, #4a3f2d); }
            .selected { box-shadow: inset 0 0 20px 5px rgba(255, 215, 0, 0.8); }
            .valid-move { box-shadow: inset 0 0 15px 3px rgba(0, 255, 0, 0.6); }
            .capture-move { box-shadow: inset 0 0 15px 3px rgba(255, 0, 0, 0.6); }
            .piece-white { filter: drop-shadow(2px 2px 2px rgba(0,0,0,0.3)); }
            .piece-black { filter: drop-shadow(2px 2px 2px rgba(0,0,0,0.5)); }
        </style>
        <div class="board">
        '''

        for row in range(8):
            for col in range(8):
                cell_class = 'light' if (row + col) % 2 == 0 else 'dark'
                piece = self.state[row][col]
                piece_class = 'piece-white' if piece > 0 else 'piece-black' if piece < 0 else ''

                # Подсветка выбранной клетки
                if self.selected_cell == (row, col):
                    cell_class += ' selected'

                # Подсветка допустимых ходов
                for move, is_capture in self.valid_moves_for_selected:
                    _, _, to_row, to_col = self.game._decode_action(move)
                    if (to_row, to_col) == (row, col):
                        if is_capture:
                            cell_class += ' capture-move'
                        else:
                            cell_class += ' valid-move'

                symbol = pieces.get(piece, '')
                if piece == -2:  # чёрная дамка - показываем корону
                    symbol = '👑'
                    piece_class = 'piece-black'

                html += f'<div class="cell {cell_class} {piece_class}">{symbol}</div>'

        html += '</div>'
        return html

    def get_valid_moves_for_piece(self, row, col):
        """Получить допустимые ходы для выбранной шашки"""
        piece = self.state[row][col]
        if piece == 0:
            return []

        if (self.player == 1 and piece < 0) or (self.player == -1 and piece > 0):
            return []  # Не та шашка

        moves = []
        captures = []

        # Получаем все ходы этой шашки
        actions, capture_actions = self.game._get_piece_moves(self.state, row, col, piece)

        # Проверяем, есть ли взятия у любой шашки
        all_valid = self.game._get_valid_actions(self.state, self.player)
        has_any_captures = any(
            abs(self.game._decode_action(a)[2] - self.game._decode_action(a)[0]) == 2
            for a in all_valid
        )

        if has_any_captures:
            # Только взятия допустимы
            for action in capture_actions:
                captures.append((action, True))
            return captures
        else:
            for action in actions:
                moves.append((action, False))
            return moves

    def handle_click(self, row, col, game_mode):
        """Обработка клика по доске"""
        if self.game_over:
            return self.state_to_html(), self.message, self.get_history_html()

        # Проверяем, чей ход в зависимости от режима
        if game_mode == "AI vs AI":
            return self.state_to_html(), "В режиме AI vs AI кликать нельзя. Нажмите 'Ход AI'.", self.get_history_html()

        if game_mode == "Человек vs AI" and self.player == -1:
            return self.state_to_html(), "Сейчас ход AI. Нажмите 'Ход AI'.", self.get_history_html()

        if game_mode == "AI vs Человек" and self.player == 1:
            return self.state_to_html(), "Сейчас ход AI. Нажмите 'Ход AI'.", self.get_history_html()

        piece = self.state[row][col]

        # Если кликнули на допустимый ход
        for move, is_capture in self.valid_moves_for_selected:
            _, _, to_row, to_col = self.game._decode_action(move)
            if (to_row, to_col) == (row, col):
                self.make_move(move)
                return self.state_to_html(), self.message, self.get_history_html()

        # Если кликнули на свою шашку - выбираем её
        if (self.player == 1 and piece > 0) or (self.player == -1 and piece < 0):
            self.selected_cell = (row, col)
            self.valid_moves_for_selected = self.get_valid_moves_for_piece(row, col)
            if self.valid_moves_for_selected:
                self.message = f"Выбрана шашка ({row}, {col}). Выберите куда ходить."
            else:
                self.message = "У этой шашки нет допустимых ходов."
        else:
            self.selected_cell = None
            self.valid_moves_for_selected = []
            self.message = f"Ход {'белых' if self.player == 1 else 'чёрных'}. Выберите свою шашку."

        return self.state_to_html(), self.message, self.get_history_html()

    def make_move(self, action):
        """Выполнить ход"""
        from_row, from_col, to_row, to_col = self.game._decode_action(action)

        # Записываем ход в историю
        cols = 'ABCDEFGH'
        is_capture = abs(to_row - from_row) == 2
        move_str = f"{'⚪' if self.player == 1 else '⚫'} {cols[from_col]}{8 - from_row} {'x' if is_capture else '→'} {cols[to_col]}{8 - to_row}"
        self.move_history.append(move_str)

        # Применяем ход
        self.state = self.game.get_next_state(self.state, action, self.player)

        # Проверяем окончание
        value, is_terminate = self.game.get_value_and_terminated(self.state, action)
        if is_terminate:
            self.game_over = True
            if value == 1:
                self.winner = "Белые"
            else:
                self.winner = "Чёрные"
            self.message = f"🏆 Игра окончена! Победили {self.winner}!"
        else:
            # Проверяем продолжение взятия
            if is_capture:
                more_captures = self.get_valid_moves_for_piece(to_row, to_col)
                more_captures = [(m, c) for m, c in more_captures if c]  # только взятия
                if more_captures:
                    self.selected_cell = (to_row, to_col)
                    self.valid_moves_for_selected = more_captures
                    self.message = f"Продолжайте взятие с ({to_row}, {to_col})!"
                    return

            # Переход хода
            self.player = self.game.get_opponent(self.player)
            self.selected_cell = None
            self.valid_moves_for_selected = []
            self.message = f"Ход {'белых' if self.player == 1 else 'чёрных'}. Выберите шашку."

    def ai_move(self, game_mode):
        """Ход AI"""
        if self.game_over:
            return self.state_to_html(), self.message, self.get_history_html()

        # Проверяем, должен ли AI ходить
        should_ai_move = False
        if game_mode == "AI vs AI":
            should_ai_move = True
        elif game_mode == "Человек vs AI" and self.player == -1:
            should_ai_move = True
        elif game_mode == "AI vs Человек" and self.player == 1:
            should_ai_move = True

        if not should_ai_move:
            return self.state_to_html(), "Сейчас ход человека.", self.get_history_html()

        # Выбираем MCTS
        mcts = self.mcts_white if self.player == 1 else self.mcts_black

        # Получаем состояние с перспективы текущего игрока
        neutral_state = self.game.change_perspective(self.state, self.player)

        # Поиск MCTS
        mcts_probs, net_win_value = mcts.search(neutral_state)

        # Маскируем недопустимые ходы
        valid_moves = self.game.get_valid_moves(neutral_state)
        mcts_probs = mcts_probs * valid_moves

        if mcts_probs.sum() == 0:
            valid_indices = np.where(valid_moves == 1)[0]
            if len(valid_indices) == 0:
                self.game_over = True
                self.winner = "Белые" if self.player == -1 else "Чёрные"
                self.message = f"🏆 Нет ходов! Победили {self.winner}!"
                return self.state_to_html(), self.message, self.get_history_html()
            action_neutral = np.random.choice(valid_indices)
        else:
            action_neutral = np.argmax(mcts_probs)

        # Переводим ход для чёрных
        if self.player == -1:
            action = self.game.flip_action(action_neutral)
        else:
            action = action_neutral

        # Выполняем ход
        self.selected_cell = None
        self.valid_moves_for_selected = []
        self.make_move(action)

        # Добавляем оценку
        if not self.game_over:
            self.message += f" (Оценка AI: {net_win_value:.2f})"

        return self.state_to_html(), self.message, self.get_history_html()

    def get_history_html(self):
        """История ходов в HTML"""
        if not self.move_history:
            return "<p style='color: #888;'>Игра началась...</p>"

        html = "<div style='max-height: 300px; overflow-y: auto;'>"
        for i, move in enumerate(self.move_history, 1):
            html += f"<p>{i}. {move}</p>"
        html += "</div>"
        return html

    def get_stats(self):
        """Статистика игры"""
        white = np.sum(self.state == 1)
        white_kings = np.sum(self.state == 2)
        black = np.sum(self.state == -1)
        black_kings = np.sum(self.state == -2)

        return f"""
        ⚪ Белые: {white} шашек, {white_kings} дамок
        ⚫ Чёрные: {black} шашек, {black_kings} дамок
        📊 Всего ходов: {len(self.move_history)}
        """


# Создаём экземпляр игры
game_instance = CheckersGame()


def create_click_handler(row, col):
    """Создаёт обработчик клика для конкретной клетки"""

    def handler(game_mode):
        return game_instance.handle_click(row, col, game_mode)

    return handler


def reset_game():
    """Сброс игры"""
    game_instance.reset_game()
    return (
        game_instance.state_to_html(),
        game_instance.message,
        game_instance.get_history_html(),
        game_instance.get_stats()
    )


def ai_move(game_mode):
    """Ход AI"""
    board, msg, history = game_instance.ai_move(game_mode)
    return board, msg, history, game_instance.get_stats()


def make_click(row, col, game_mode):
    """Универсальный обработчик клика"""
    board, msg, history = game_instance.handle_click(row, col, game_mode)
    return board, msg, history, game_instance.get_stats()


# Создаём Gradio интерфейс
with gr.Blocks(title="Шашки с MCTS AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎮 Шашки с MCTS AI
    ### Играйте против нейросетевого алгоритма!
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # Доска
            board_html = gr.HTML(value=game_instance.state_to_html(), label="Доска")

            # Кнопки для кликов по доске (8x8 сетка)
            gr.Markdown("### 🖱️ Кликните по клетке:")

            with gr.Group():
                click_buttons = []
                for row in range(8):
                    with gr.Row():
                        for col in range(8):
                            btn_label = f"{row},{col}"
                            cell_color = "secondary" if (row + col) % 2 == 0 else "primary"
                            btn = gr.Button(
                                btn_label,
                                size="sm",
                                variant=cell_color,
                                min_width=40
                            )
                            click_buttons.append((btn, row, col))

        with gr.Column(scale=1):
            # Панель управления
            gr.Markdown("### ⚙️ Управление")

            game_mode = gr.Radio(
                choices=["Человек vs Человек", "Человек vs AI", "AI vs Человек", "AI vs AI"],
                value="Человек vs AI",
                label="Режим игры"
            )

            with gr.Row():
                reset_btn = gr.Button("🔄 Новая игра", variant="primary")
                ai_btn = gr.Button("🤖 Ход AI", variant="secondary")

            # Статус
            status_text = gr.Textbox(
                value=game_instance.message,
                label="📢 Статус",
                interactive=False
            )

            # Статистика
            stats_text = gr.Textbox(
                value=game_instance.get_stats(),
                label="📊 Статистика",
                interactive=False,
                lines=4
            )

            # История ходов
            history_html = gr.HTML(
                value=game_instance.get_history_html(),
                label="📜 История ходов"
            )

    # Подключаем обработчики
    outputs = [board_html, status_text, history_html, stats_text]

    reset_btn.click(reset_game, outputs=outputs)
    ai_btn.click(ai_move, inputs=[game_mode], outputs=outputs)

    # Подключаем клики по кнопкам доски
    for btn, row, col in click_buttons:
        btn.click(
            lambda r=row, c=col, gm=game_mode: make_click(r, c, gm.value if hasattr(gm, 'value') else gm),
            inputs=[game_mode],
            outputs=outputs
        )

    gr.Markdown("""
    ---
    ### 📖 Правила:
    - ⚪ Белые ходят первыми (вверх)
    - ⚫ Чёрные ходят вторыми (вниз)
    - 👑 Дамка может ходить в любом направлении
    - Взятие обязательно!
    - При возможности нескольких взятий - нужно бить все

    ### 🎯 Как играть:
    1. Выберите режим игры
    2. Кликните на свою шашку (она подсветится)
    3. Кликните на подсвеченную клетку для хода
    4. Для хода AI нажмите "Ход AI"
    """)

if __name__ == "__main__":
    demo.launch(share=True)