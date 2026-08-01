import sys

from games.carracing import play_car_racing
from games.doom import play_doom


def main():
    if len(sys.argv) > 1:
        game = sys.argv[1].lower()
    else:
        print("Выберите игру:")
        print("  1 — CarRacing")
        print("  2 — ViZDoom")
        game = input("Введите 1 или 2: ").strip()

    if game in ("1", "car", "carracing", "car_racing"):
        play_car_racing()
    elif game in ("2", "doom", "vizdoom", "doom-viz"):
        play_doom()
    else:
        print(f"Неизвестный выбор: '{game}'. Запускаю CarRacing по умолчанию.")
        play_car_racing()


if __name__ == "__main__":
    main()
