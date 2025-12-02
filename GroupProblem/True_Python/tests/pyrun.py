import sys
import argparse
from e2e import run_e2e


def main():
    parser = argparse.ArgumentParser(
        description="Run A* E2E pipeline for bird/branch/tree solver"
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Имя тестовой директории в data/tests/<test>"
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Корневая директория с tests/, по умолчанию: data/"
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="Сохранять state.json, позволяет перезапускать пайплайн с места остановки"
    )

    args = parser.parse_args()

    print(f"Test:\t\t{args.test}")
    print(f"Data root:\t{args.data_root}")
    print(f"Save state:\t{args.save_state}")

    try:
        run_e2e(
            test_name=args.test,
            data_root=args.data_root,
            save_state=args.save_state
        )
        print("\n[OK] Pipeline finished successfully.")
    except Exception as e:
        print("\n[ERROR] Pipeline failed:")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()