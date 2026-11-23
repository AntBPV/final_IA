import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.assistant import OccupancyAssistant


def main():
    parser = argparse.ArgumentParser(description="Consultar estadisticas por terminal")
    parser.add_argument("pregunta", nargs="*", default=["ayuda"])
    args = parser.parse_args()
    assistant = OccupancyAssistant()
    print(assistant.ask(" ".join(args.pregunta)))


if __name__ == "__main__":
    main()

