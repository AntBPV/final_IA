import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.device import device_summary


def main():
    print(json.dumps(device_summary("auto"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
