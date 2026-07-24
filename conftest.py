import sys
from pathlib import Path

# allow running pytest / scripts from a clone without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
