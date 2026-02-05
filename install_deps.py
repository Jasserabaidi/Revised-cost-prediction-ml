"""
Installe les dépendances du projet dans l'interpréteur Python actuel.
Exécuter ce fichier une fois (Run Python File) avec le même interpréteur
que le noyau Jupyter du notebook, puis lancer ml.ipynb.
"""
import subprocess
import sys
from pathlib import Path

def main():
    req_file = Path(__file__).resolve().parent / "requirements.txt"
    print(f"Python: {sys.executable}")
    print(f"Installation depuis {req_file}...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout)
        sys.exit(1)
    print("OK. Vous pouvez maintenant exécuter ml.ipynb (Run All).")

if __name__ == "__main__":
    main()
