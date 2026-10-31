import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

def execute_notebook(nb_path: str):
    path = Path(nb_path)
    print(f"Loading notebook: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    print("Executing notebook cells...")
    client.execute()
    print("Execution completed successfully! Writing outputs...")
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Saved executed notebook to {path.resolve()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python execute_notebook.py <notebook_path>")
        sys.exit(1)
    execute_notebook(sys.argv[1])

