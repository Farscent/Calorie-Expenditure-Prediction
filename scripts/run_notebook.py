"""Execute every notebook cell from a fresh kernel; replace only on success."""
import os
import asyncio
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "group-2-olc-final-project-bersih-v2.ipynb"


def main():
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("IPYTHONDIR", str(cache / "ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(cache / "jupyter"))
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    # Explicitly use this environment's interpreter, rather than a stale kernelspec.
    manager = client.create_kernel_manager()
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client.execute()
    temporary = NOTEBOOK.with_suffix(".executed.tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(NOTEBOOK)
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells successfully.")


if __name__ == "__main__":
    main()
