# Iris Decision Tree Lab

An interactive learning app for training and exploring a decision-tree
classifier on scikit-learn's built-in Iris dataset.

## Run the web app

Install Python 3.9 or newer, then open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000> in your browser. The navigation panel has five
pages:

1. **ML concepts** - the learning pipeline and decision-tree split concepts.
2. **The Iris dataset** - measurements (features) and species (labels).
3. **Notebook | HTML** - a readable, non-executing view of the project notebook.
4. **Run Python cells** - execute and edit notebook code cell by cell in a
   persistent Python kernel. Reset the kernel to clear its variables.
5. **Results & benchmarks** - held-out accuracy, five-fold cross-validation,
   per-species scores, confusion matrix, and a tree visualization.

The app starts a Python kernel to run notebook cells. It is intended for local
use only and binds to `127.0.0.1`; do not expose it to an untrusted network.

## Run with Anaconda

In Anaconda Prompt, from the project folder:

```bat
conda create -n iris-ml python=3.11 -y
conda activate iris-ml
python -m pip install -r requirements.txt
python -m ipykernel install --prefix "%CONDA_PREFIX%" --name python3 --display-name "Python 3 (Iris ML)"
python app.py
```

Open <http://127.0.0.1:5000>. Registering the `python3` kernel in the active
environment ensures the interactive notebook uses the same Anaconda Python.

## Standalone script and notebook

```powershell
python train_iris_tree.py
```

Or open `iris_decision_tree_walkthrough.ipynb` in VS Code or Jupyter and run
its cells from top to bottom.
