import atexit
import base64
import html
import io
import json
import logging
import secrets
import threading
import time
from pathlib import Path
from queue import Empty
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import markdown
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from jupyter_client import KernelManager
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "iris_decision_tree_walkthrough.ipynb"
MAX_CELL_BYTES = 20_000
KERNEL_TIMEOUT_SECONDS = 20
MAX_OUTPUT_CHARS = 20_000
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config.update(
    MAX_CONTENT_LENGTH=64 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Strict",
)


class NotebookKernel:
    def __init__(self):
        self.manager = KernelManager(kernel_name="python3")
        self.manager.start_kernel(cwd=str(ROOT))
        self.client = self.manager.client()
        self.client.start_channels()
        try:
            self.client.wait_for_ready(timeout=15)
        except Exception:
            self.client.stop_channels()
            self.manager.shutdown_kernel(now=True)
            raise
        self.lock = threading.Lock()

    def execute(self, code):
        with self.lock:
            message_id = self.client.execute(code, allow_stdin=False)
            outputs = []
            output_length = 0
            deadline = time.monotonic() + KERNEL_TIMEOUT_SECONDS

            while time.monotonic() < deadline:
                try:
                    message = self.client.get_iopub_msg(
                        timeout=max(0.1, deadline - time.monotonic())
                    )
                except Empty:
                    break

                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue

                content = message.get("content", {})
                message_type = message.get("msg_type")
                if message_type == "stream":
                    text = content.get("text", "")
                    remaining = MAX_OUTPUT_CHARS - output_length
                    if remaining > 0:
                        outputs.append({"type": "text", "text": text[:remaining]})
                        output_length += min(len(text), remaining)
                elif message_type in ("execute_result", "display_data"):
                    data = content.get("data", {})
                    image = data.get("image/png")
                    text = data.get("text/plain")
                    if image:
                        outputs.append({"type": "image", "data": image})
                    if text and output_length < MAX_OUTPUT_CHARS:
                        remaining = MAX_OUTPUT_CHARS - output_length
                        outputs.append({"type": "text", "text": text[:remaining]})
                        output_length += min(len(text), remaining)
                elif message_type == "error":
                    traceback = "\n".join(content.get("traceback", []))
                    text = content.get("evalue", "Kernel execution failed.")
                    if traceback:
                        text = f"{text}\n{traceback}"
                    remaining = MAX_OUTPUT_CHARS - output_length
                    if remaining > 0:
                        outputs.append(
                            {"type": "error", "text": text[:remaining]}
                        )
                elif message_type == "status" and content.get("execution_state") == "idle":
                    return outputs, False

            self.manager.interrupt_kernel()
            return outputs + [
                {
                    "type": "error",
                    "text": f"Cell exceeded the {KERNEL_TIMEOUT_SECONDS}-second limit; the kernel was interrupted.",
                }
            ], True

    def shutdown(self):
        self.client.stop_channels()
        self.manager.shutdown_kernel(now=True)


kernels = {}
kernels_lock = threading.Lock()
chart_lock = threading.Lock()


def shutdown_kernels():
    with kernels_lock:
        active_kernels = list(kernels.values())
        kernels.clear()
    for kernel in active_kernels:
        try:
            kernel.shutdown()
        except Exception:
            logger.exception("Could not shut down a notebook kernel cleanly.")


atexit.register(shutdown_kernels)


def get_notebook_cells():
    with NOTEBOOK_PATH.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)
    cells = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            cells.append(
                {
                    "kind": "markdown",
                    "html": markdown.markdown(
                        html.escape(source), extensions=["fenced_code"]
                    ),
                }
            )
        elif cell["cell_type"] == "code":
            cells.append({"kind": "code", "source": source})
    return cells


def get_kernel():
    kernel_id = session.get("kernel_id")
    if not kernel_id:
        kernel_id = secrets.token_urlsafe(24)
        session["kernel_id"] = kernel_id
    with kernels_lock:
        kernel = kernels.get(kernel_id)
        if kernel is None:
            kernel = NotebookKernel()
            kernels[kernel_id] = kernel
    return kernel_id, kernel


def make_chart_png(draw_chart):
    with chart_lock:
        buffer = io.BytesIO()
        figure = draw_chart()
        figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def get_benchmark_results():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=iris.target_names,
        output_dict=True,
        zero_division=0,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        DecisionTreeClassifier(max_depth=3, random_state=42),
        iris.data,
        iris.target,
        cv=folds,
        scoring="accuracy",
    )

    def draw_tree():
        figure, axis = plt.subplots(figsize=(16, 9))
        plot_tree(
            model,
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            filled=True,
            rounded=True,
            ax=axis,
        )
        axis.set_title("Decision tree trained on the 80% training split")
        figure.tight_layout()
        return figure

    def draw_confusion_matrix():
        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set(
            xticks=range(len(iris.target_names)),
            yticks=range(len(iris.target_names)),
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            xlabel="Predicted species",
            ylabel="Actual species",
            title="Confusion matrix (held-out test set)",
        )
        plt.setp(axis.get_xticklabels(), rotation=25, ha="right")
        threshold = matrix.max() / 2
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        figure.tight_layout()
        return figure

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "test_samples": len(y_test),
        "report": report,
        "matrix": matrix.tolist(),
        "class_names": iris.target_names.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "tree_image": make_chart_png(draw_tree),
        "matrix_image": make_chart_png(draw_confusion_matrix),
    }


@app.get("/")
def home():
    return redirect(url_for("concepts"))


@app.get("/concepts")
def concepts():
    return render_template("concepts.html", active_page="concepts")


@app.get("/dataset")
def dataset():
    iris = load_iris()
    feature_rows = [
        ("Sepal length", "cm", iris.feature_names[0], "Length of the outer flower part"),
        ("Sepal width", "cm", iris.feature_names[1], "Width of the outer flower part"),
        ("Petal length", "cm", iris.feature_names[2], "Length of the inner flower part"),
        ("Petal width", "cm", iris.feature_names[3], "Width of the inner flower part"),
    ]
    label_rows = [
        (index, name, int((iris.target == index).sum()))
        for index, name in enumerate(iris.target_names)
    ]
    return render_template(
        "dataset.html",
        active_page="dataset",
        feature_rows=feature_rows,
        label_rows=label_rows,
    )


@app.get("/notebook")
def notebook():
    return render_template(
        "notebook.html", active_page="notebook", cells=get_notebook_cells()
    )


@app.get("/kernel")
def kernel_page():
    code_cells = []
    cell_description = ""
    for cell in get_notebook_cells():
        if cell["kind"] == "markdown":
            cell_description = cell["html"]
        else:
            code_cells.append(
                {
                    "source": cell["source"],
                    "description": cell_description,
                    "rows": max(4, min(cell["source"].count("\n") + 1, 16)),
                }
            )
    return render_template(
        "kernel.html", active_page="kernel", code_cells=code_cells
    )


@app.post("/api/kernel/execute")
def execute_cell():
    origin = request.headers.get("Origin")
    if origin and urlparse(origin).netloc != request.host:
        return jsonify(error="Cross-origin notebook requests are not allowed."), 403
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("code"), str):
        return jsonify(error="Request must contain a string 'code' field."), 400
    code = body["code"]
    if not code.strip():
        return jsonify(error="Enter some Python code before running this cell."), 400
    if len(code.encode("utf-8")) > MAX_CELL_BYTES:
        return jsonify(error=f"Cell code must be at most {MAX_CELL_BYTES} bytes."), 413
    try:
        _, kernel = get_kernel()
        outputs, timed_out = kernel.execute(code)
    except Exception as error:
        logger.exception("Notebook cell execution failed.")
        return (
            jsonify(error=f"The Python kernel could not run this cell: {error}"),
            503,
        )
    return jsonify(outputs=outputs, timed_out=timed_out)


@app.post("/api/kernel/reset")
def reset_kernel():
    origin = request.headers.get("Origin")
    if origin and urlparse(origin).netloc != request.host:
        return jsonify(error="Cross-origin notebook requests are not allowed."), 403
    kernel_id = session.get("kernel_id")
    if kernel_id:
        with kernels_lock:
            kernel = kernels.pop(kernel_id, None)
        if kernel is not None:
            try:
                kernel.shutdown()
            except Exception as error:
                logger.exception("Could not shut down the notebook kernel.")
                return (
                    jsonify(error=f"The kernel could not be shut down cleanly: {error}"),
                    500,
                )
        session.pop("kernel_id", None)
    return jsonify(message="Kernel reset. Run the cells again from the top.")


@app.get("/results")
def results():
    return render_template(
        "results.html", active_page="results", results=get_benchmark_results()
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
