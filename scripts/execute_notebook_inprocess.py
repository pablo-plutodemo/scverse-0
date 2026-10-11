"""Execute a notebook in the current Python process (no Jupyter kernel / sockets needed).

Fallback for sandboxed environments where `jupyter nbconvert --execute` cannot bind ports.
Captures stdout, the repr / HTML of the last expression, and matplotlib figures for every code cell
and writes them back into the notebook as nbformat outputs.

Usage: python scripts/execute_notebook_inprocess.py notebooks/wat_snrna_depot_analysis.ipynb
"""
import ast
import base64
import io
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_output


def _rich(obj, count):
    if obj is None:
        return None
    data = {"text/plain": repr(obj)}
    if hasattr(obj, "_repr_html_"):
        try:
            html = obj._repr_html_()
            if html:
                data["text/html"] = html
        except Exception:
            pass
    return new_output("execute_result", data=data, execution_count=count)


def _figures():
    outs = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        outs.append(new_output("display_data", data={"image/png": base64.b64encode(buf.getvalue()).decode()}))
    plt.close("all")
    return outs


def run(nb_path: Path):
    nb = nbformat.read(nb_path, as_version=4)
    os.chdir(nb_path.parent)
    ns = {"__name__": "__main__"}
    plt.show = lambda *a, **k: None  # figures are harvested after each cell instead
    count = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        cell.outputs = []
        cell.execution_count = count
        out, err = io.StringIO(), io.StringIO()
        t0 = time.time()
        result = None
        try:
            tree = ast.parse(cell.source)
            last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            body = tree.body[:-1] if last is not None else tree.body
            with redirect_stdout(out), redirect_stderr(err):
                if body:
                    exec(compile(ast.Module(body=body, type_ignores=[]), str(nb_path), "exec"), ns)
                if last is not None:
                    result = eval(compile(ast.Expression(last.value), str(nb_path), "eval"), ns)
        except Exception:
            tb = traceback.format_exc()
            cell.outputs.append(new_output("error", ename="Exception", evalue="", traceback=tb.splitlines()))
            print(f"cell {count} FAILED\n{tb}", file=sys.stderr)
        if out.getvalue():
            cell.outputs.append(new_output("stream", name="stdout", text=out.getvalue()))
        if err.getvalue().strip():
            cell.outputs.append(new_output("stream", name="stderr", text=err.getvalue()))
        cell.outputs.extend(_figures())
        r = _rich(result, count)
        if r is not None:
            cell.outputs.append(r)
        print(f"cell {count}: {time.time() - t0:.1f}s", flush=True)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.write(nb, nb_path)


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
