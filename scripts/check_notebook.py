"""Execute the walkthrough's ordinary Python cells without a networked kernel.

This notebook has no magics, widget state, or shell cells. Standard Python
execution therefore also works on hosts that disallow Jupyter's local sockets.
Use --write to refresh outputs; otherwise the notebook is left unchanged.
"""

import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "notebooks/quickstart.ipynb",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text())
    namespace = {"__name__": "__main__"}
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        count += 1
        source = "".join(cell["source"])
        tree = ast.parse(source)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        stdout, stderr = io.StringIO(), io.StringIO()
        value = None
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(tree, f"notebook-cell-{index}", "exec"), namespace)
            if last is not None:
                value = eval(
                    compile(ast.Expression(last.value), f"notebook-cell-{index}", "eval"), namespace
                )
        outputs = []
        for name, stream in (("stdout", stdout), ("stderr", stderr)):
            if stream.getvalue():
                outputs.append({"output_type": "stream", "name": name, "text": stream.getvalue()})
        if value is not None:
            data = {"text/plain": repr(value)}
            if hasattr(value, "_repr_html_"):
                html = value._repr_html_()
                if html is not None:
                    data["text/html"] = html
            outputs.append(
                {
                    "output_type": "execute_result",
                    "execution_count": count,
                    "metadata": {},
                    "data": data,
                }
            )
        cell["execution_count"], cell["outputs"] = count, outputs
    if args.write:
        args.notebook.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Executed {count} ordinary Python notebook cells successfully.")


if __name__ == "__main__":
    main()
