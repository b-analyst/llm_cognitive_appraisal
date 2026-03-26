"""
Capture pipeline-relevant dependencies from the current Python (e.g. your venv) into
requirements_standalone.txt with pinned versions.

Run from repo root with your venv activated:
  python pipeline/capture_requirements.py

Or with explicit interpreter:
  path/to/your/venv/Scripts/python pipeline/capture_requirements.py

Writes pipeline/requirements_captured.txt (or, if run from standalone bundle,
requirements_standalone.txt in the bundle root). Use --full to include the full
pip freeze (all packages) instead of only pipeline-relevant ones.
"""
import subprocess
import sys
from pathlib import Path

# Packages we care about (pipeline + training_utils + LLMs + data_utils + intervention_utils)
PIPELINE_DEPS = {
    "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors",
    "scikit-learn", "pandas", "numpy", "tqdm", "matplotlib", "seaborn",
    "requests", "filelock", "packaging", "PyYAML",
    "accelerate", "sympy", "networkx", "jinja2", "regex", "protobuf",
}


def main():
    full = "--full" in sys.argv
    out_name = "requirements_captured_full.txt" if full else "requirements_captured.txt"
    pipeline_dir = Path(__file__).resolve().parent.parent
    # If we're inside the standalone bundle, write to bundle root
    if (pipeline_dir.parent / "STANDALONE_README.md").exists():
        out_path = pipeline_dir.parent / "requirements_standalone.txt"
    else:
        out_path = pipeline_dir / out_name

    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print("pip freeze failed:", result.stderr or result.stdout, file=sys.stderr)
        sys.exit(1)
    lines = [ln.strip() for ln in result.stdout.strip().splitlines() if "==" in ln]
    if full:
        chosen = lines
        header = "# Full pip freeze from current environment\n"
    else:
        chosen = []
        seen = set()
        for ln in lines:
            pkg = ln.split("==")[0].split("[")[0].lower().replace("_", "-")
            if pkg in PIPELINE_DEPS or any(d in pkg for d in PIPELINE_DEPS):
                chosen.append(ln)
                seen.add(pkg)
        # Include torch first with a note if it has +cu
        header = "# Pipeline-relevant deps captured from current env. Install: pip install -r this_file\n"
        if not any(l.startswith("torch") for l in chosen):
            chosen.insert(0, "# torch not in filtered list; add manually if needed")
    out_path.write_text(header + "\n".join(chosen) + "\n", encoding="utf-8")
    print("Wrote", out_path, "(%d packages)" % len(chosen))


if __name__ == "__main__":
    main()
