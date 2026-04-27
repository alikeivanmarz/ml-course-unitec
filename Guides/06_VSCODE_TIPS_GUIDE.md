# VSCode for Python and Jupyter Workflows

Visual Studio Code is the default editor for many Python and machine learning workflows. This guide covers the extensions, settings, and keyboard shortcuts that matter for productive Python and Jupyter work, with a focus on the configurations that pay back the time spent setting them up.

**Table of Contents**

1. [Required and Recommended Extensions](#1-required-and-recommended-extensions)
2. [Kernel Management](#2-kernel-management)
3. [Variable Explorer and Data Viewer](#3-variable-explorer-and-data-viewer)
4. [Debugging Notebook Cells](#4-debugging-notebook-cells)
5. [Integrated Terminal Workflows](#5-integrated-terminal-workflows)
6. [Useful `settings.json` Snippets](#6-useful-settingsjson-snippets)
7. [Keyboard Shortcuts](#7-keyboard-shortcuts)
8. [Resources](#8-resources)

---

## 1. Required and Recommended Extensions

| Extension | Publisher | Purpose |
|-----------|-----------|---------|
| Python | Microsoft | Core Python support; debugging; environment selection |
| Pylance | Microsoft | Fast type checking, autocomplete, import resolution |
| Jupyter | Microsoft | Notebook editing, kernel management, variable explorer |
| Jupyter Keymap | Microsoft | Classic Jupyter keyboard shortcuts |
| Python Debugger | Microsoft | Modern debugger for breakpoints, watch, conditional stops |
| Ruff | Astral Software | Fast linter and formatter; replaces flake8 + isort + many plugins |
| GitLens | GitKraken | Inline blame, history navigation, branch comparison |
| Error Lens | Alexander | Inline display of warnings and errors next to the offending line |
| Even Better TOML | tamasfe | Syntax highlighting and validation for `pyproject.toml` |
| YAML | Red Hat | Schema validation for YAML configs |
| Rainbow CSV | mechatroner | Column-aware CSV viewing |
| Markdown All in One | Yu Zhang | Live preview, table-of-contents generation |

### 1.1 Optional Additions

| Extension | When useful |
|-----------|-------------|
| Docker | Building and running containers from VSCode |
| Remote — SSH | Editing files on a remote machine as if local |
| Dev Containers | Project-specific reproducible dev environments |
| Polyglot Notebooks | Multi-language notebooks (Python + SQL + R in one notebook) |
| Live Share | Real-time collaborative editing |

A minimal effective set is just the first five (Python, Pylance, Jupyter, Jupyter Keymap, Ruff). Adding more without need slows VSCode startup.

---

## 2. Kernel Management

### 2.1 Selecting a Kernel

In any open notebook, the kernel selector appears in the top-right of the editor. The list includes:

- All Python interpreters discovered (system, conda envs, virtualenvs, pyenv installs)
- Any registered Jupyter kernels (via `ipykernel install`)
- Remote kernels (when connected via Remote — SSH or Dev Containers)

### 2.2 Registering a Conda Environment as a Jupyter Kernel

```bash
conda activate myenv
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

After registration, the kernel appears in the notebook kernel picker by its display name and persists across notebooks.

### 2.3 Inspecting the Active Kernel

```python
import sys
sys.executable          # path to Python interpreter
sys.version             # Python version
```

A `ModuleNotFoundError` for a package known to be installed almost always means the wrong kernel is selected. Check `sys.executable` first.

### 2.4 Kernel Restart vs Cell Re-Run

| Symptom | Action |
|---------|--------|
| Stale variable values after editing a function | Restart kernel; functions are not redefined automatically without re-running their cell |
| Long-running cell appears stuck | Interrupt kernel (do not restart unless interrupt fails) |
| Out-of-memory error | Restart kernel to release tensors and large objects |
| Imports fail after `pip install` from another terminal | Restart kernel; sys.path is read at startup |

The `%load_ext autoreload` and `%autoreload 2` magics re-import modules automatically when their source files change — useful when iterating on code in `src/`.

---

## 3. Variable Explorer and Data Viewer

The Jupyter extension provides two inspection tools accessible from the notebook toolbar.

### 3.1 Variable Explorer

Lists all variables in the active kernel with their types and short string representations. Filtering by name and sorting by size are useful for diagnosing memory issues.

### 3.2 Data Viewer

For pandas DataFrames, NumPy arrays, and tensors, the data viewer opens a spreadsheet-like view with:

- Column-wise filtering and sorting
- Statistical summary per column
- Column type display

Particularly effective for inspecting intermediate transformations during preprocessing without generating ad-hoc print cells.

---

## 4. Debugging Notebook Cells

Setting a breakpoint in a notebook cell and clicking "Debug Cell" launches the interactive debugger. Available actions:

| Action | Default keybinding | Purpose |
|--------|--------------------|---------|
| Step Over | F10 | Execute current line; do not enter functions |
| Step Into | F11 | Enter the function being called |
| Step Out | Shift+F11 | Run to the end of current function and return |
| Continue | F5 | Resume execution until next breakpoint |
| Watch | (sidebar) | Evaluate expressions on every step |

### 4.1 Conditional Breakpoints

Right-click in the gutter and choose "Add Conditional Breakpoint". Useful for inspecting only specific iterations:

```python
i == 1000        # break only when i equals 1000
isinstance(x, np.ndarray) and x.shape != (224, 224, 3)
```

### 4.2 Post-Mortem Debugging

After an exception, the debug console can be activated via the "Debug" pane to inspect the stack frame at the point of failure — variables remain accessible without re-running.

For non-notebook scripts, `breakpoint()` in source code triggers the debugger when the script is launched in debug mode.

---

## 5. Integrated Terminal Workflows

Open a terminal with `Ctrl+\`` (`Cmd+\`` on macOS). The terminal automatically activates the project's Python environment if `python.terminal.activateEnvironment` is enabled.

### 5.1 Common Tasks

| Task | Command |
|------|---------|
| Install a package into the active env | `pip install <pkg>` |
| List installed packages | `pip list` or `pip freeze` |
| Run all tests | `pytest` |
| Run a single test | `pytest tests/test_x.py::test_y -xvs` |
| Format the codebase | `ruff format .` |
| Lint the codebase | `ruff check .` |
| Start a Jupyter server (for non-VSCode access) | `jupyter lab` |

### 5.2 Multiple Terminals

Split terminals (`Ctrl+Shift+5`) provide side-by-side panes — one for running a long process (training, server), one for ad-hoc commands. Each terminal can run a different shell or remote connection.

---

## 6. Useful `settings.json` Snippets

Workspace settings live in `.vscode/settings.json` and override user settings for the project.

### 6.1 Python and Jupyter Defaults

```json
{
  "python.terminal.activateEnvironment": true,
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.autoImportCompletions": true,
  "jupyter.askForKernelRestart": false,
  "jupyter.interactiveWindow.creationMode": "perFile",
  "notebook.output.scrolling": true,
  "notebook.output.textLineLimit": 50
}
```

### 6.2 Format on Save with Ruff

```json
{
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.organizeImports.ruff": "explicit",
      "source.fixAll.ruff": "explicit"
    }
  }
}
```

### 6.3 Excluding Noise from File Search

```json
{
  "search.exclude": {
    "**/node_modules": true,
    "**/__pycache__": true,
    "**/.ipynb_checkpoints": true,
    "**/data": true,
    "**/models": true,
    "**/.venv": true
  }
}
```

Excluding `data/` and `models/` from search prevents accidental matches in CSV contents and binary blobs that overwhelm the results pane.

### 6.4 Ruler at 100 Characters

```json
{
  "editor.rulers": [100],
  "editor.wordWrap": "off"
}
```

A visible ruler discourages drift past the chosen line length without enforcing reformatting on every keystroke.

---

## 7. Keyboard Shortcuts

### 7.1 Editor

| Shortcut (macOS / Win-Linux) | Action |
|-------------------------------|--------|
| `Cmd+P` / `Ctrl+P` | Quick file open |
| `Cmd+Shift+P` / `Ctrl+Shift+P` | Command palette |
| `Cmd+B` / `Ctrl+B` | Toggle sidebar |
| `Cmd+\`` / `Ctrl+\`` | Toggle terminal |
| `Cmd+/` / `Ctrl+/` | Toggle line comment |
| `Option+↑/↓` / `Alt+↑/↓` | Move line up/down |
| `Shift+Option+↓` / `Shift+Alt+↓` | Duplicate line |
| `Cmd+D` / `Ctrl+D` | Select next occurrence |
| `Cmd+Shift+L` / `Ctrl+Shift+L` | Select all occurrences |
| `F12` | Go to definition |
| `Shift+F12` | Find all references |
| `F2` | Rename symbol (across files) |

### 7.2 Notebook

| Shortcut (macOS / Win-Linux) | Action |
|-------------------------------|--------|
| `Shift+Enter` | Run cell, advance to next |
| `Ctrl+Enter` | Run cell, stay |
| `Option+Enter` / `Alt+Enter` | Run cell, insert below |
| `A` (in command mode) | Insert cell above |
| `B` (in command mode) | Insert cell below |
| `DD` (in command mode) | Delete cell |
| `M` / `Y` (in command mode) | Convert to markdown / code |
| `Esc` / `Enter` | Switch to / from command mode |
| `Z` (in command mode) | Undo cell deletion |
| `Cmd+S` / `Ctrl+S` | Save notebook |

### 7.3 Multi-Cursor

| Shortcut (macOS / Win-Linux) | Action |
|-------------------------------|--------|
| `Option+click` / `Alt+click` | Add cursor at click position |
| `Cmd+Option+↑/↓` / `Ctrl+Alt+↑/↓` | Add cursor above/below |
| `Cmd+U` / `Ctrl+U` | Undo last cursor addition |

Multi-cursor editing replaces many uses of regex find-and-replace; familiarity with `Cmd+D` to select successive matches is one of the highest-leverage skills in VSCode.

---

## 8. Resources

- [VSCode Python documentation](https://code.visualstudio.com/docs/languages/python) — official guide to Python support.
- [VSCode Jupyter documentation](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) — notebook editing, kernels, and debugging.
- [Pylance documentation](https://github.com/microsoft/pylance-release) — type-checking modes and configuration.
- [Ruff documentation](https://docs.astral.sh/ruff/) — linter and formatter rules.
- [VSCode Keyboard Shortcuts Reference (PDF)](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-macos.pdf) — printable cheatsheet (per-OS variants available).
- [VSCode Tips and Tricks](https://code.visualstudio.com/docs/getstarted/tips-and-tricks) — official compendium of less-obvious features.

---

[← Previous: Workflow Guide](05_WORKFLOW_GUIDE.md) | [Index](README.md) | [Next: Python Essentials for ML →](07_PYTHON_ESSENTIALS_FOR_ML.md)
