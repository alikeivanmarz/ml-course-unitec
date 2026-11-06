# Week 4 A: Practical LLMs, VLMs and Agent Workflows

After pulling the course repository, open `Course_Sessions/Week_4/activities/Session_A`. Open [the companion notebook](Week_4_A_Class_Activities.ipynb) in VS Code or Jupyter with the `mlcourse` kernel. Keep `llm_lab.py` and `lab_data/` beside the notebook. Save your own work in a separate notebook in this folder.

Open [the presentation](../../presentations/Session_A_Practical_LLMs_VLMs_and_Agent_Workflows.html) in a browser. The HTML is standalone and works offline. These activity files are supplied alongside it for the coding exercises.

The activities cover language-model requests, structured extraction, course RAG, data tools, document images, drawing comparison, application claims, research summaries, debugging and an integrated tool-using assistant. For each cell: predict the result, match the supplied comments to the numbered blocks, run, inspect and explain.

## Files

- `llm_lab.py`: supplied helpers and small tools. Core code uses Python's standard library, with no PyTorch import or automatic download.
- `setup_lab.py`: preparation checks, local service startup and explicitly permitted missing-model downloads. Importing it changes nothing.
- `lab_data/`: course paraphrases, synthetic model results, a fictional candidate and advertisement, source-linked research summaries, an arithmetic practice function and original teaching SVG/PNG documents.
- `lab_data/recorded_examples.json`: four original model responses from a CPU verification run, including a faulty clothing explanation, checked invoice and drawing extractions, and an agent claim that exceeded its evidence. These are examples to inspect, not benchmark scores.
- `Week_4_A_Class_Activities.ipynb`: 24 authored code cells, with instructions, comment options and outcome checks.

Notebook display uses IPython. Two cells use pandas for tables, already listed in `Environment_Setup/mlcourse.yml`. You do not need to write functions or classes. The supplied helper contains the reusable implementation.

## Model setup

Live model cells need a running local model service or an endpoint supplied by the lecturer. The presentation, TF-IDF retrieval, CSV comparison, arithmetic checks, evidence checks and small code checker work without model inference.

### Before class: install once

1. Open the official [Windows download page](https://ollama.com/download/windows) or [Mac download page](https://ollama.com/download/mac), then choose **Download manually** for the application installer. Installing it does not require changing the `mlcourse` conda environment.
2. **Windows:** run the official installer, then open Ollama from Start. Check the [Windows requirements](https://docs.ollama.com/windows) if installation is blocked.
3. **Mac:** open the downloaded disk image, drag Ollama into Applications, then open it. Check the [Mac requirements](https://docs.ollama.com/macos), including the supported macOS version.
4. Return to VS Code or Jupyter and select the Python kernel for your `mlcourse` environment. Open the notebook from this activity folder. After pulling updated helper files, restart the kernel and run from the top so it loads the new code.

Use a current Ollama release; Qwen3-VL requires at least 0.12.7. The official platform guides above give current installation requirements. You do not need terminal commands for the normal notebook setup path.

Use local models for these activities. A paid API key or cloud-model subscription is not required for the local setup.

### In the notebook: prepare, then continue

Run **Open the Week 4 activity folder**, followed by **Prepare the model service**. The preparation cell is:

```python
from setup_lab import prepare_lab

SERVICE = "local"
CLASS_URL = ""
DOWNLOAD_MISSING_MODELS = True

setup = prepare_lab(SERVICE, CLASS_URL, download_missing=DOWNLOAD_MISSING_MODELS)
```

Running this cell with `DOWNLOAD_MISSING_MODELS = True` permits downloads to this laptop. The first setup needs internet, sufficient storage for several GB of models, and time to finish. Run it before class when possible. Set the value to `False` if you only want to check the installation without downloading.

The cell checks for a running service. If needed, it finds the installed application in common Windows and Mac locations and starts its local service. It checks the version and model names, then downloads only missing models. Existing running services and installed models are reused. No installer runs automatically and nothing downloads simply by importing a helper. Model files are stored by Ollama, outside this repository.

**READY** means the service responds and the required model names are available. Continue to the first language-model request. It does not guarantee a model's answers will be correct.

**ACTION NEEDED** means stop before live requests. Follow the message, then rerun the same preparation cell. Use Run All only after preparation succeeds. An interrupted download can be resumed by rerunning setup; do not delete your model files or repeatedly reinstall the environment.

The three default models have separate roles: `qwen3:1.7b` for text and tools, `qwen3-vl:2b` for images, and `all-minilm` for embeddings. All live requests use the actual service. There is no scripted replacement for an unavailable model.

### If the lecturer supplies a class server

No local Ollama installation or model download is needed for this mode. In the same preparation cell, set `SERVICE = "class"` and put the actual supplied endpoint in `CLASS_URL`. An empty or invented URL will not work. `localhost` always means your own laptop, not the lecturer's machine. Ask which network or VPN the class endpoint requires.

Class mode checks the endpoint and required model names. It never starts a remote service or downloads weights to the lecturer's server. If a model is missing, ask the lecturer. If different model names are supplied, set `LLM_TEXT_MODEL`, `LLM_VISION_MODEL` and `LLM_EMBED_MODEL` before importing `llm_lab`, then restart the kernel and run from the beginning. For a protected class endpoint, keep the token in `LLM_LAB_TOKEN`; never put it in a URL, saved cell or submitted output. Use only the supplied synthetic/public teaching data.

Only if the lecturer supplies an access token, run this before preparation. The prompt hides the token as you type; the token itself is not written into the code:

```python
import os
from getpass import getpass
os.environ["LLM_LAB_TOKEN"] = getpass("Class access token: ")
```

## Activity structure

| Case | Result to inspect |
|---|---|
| Course assistant | Retrieved passages, generated answer and citation support |
| Document reviewer | VLM fields, source agreement and arithmetic |
| Data assistant | Correct split, actual tool arguments and calculated results |
| Application assistant | Requirement evidence, candidate claims and exact quotations |
| Research assistant | Claims supported by the supplied summaries, with missing information visible |
| Coding agent | A proposed correction, check results and the observed tool trace |
| Final integration | Different requests, selected tools and a reviewed evidence table |

Run the shared setup first, then the cells within each case in order. Some activities use results from their preceding cell. Slide parts are complete Python blocks: run the parts in order, or use **Copy complete cell** once. The notebook and slide code come from the same authored collection.

## Checking the results

Read the source before judging an extraction. The invoice reference total is NZD 35.65, from 2 × 12.50 + 3 × 2.00 = 31.00, plus 4.65 GST. Drawing revisions A and B print 2850 mm and 3000 mm, a 150 mm difference. These are synthetic teaching records.

The results CSV contains synthetic scores. Compare only rows with the same `split`. The candidate is fictional. The research file contains authored summaries linked to three actual papers; it does not provide full-paper evidence or a shared accuracy benchmark.

The code checker interprets a small allowed arithmetic AST and never executes arbitrary generated Python. Its four positive-total examples are a bounded check, not proof for every input. The agent's final message and its status also need evidence review. A model can choose the right tool and still explain its result incorrectly.

If a proposed correction repeats the bug, or an agent says it will test without returning a test result, the task is unfinished. Keep that output, make the requirement more specific and compare the next attempt. A response claiming that a supplied tool is unavailable should be checked against the available tool definitions and actual trace.

## Troubleshooting

| Message or situation | What to do |
|---|---|
| Ollama is not installed | Use the official installation link above, open the app, then rerun preparation. `pip install ollama` alone does not install the service. |
| App is installed but the notebook cannot find it | Open the app once. The helper checks common app locations even when the notebook has an old PATH. For a custom install, ask the lecturer or restart VS Code/Jupyter after installation. |
| Connection refused | Rerun preparation. Local mode needs the local service; class mode needs the supplied server URL and network access. |
| Missing model or interrupted download | Check free storage and internet access; rerun preparation with local downloads enabled. In class mode, ask the lecturer. |
| Version too old | Update the Ollama application and reopen it. In class mode, ask the lecturer to update the server. |
| HTTP 401 or 403 | Ask the lecturer about access permissions and token configuration. Do not paste a token into saved code. |
| First response takes a long time | Model loading and CPU inference can be slow. Wait for the cell to finish; do not queue repeated requests. After a timeout, try a shorter request or ask whether the class server is busy. |
| Wrong Python kernel or missing notebook library | Select the existing `mlcourse` environment and restart the kernel. Do not install packages into an unrelated Python environment. |
| Installation is blocked or the laptop is unsupported | Ask whether a class endpoint is available. You can still complete the offline tasks below. A class server must actually be supplied; it is not automatically provided by this repository. |

Offline tasks include TF-IDF retrieval, the results CSV comparison, arithmetic checks on the supplied reference, evidence lookups and the bounded bug checker. These do not pretend to be LLM/VLM inference. Tasks which depend on a generated answer need the model service.

If the local service does not start, share the preparation message and `logs/ollama-setup.log` with the lecturer. The local log folder is ignored by Git. Do not include credentials or private files.

The JSON helper removes a surrounding JSON code fence and allows one real model repair request after a validation failure. It never replaces field values with reference answers. If validation still fails, the error retains the response: inspect the prompt, schema and model. If a tool reports a different split, invalid arguments or missing evidence, inspect that outcome before continuing.

Model outputs vary across runtimes and quantisation settings. Treat the presentation's computed diagrams, interactive teaching walkthroughs and recorded model runs according to their labels. Evaluate the actual output you receive.
