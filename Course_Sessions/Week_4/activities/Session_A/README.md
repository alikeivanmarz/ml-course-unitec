# Week 4 A: Practical LLMs, VLMs and Agent Workflows

After pulling the course repository, open `Course_Sessions/Week_4/activities/Session_A`. Open [the companion notebook](Week_4_A_Class_Activities.ipynb) in VS Code or Jupyter with the `mlcourse` kernel. Keep `llm_lab.py` and `lab_data/` beside the notebook. Save your own work in a separate notebook in this folder.

Open [the presentation](../../presentations/Session_A_Practical_LLMs_VLMs_and_Agent_Workflows.html) in a browser. The HTML is standalone and works offline. These activity files are supplied alongside it for the coding exercises.

The activities cover language-model requests, structured extraction, course RAG, data tools, document images, drawing comparison, application claims, research summaries, debugging and an integrated tool-using assistant. For each cell: predict the result, match the supplied comments to the numbered blocks, run, inspect and explain.

## Files

- `llm_lab.py`: supplied helpers and small tools. Core code uses Python's standard library, with no PyTorch import or automatic download.
- `setup_lab.py`: preparation checks for the class server (built-in address and semester token) or the local service, local service startup and explicitly permitted missing-model downloads. Importing it changes nothing.
- `lab_data/`: course paraphrases, synthetic model results, a fictional candidate and advertisement, source-linked research summaries, an arithmetic practice function and original teaching SVG/PNG documents.
- `lab_data/recorded_examples.json`: four original model responses from a CPU verification run, including a faulty clothing explanation, checked invoice and drawing extractions, and an agent claim that exceeded its evidence. These are examples to inspect, not benchmark scores.
- `Week_4_A_Class_Activities.ipynb`: 24 authored code cells, with instructions, comment options and outcome checks.

Notebook display uses IPython. Two cells use pandas for tables, already listed in `Environment_Setup/mlcourse.yml`. You do not need to write functions or classes. The supplied helper contains the reusable implementation.

## Model setup

Live model cells need a model service. **In class, use the class server: nothing to install.** Outside class, use Ollama on your own laptop. The presentation, TF-IDF retrieval, CSV comparison, arithmetic checks, evidence checks and small code checker work without model inference.

### In class: use the class server (no installation)

| What | Value |
|---|---|
| Class server address | [https://ml-class.roadinsite.co.nz](https://ml-class.roadinsite.co.nz) (built into the notebook) |
| Class token | Built into `setup_lab.py` for this semester. Nothing to type. |
| When it runs | During class only |

1. Select the Python kernel for your `mlcourse` environment and open the notebook from this folder. After pulling updated helper files, restart the kernel and run from the top.
2. Run **Open the Week 4 activity folder**.
3. Run **Prepare the model service**. It is already set to the class server:

```python
from setup_lab import prepare_lab, CLASS_SERVER_URL

SERVICE = "class"
CLASS_URL = CLASS_SERVER_URL
DOWNLOAD_MISSING_MODELS = True

setup = prepare_lab(SERVICE, CLASS_URL, download_missing=DOWNLOAD_MISSING_MODELS)
```

4. Wait for **READY**, then continue to the first language-model request.

The token is updated each semester through the course repository. If the cell reports that the token was refused, pull the latest course files (`git pull origin main`), restart the kernel and run from the top. If the lecturer shows a token on screen, you can instead rerun the cell and type it when asked. Class mode never installs anything on your laptop and never downloads models to the class server.

You can open [https://ml-class.roadinsite.co.nz](https://ml-class.roadinsite.co.nz) in a browser to check that the server is on. It shows a one-line status message while class is running.

### Outside class: install Ollama once on your laptop

The class server is switched off outside class. To keep working at home, install Ollama and use local mode.

1. Open the official [Windows download page](https://ollama.com/download/windows) or [Mac download page](https://ollama.com/download/mac), then choose **Download manually** for the application installer. Installing it does not require changing the `mlcourse` conda environment.
2. **Windows:** run the official installer, then open Ollama from Start. Check the [Windows requirements](https://docs.ollama.com/windows) if installation is blocked.
3. **Mac:** open the downloaded disk image, drag Ollama into Applications, then open it. Check the [Mac requirements](https://docs.ollama.com/macos), including the supported macOS version.
4. In the preparation cell, change `SERVICE = "class"` to `SERVICE = "local"` and run it. No token is needed.

Use a current Ollama release; Qwen3-VL requires at least 0.12.7. You do not need terminal commands for the normal notebook setup path. A paid API key or cloud-model subscription is not required.

Running the cell in local mode with `DOWNLOAD_MISSING_MODELS = True` permits downloads to this laptop. The first local setup needs internet, sufficient storage for several GB of models, and time to finish. Set the value to `False` if you only want to check the installation without downloading.

In local mode the cell checks for a running service. If needed, it finds the installed application in common Windows and Mac locations and starts its local service. It checks the version and model names, then downloads only missing models. Existing running services and installed models are reused. No installer runs automatically and nothing downloads simply by importing a helper. Model files are stored by Ollama, outside this repository.

### READY and ACTION NEEDED

**READY** means the service responds and the required model names are available. Continue to the first language-model request. It does not guarantee a model's answers will be correct.

**ACTION NEEDED** means stop before live requests. Follow the message, then rerun the same preparation cell. Use Run All only after preparation succeeds. An interrupted local download can be resumed by rerunning setup; do not delete your model files or repeatedly reinstall the environment.

The three default models have separate roles: `qwen3:1.7b` for text and tools, `qwen3-vl:2b-instruct` for images, and `all-minilm` for embeddings. The class server provides these models. The instruct build of Qwen3-VL answers directly; the default `qwen3-vl:2b` build can spend its whole output budget on hidden reasoning and return an empty answer. All live requests use the actual service. There is no scripted replacement for an unavailable model.

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
| Class server cannot be reached or is not running | The class server runs only during class. Check your internet connection and that `CLASS_URL = CLASS_SERVER_URL`. Outside class, set `SERVICE = "local"`. |
| Connection refused in local mode | Open the Ollama app, then rerun preparation. |
| Missing model or interrupted download | Check free storage and internet access; rerun preparation with local downloads enabled. In class mode, tell the lecturer. |
| Version too old | Update the Ollama application and reopen it. In class mode, ask the lecturer to update the server. |
| Class token refused (HTTP 401) | Pull the latest course files (`git pull origin main`), restart the kernel and run from the top. Or rerun the preparation cell and type the token shown by the lecturer. |
| Too many requests (HTTP 429) | Many requests arrived from your network at once. Wait a minute, then run the cell once. |
| First response takes a long time | The whole class shares the class server, so image requests can take up to a minute when everyone runs them together. Wait for the cell to finish; do not press Run repeatedly. In local mode, model loading and CPU inference can be slow. |
| Wrong Python kernel or missing notebook library | Select the existing `mlcourse` environment and restart the kernel. Do not install packages into an unrelated Python environment. |
| Installation is blocked or the laptop is unsupported | Use the class server during class; it needs no installation. Outside class you can still complete the offline tasks below. |

Offline tasks include TF-IDF retrieval, the results CSV comparison, arithmetic checks on the supplied reference, evidence lookups and the bounded bug checker. These do not pretend to be LLM/VLM inference. Tasks which depend on a generated answer need the model service.

If the local service does not start, share the preparation message and `logs/ollama-setup.log` with the lecturer. The local log folder is ignored by Git. Do not include credentials or private files.

The JSON helper removes a surrounding JSON code fence and allows one real model repair request after a validation failure. It never replaces field values with reference answers. If validation still fails, the error retains the response: inspect the prompt, schema and model. If a tool reports a different split, invalid arguments or missing evidence, inspect that outcome before continuing.

Model outputs vary across runtimes and quantisation settings. Treat the presentation's computed diagrams, interactive teaching walkthroughs and recorded model runs according to their labels. Evaluate the actual output you receive.
