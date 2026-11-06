# Week 4 A: Practical LLMs, VLMs and Agent Workflows

After pulling the course repository, open `Course_Sessions/Week_4/activities/Session_A`. Open [the companion notebook](Week_4_A_Class_Activities.ipynb) in VS Code or Jupyter with the `mlcourse` kernel. Keep `llm_lab.py` and `lab_data/` beside the notebook. Save your own work in a separate notebook in this folder.

Open [the presentation](../../presentations/Session_A_Practical_LLMs_VLMs_and_Agent_Workflows.html) in a browser. The HTML is standalone and works offline. These activity files are supplied alongside it for the coding exercises.

The activities cover language-model requests, structured extraction, course RAG, data tools, document images, drawing comparison, application claims, research summaries, debugging and an integrated tool-using assistant. For each cell: predict the result, match the supplied comments to the numbered blocks, run, inspect and explain.

## Files

- `llm_lab.py`: supplied helpers and small tools. Core code uses Python's standard library, with no PyTorch import or automatic download.
- `lab_data/`: course paraphrases, synthetic model results, a fictional candidate and advertisement, source-linked research summaries, an arithmetic practice function and original teaching SVG/PNG documents.
- `lab_data/recorded_examples.json`: four original model responses from a CPU verification run, including a faulty clothing explanation, checked invoice and drawing extractions, and an agent claim that exceeded its evidence. These are examples to inspect, not benchmark scores.
- `Week_4_A_Class_Activities.ipynb`: 24 authored code cells, with instructions, comment options and outcome checks.

Notebook display uses IPython. Two cells use pandas for tables, already listed in `Environment_Setup/mlcourse.yml`. You do not need to write functions or classes. The supplied helper contains the reusable implementation.

## Model setup

Live model cells need a running local model service or an endpoint supplied by the lecturer. The presentation, TF-IDF retrieval, CSV comparison, arithmetic checks, evidence checks and small code checker work without model inference.

For a local setup, install [Ollama](https://ollama.com/download) using its official installer. Use a version supporting Qwen3-VL, at least 0.12.7. Open the Ollama application, then run these commands in a terminal:

```sh
ollama pull qwen3:1.7b
ollama pull qwen3-vl:2b
ollama pull all-minilm
ollama list
```

These are explicit initial downloads, totalling several GB. Check that your machine has sufficient storage and memory. If the service is not running, start the application or run `ollama serve`. If it is already running, do not start a second server on the same port.

The notebook helper sends actual requests to the service. It does not replace an unavailable model with a scripted answer. Run the model-check cell before the first live request. The text, vision and embedding models have separate roles. A class endpoint can provide the same API without every student loading models locally.

To use a lecturer endpoint, put this cell before importing `llm_lab`, using the actual address supplied in class:

```python
import os
os.environ["OLLAMA_HOST"] = input("Lecturer model-service URL: ").strip()
```

Defaults are `http://localhost:11434`, text `qwen3:1.7b`, vision `qwen3-vl:2b` and embeddings `all-minilm`. If the lecturer supplies different model names, set `LLM_TEXT_MODEL`, `LLM_VISION_MODEL` and `LLM_EMBED_MODEL` before importing the helper. Restart the kernel if changing model configuration after import. For a token-protected classroom proxy, the helper reads `LLM_LAB_TOKEN` from the environment; never paste a secret into submitted notebooks or presentation HTML.

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

If the connection fails, inspect `ollama list`, the selected model names and the endpoint. If a model name is missing, download that model explicitly or use the class service. The JSON helper removes a surrounding JSON code fence and allows one real model repair request after a validation failure. It never replaces field values with reference answers. If validation still fails, the error retains the response: inspect the prompt, schema and model. If a tool reports a different split, invalid arguments or missing evidence, inspect that outcome before continuing.

Model outputs vary across runtimes and quantisation settings. Treat the presentation's computed diagrams, interactive teaching walkthroughs and recorded model runs according to their labels. Evaluate the actual output you receive.
