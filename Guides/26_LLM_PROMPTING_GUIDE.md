# Building with Large Language Models

This guide covers the application craft of building with text-based large language models: structuring requests, designing system prompts, extracting structured output, calling tools, managing multi-turn state, and evaluating outputs. The focus is on patterns that recur across providers (OpenAI, Anthropic, open-weight models served locally) rather than provider-specific syntax. Examples use the Anthropic and OpenAI Python SDKs as representative.

**Table of Contents**

1. [Anatomy of a Chat Completion Request](#1-anatomy-of-a-chat-completion-request)
2. [System-Prompt Design Patterns](#2-system-prompt-design-patterns)
3. [Few-Shot and In-Context Examples](#3-few-shot-and-in-context-examples)
4. [Structured Output](#4-structured-output)
5. [Tool / Function Calling](#5-tool--function-calling)
6. [Streaming Responses](#6-streaming-responses)
7. [Multi-Turn Conversation Management](#7-multi-turn-conversation-management)
8. [Retrieval-Augmented Generation](#8-retrieval-augmented-generation)
9. [Cost, Tokens, and Latency](#9-cost-tokens-and-latency)
10. [Evaluating LLM Outputs](#10-evaluating-llm-outputs)
11. [Resources](#11-resources)

---

## 1. Anatomy of a Chat Completion Request

Modern LLM APIs are organised around a sequence of messages with explicit roles.

| Role | Purpose |
|------|---------|
| `system` | Sets behaviour, style, and constraints. Persists for the whole conversation. |
| `user` | Input from the human (or upstream system) |
| `assistant` | Prior model responses, replayed to provide conversational context |
| `tool` (provider-specific naming) | Result of a tool call invoked by the assistant |

```python
# OpenAI SDK
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Reply in formal English."},
        {"role": "user", "content": "Summarize the second law of thermodynamics."},
    ],
    temperature=0.2,
    max_tokens=300,
)
print(response.choices[0].message.content)
```

```python
# Anthropic SDK
from anthropic import Anthropic
client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    system="Reply in formal English.",
    messages=[
        {"role": "user", "content": "Summarize the second law of thermodynamics."},
    ],
    max_tokens=300,
)
print(response.content[0].text)
```

Key parameters: `temperature` (randomness, 0–1 typical), `max_tokens` (output cap), `top_p` (nucleus sampling cutoff). Lower temperature for factual tasks; higher for creative generation.

---

## 2. System-Prompt Design Patterns

The system prompt is the highest-leverage part of the request. Effective patterns:

### 2.1 Role + Constraints + Format

```
You are an expert technical reviewer.

Constraints:
- Respond in under 200 words.
- Cite line numbers for every issue raised.
- Do not propose changes outside the scope of the diff.

Output format: Markdown with a heading per issue.
```

Three components, each on its own block: who the model is, what limits apply, and how the response should be shaped.

### 2.2 Refusal Conditions

Explicit refusal conditions improve safety and predictability:

```
If the request involves <X>, refuse and explain why in one sentence.
If the request is ambiguous, ask one clarifying question before proceeding.
```

### 2.3 Instruction Stability

Prefer positive instructions ("Use Markdown tables for comparisons") to negative ones ("Don't use bullet lists"). Negative instructions are followed less reliably than positive equivalents.

### 2.4 Anti-Patterns

| Anti-pattern | Symptom | Better |
|--------------|---------|--------|
| Long, unstructured prose | Inconsistent compliance | Bulleted constraints |
| Conflicting instructions | Erratic behaviour | Order by priority; remove conflicts |
| Vague qualifiers ("be helpful", "be concise") | Subjective output | Quantify ("≤ 100 words", "include 3 examples") |
| Examples mixed with instructions | Confused style | Separate "Examples:" section |

---

## 3. Few-Shot and In-Context Examples

Demonstrating the desired output format with examples is often more effective than describing it.

```python
messages = [
    {"role": "system", "content": "Classify each sentence as POS, NEG, or NEU."},
    {"role": "user", "content": "I loved the film."},
    {"role": "assistant", "content": "POS"},
    {"role": "user", "content": "It was okay, nothing special."},
    {"role": "assistant", "content": "NEU"},
    {"role": "user", "content": "Worst meal of the year."},
    {"role": "assistant", "content": "NEG"},
    {"role": "user", "content": "The pacing dragged in the middle act."},
]
```

### 3.1 Example Selection

- 3–5 examples are usually sufficient for format tasks.
- Cover edge cases: short, long, ambiguous, near-boundary.
- Maintain class balance across examples; imbalance biases the model.

### 3.2 When Few-Shot Fails

For tasks requiring novel reasoning (rather than format imitation), few-shot examples may be ignored. Consider zero-shot with explicit reasoning instructions ("think step by step before answering") or move to fine-tuning if the same task is run at scale.

---

## 4. Structured Output

Production systems consume structured data, not prose. Three mechanisms:

### 4.1 JSON Mode

Most providers support a `response_format={"type": "json_object"}` parameter that constrains output to valid JSON.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Respond with a JSON object containing 'sentiment' and 'confidence'."},
        {"role": "user", "content": "The pacing dragged in the middle act."},
    ],
    response_format={"type": "json_object"},
)

import json
data = json.loads(response.choices[0].message.content)
```

### 4.2 JSON Schema

Stricter than JSON mode: the model output is constrained to a specific schema, with required fields and types enforced.

```python
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["POS", "NEG", "NEU"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["sentiment", "confidence"],
}

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    response_format={"type": "json_schema", "json_schema": {"name": "Sentiment", "schema": schema}},
)
```

### 4.3 Pydantic Validation

Even with schema enforcement, parse and validate downstream:

```python
from pydantic import BaseModel, Field

class Sentiment(BaseModel):
    sentiment: str = Field(pattern="^(POS|NEG|NEU)$")
    confidence: float = Field(ge=0, le=1)

result = Sentiment.model_validate_json(response.choices[0].message.content)
```

Pydantic catches schema drift, type mismatches, and out-of-range values that pure JSON parsing accepts silently.

---

## 5. Tool / Function Calling

Tool calling lets the model invoke external functions — fetch data, perform calculations, query databases — instead of fabricating answers.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Returns current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)

# If the model decides to call a tool:
tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
result = get_weather(**args)

# Send the result back for a final response
followup = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"},
        response.choices[0].message,
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
    ],
    tools=tools,
)
```

### 5.1 Tool Design Principles

| Principle | Rationale |
|-----------|-----------|
| Descriptive names and descriptions | The model decides whether to call the tool from the description alone |
| Narrow scope per tool | Easier for the model to choose correctly |
| Validated arguments | Schema constraints reduce invalid calls |
| Structured returns | Plain text returns cause the model to re-narrate; structured returns cite cleanly |
| Explicit error returns | "Failed because X" is more useful than a thrown exception |

---

## 6. Streaming Responses

Streaming returns tokens as they are generated, reducing perceived latency.

```python
# OpenAI
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

```python
# Anthropic
with client.messages.stream(
    model="claude-sonnet-4-6",
    messages=[...],
    max_tokens=500,
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

Streaming complicates structured-output handling (incomplete JSON cannot be parsed mid-stream); collect the full response before validation.

---

## 7. Multi-Turn Conversation Management

### 7.1 Context Window Limits

Every model has a maximum context length. As a conversation grows, older messages must be dropped or summarized.

| Strategy | Mechanism | Trade-off |
|----------|-----------|-----------|
| Sliding window | Keep last N turns | Loses early context entirely |
| Summarization | Replace old turns with a generated summary | Lossy but preserves themes |
| Hybrid | Summary of old + verbatim recent turns | Most common in production |
| External memory | Store turns in a vector DB; retrieve relevant ones | Decouples conversation length from context length |

### 7.2 Token Counting

Count tokens before sending requests to avoid context overflow:

```python
# OpenAI uses tiktoken
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o-mini")
n_tokens = len(enc.encode("Some text"))
```

```python
# Anthropic
client.messages.count_tokens(model="claude-sonnet-4-6", messages=messages)
```

Reserve 10–20% of the context for the response and any tool calls.

---

## 8. Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) supplements model knowledge with retrieved documents at inference time. The minimal pattern:

```python
# Pseudocode — vector store and embedding model are pluggable

# 1. Index (offline, once)
docs = load_corpus()
embeddings = embed(docs)
index.add(embeddings, metadata=docs)

# 2. Retrieve (per query)
query_emb = embed([user_query])
hits = index.search(query_emb, top_k=5)
context = "\n\n".join(h.text for h in hits)

# 3. Generate with context
messages = [
    {"role": "system", "content": "Answer the question using only the provided context. If the answer is not in the context, say so."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"},
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
```

### 8.1 RAG Quality Levers

| Lever | Effect |
|-------|--------|
| Chunk size (200–800 tokens) | Smaller = more precise retrieval; larger = more context per chunk |
| Embedding model | Quality of semantic match; domain-specific models help |
| Top-k retrieved | Recall vs context budget |
| Reranking | A second model reorders retrieved chunks; usually improves precision |
| Hybrid search | Combine semantic and keyword (BM25) search for robustness |

---

## 9. Cost, Tokens, and Latency

### 9.1 Token Math

Pricing is per million tokens, billed separately for input and output. A rough heuristic: 1 token ≈ 4 characters of English; a 750-word page is ~1,000 tokens.

| Cost driver | Mitigation |
|-------------|------------|
| Long system prompts | Prompt caching (re-uses cached prefixes at reduced cost) |
| Verbose outputs | Lower `max_tokens`; instruct concision |
| Repeated context | Cache, summarize, or move to external memory |
| Wrong model tier | Use cheaper models for routing/triage; expensive only for final answer |

### 9.2 Latency Reduction

| Technique | Typical effect |
|-----------|----------------|
| Streaming | Lower time-to-first-token |
| Smaller model | Lower per-token latency |
| Parallel requests | Higher throughput; unchanged per-request latency |
| Prompt caching | Lower cost; modest latency improvement on cached prefix |
| Reduced output length | Linear reduction in generation time |

---

## 10. Evaluating LLM Outputs

LLM outputs cannot be evaluated by exact-match accuracy in most cases. A practical evaluation suite combines several signals.

### 10.1 Evaluation Layers

| Layer | Method | Cost | Use |
|-------|--------|------|-----|
| Unit assertions | Programmatic checks (length, format, regex, schema) | Low | Always; first line of defence |
| Heuristic metrics | BLEU, ROUGE, exact match | Low | Reference-based tasks (translation, summarization with reference) |
| Embedding similarity | Cosine similarity to reference | Low | Semantic similarity without exact match |
| LLM-as-judge | Second model scores the output | Medium | Open-ended tasks; pair with rubric |
| Human review | Annotators score samples | High | Ground truth; calibrate cheaper methods |

### 10.2 Regression Suites

Maintain a test set of representative inputs with expected behaviours. Run after every prompt or model change. Track:

- Pass rate per category (format, factuality, refusal correctness, tool-call correctness)
- Cost per evaluated sample
- Drift between model versions

### 10.3 LLM-as-Judge Patterns

```python
judge_prompt = """
Score the response on a scale of 1–5 for the following criteria:
- Factual accuracy
- Adherence to format
- Completeness

Response to score:
{response}

Reference (ground truth):
{reference}

Output JSON: {"accuracy": int, "format": int, "completeness": int, "rationale": str}
"""
```

LLM judges are biased toward longer responses and toward outputs from models in the same family. Calibrate against a small human-annotated sample before using at scale.

---

## 11. Resources

- [OpenAI API documentation](https://platform.openai.com/docs/) — chat completions, structured output, function calling.
- [Anthropic API documentation](https://docs.anthropic.com/) — messages API, tool use, prompt caching.
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) — applied prompting patterns.
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — model-specific recommendations.
- [Lewis et al., *Retrieval-Augmented Generation* (2020)](https://arxiv.org/abs/2005.11401) — original RAG paper.
- [Wei et al., *Chain-of-Thought Prompting* (2022)](https://arxiv.org/abs/2201.11903) — step-by-step reasoning prompts.
- [Zheng et al., *Judging LLM-as-a-Judge* (2023)](https://arxiv.org/abs/2306.05685) — biases and calibration of LLM judges.
- [Pydantic documentation](https://docs.pydantic.dev/) — schema validation for LLM outputs.

---

[← Previous: Generative AI](25_GENERATIVE_AI_GUIDE.md) | [Index](README.md) | [Next: ML Project Structure →](27_PROJECT_STRUCTURE_GUIDE.md)
