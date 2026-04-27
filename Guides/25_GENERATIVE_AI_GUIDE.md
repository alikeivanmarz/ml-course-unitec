# Generative AI Guide

Generative AI creates new content -- text, images, audio -- from learned patterns. This guide covers the generative models and tools used in Weeks 3 and 4 of the course, from GANs to LLMs to diffusion models, including how to build simple applications.

**Table of Contents**

1. [Overview of Generative Models](#1-overview-of-generative-models)
2. [Prompt Engineering Basics](#2-prompt-engineering-basics)
3. [Using OpenAI and Anthropic APIs](#3-using-openai-and-anthropic-apis)
4. [Text Generation with Transformers](#4-text-generation-with-transformers)
5. [Image Generation with Stable Diffusion](#5-image-generation-with-stable-diffusion)
6. [Image Upscaling with Real-ESRGAN](#6-image-upscaling-with-real-esrgan)
7. [LangChain Basics](#7-langchain-basics)
8. [Practical Tips for Inference](#8-practical-tips-for-inference)
9. [Building Simple Apps](#9-building-simple-apps)
10. [Quick Reference Tables](#10-quick-reference-tables)
11. [Resources](#11-resources)

---

## 1. Overview of Generative Models

### 1.1 Types of Generative Models

| Model Type | How It Works | Outputs | Examples |
|-----------|-------------|---------|----------|
| **GANs** | Two networks compete (generator vs discriminator) | Images, video | StyleGAN, CycleGAN |
| **VAEs** | Encode to latent space, decode back | Images, data augmentation | Variational Autoencoder |
| **Diffusion Models** | Learn to reverse a noising process | High-quality images | Stable Diffusion, DALL-E |
| **LLMs** | Predict next token autoregressively | Text | GPT-4, Claude, Llama |
| **Autoregressive** | Generate one element at a time based on previous | Text, audio | GPT, WaveNet |

### 1.2 GANs (Conceptual)

A GAN consists of two neural networks trained together:

- **Generator:** creates fake samples trying to fool the discriminator
- **Discriminator:** tries to distinguish real samples from fakes

They improve together through competition, like a counterfeiter and a detective.

### 1.3 Diffusion Models (Conceptual)

Diffusion models work in two phases:

1. **Forward process:** gradually add noise to an image until it becomes pure noise
2. **Reverse process:** learn to remove noise step by step, reconstructing the image

By learning the reverse process, the model can generate new images from pure noise.

### 1.4 Large Language Models (LLMs)

LLMs generate text by predicting the **next token** (word or subword) given all previous tokens. Key concepts:

- **Temperature:** controls randomness (0 = deterministic, 1 = creative, >1 = chaotic)
- **Top-k:** only consider the top k most likely next tokens
- **Top-p (nucleus):** only consider tokens whose cumulative probability exceeds p
- **Max tokens:** limit the length of generated text

---

## 2. Prompt Engineering Basics

### 2.1 What is Prompt Engineering?

The way you phrase your request to an LLM significantly affects the quality of its response. **Prompt engineering** is the practice of crafting effective prompts.

### 2.2 Prompt Strategies

**Zero-shot:** Ask directly with no examples.

```
Classify the following review as positive or negative:
"The battery life is terrible and the screen cracks easily."
```

**Few-shot:** Provide examples before asking.

```
Classify these reviews:
Review: "Amazing product, works perfectly!" -> Positive
Review: "Broke after two days." -> Negative
Review: "The battery life is terrible and the screen cracks easily." ->
```

**Chain-of-thought:** Ask the model to reason step by step.

```
Determine if this email is spam. Think step by step:
1. What is the email about?
2. Does it contain suspicious elements (urgency, links, too-good offers)?
3. Based on your analysis, is it spam or not?

Email: "Congratulations! You've won a $1000 gift card. Click here to claim now!"
```

### 2.3 Tips for Better Prompts

- **Be specific:** "Summarize in 3 bullet points" is better than "Summarize this"
- **Provide context:** "You are a data science tutor. Explain overfitting to a beginner."
- **Specify format:** "Return the answer as a JSON object with keys: category, confidence"
- **Set constraints:** "Answer in under 50 words"
- **Use system messages:** to define the model's role and behavior

---

## 3. Using OpenAI and Anthropic APIs

> **Important:** Store API keys in environment variables, never hardcode them in your code.

### 3.1 Setting Up API Keys

```bash
# Set environment variables (run in terminal)
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"
```

```python
# In Python, access from environment
import os
openai_key = os.environ.get('OPENAI_API_KEY')
anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
```

### 3.2 OpenAI API

```python
from openai import OpenAI

client = OpenAI()  # Reads OPENAI_API_KEY from environment

response = client.chat.completions.create(
    model='gpt-4o',
    messages=[
        {'role': 'system', 'content': 'You are a helpful ML tutor.'},
        {'role': 'user', 'content': 'Explain overfitting in one paragraph.'}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

### 3.3 Anthropic API

```python
import anthropic

client = anthropic.Anthropic()  # Reads ANTHROPIC_API_KEY from environment

message = client.messages.create(
    model='claude-sonnet-4-20250514',
    max_tokens=200,
    messages=[
        {'role': 'user', 'content': 'Explain gradient descent simply.'}
    ]
)

print(message.content[0].text)
```

### 3.4 API Parameters

| Parameter | What It Controls | Typical Value |
|-----------|-----------------|---------------|
| `model` | Which model to use | `'gpt-4o'`, `'claude-sonnet-4-20250514'` |
| `temperature` | Randomness (0=focused, 1=creative) | 0.0-1.0 |
| `max_tokens` | Maximum response length | 100-4000 |
| `messages` | Conversation history | List of role/content dicts |
| `system` | System instructions (Anthropic) | String |

### 3.5 API Best Practices

- **Handle errors gracefully:** wrap API calls in try/except
- **Be mindful of costs:** set `max_tokens` to limit spending
- **Use streaming** for long responses to show partial results
- **Cache responses** when testing to avoid repeated API calls

---

## 4. Text Generation with Transformers

### 4.1 Using HuggingFace Pipeline

```python
from transformers import pipeline

# Text generation with GPT-2
generator = pipeline('text-generation', model='gpt2')

result = generator(
    'Machine learning is',
    max_length=50,
    num_return_sequences=1,
    temperature=0.7
)

print(result[0]['generated_text'])
```

### 4.2 Generation Parameters

| Parameter | What It Does | Typical Value |
|-----------|-------------|---------------|
| `max_length` | Maximum total length (prompt + generated) | 50-500 |
| `max_new_tokens` | Maximum new tokens to generate | 50-500 |
| `temperature` | Randomness of generation | 0.1-1.0 |
| `top_k` | Consider only top k tokens | 50 |
| `top_p` | Consider tokens with cumulative probability p | 0.9 |
| `num_return_sequences` | Number of different outputs | 1-5 |
| `do_sample` | Enable sampling (vs greedy) | True |

### 4.3 Loading Larger Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'gpt2'  # or a larger model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize input
inputs = tokenizer('Machine learning is', return_tensors='pt')

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True
)

# Decode output
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

---

## 5. Image Generation with Stable Diffusion

### 5.1 Using HuggingFace Diffusers

```python
from diffusers import StableDiffusionPipeline
import torch

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=torch.float16       # Use float16 for less memory
)
pipe = pipe.to('cuda')              # Move to GPU

# Generate an image
image = pipe(
    'a photo of a cat wearing a graduation cap, professional photo',
    num_inference_steps=50,          # More steps = higher quality
    guidance_scale=7.5               # How closely to follow the prompt
).images[0]

# Save
image.save('generated_cat.png')

# Display
import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis('off')
plt.show()
```

### 5.2 Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_inference_steps` | 50 | Higher = better quality, slower |
| `guidance_scale` | 7.5 | Higher = follows prompt more strictly |
| `height` / `width` | 512 | Output image dimensions (multiples of 8) |
| `negative_prompt` | None | What to avoid in the image |

### 5.3 Negative Prompts

Tell the model what to avoid.

```python
image = pipe(
    prompt='a beautiful landscape painting of mountains',
    negative_prompt='blurry, low quality, distorted, ugly, watermark',
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
```

### 5.4 Generating Multiple Images

```python
images = pipe(
    prompt='a futuristic city at sunset',
    num_images_per_prompt=4,
    num_inference_steps=30
).images

# Display in a grid
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, img in zip(axes, images):
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

## 6. Image Upscaling with Real-ESRGAN

### 6.1 What is Super-Resolution?

**Super-resolution** increases the resolution of an image while adding realistic detail. Real-ESRGAN is a state-of-the-art model for this task.

### 6.2 Using Real-ESRGAN

```python
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2
import numpy as np

# Load the model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

upsampler = RealESRGANer(
    scale=4,                        # 4x upscaling
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model,
    half=True                       # Use float16 for speed
)

# Load and upscale an image
img = cv2.imread('low_res_image.jpg')
output, _ = upsampler.enhance(img, outscale=4)
cv2.imwrite('high_res_image.jpg', output)

print(f"Input:  {img.shape}")
print(f"Output: {output.shape}")
```

---

## 7. LangChain Basics

### 7.1 What is LangChain?

**LangChain** is a framework for building applications with LLMs. It provides tools for chaining prompts, connecting to data sources, and managing conversation memory.

### 7.2 Simple LLM Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful ML tutor. Explain concepts clearly.'),
    ('user', '{question}')
])

# Create the model
llm = ChatOpenAI(model='gpt-4o', temperature=0.7)

# Create and run the chain
chain = prompt | llm

response = chain.invoke({'question': 'What is the difference between bagging and boosting?'})
print(response.content)
```

### 7.3 Common Use Cases

| Use Case | LangChain Components |
|----------|---------------------|
| Q&A over documents | Document loaders + vector store + retrieval chain |
| Chatbot with memory | Chat model + conversation memory |
| Summarization | Text splitter + summarization chain |
| Code generation | Prompt template + LLM + output parser |

---

## 8. Practical Tips for Inference

### 8.1 Quantization

Reduces model precision (e.g., float32 to int8) for **faster inference and less memory**.

```python
from transformers import AutoModelForCausalLM

# Load model in 8-bit quantization (requires bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    load_in_8bit=True,
    device_map='auto'
)

# Load in 4-bit (even smaller)
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    load_in_4bit=True,
    device_map='auto'
)
```

### 8.2 Managing GPU Memory

```python
import torch

# Check available GPU memory
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
    print(f"Memory reserved:  {torch.cuda.memory_reserved() / 1024**2:.0f} MB")

# Free unused memory
torch.cuda.empty_cache()
```

### 8.3 CPU vs GPU Inference

```python
import torch

# Check availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")

# Move model to device
model = model.to(device)

# Move inputs to same device
inputs = tokenizer(text, return_tensors='pt').to(device)
```

### 8.4 Memory Reduction Tips

| Technique | Memory Savings | Speed Impact |
|-----------|---------------|-------------|
| Use float16 | ~50% reduction | Slightly faster |
| 8-bit quantization | ~75% reduction | Slightly slower |
| 4-bit quantization | ~87% reduction | Slower |
| Reduce batch size | Proportional | Slower overall |
| Gradient checkpointing | ~50% reduction | ~20% slower |

---

## 9. Building Simple Apps

### 9.1 Streamlit App

**Streamlit** turns Python scripts into interactive web apps with minimal code.

```python
# app.py
import streamlit as st
from transformers import pipeline

st.title('Sentiment Analyzer')
st.write('Enter text below to analyze its sentiment.')

# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    return pipeline('sentiment-analysis')

classifier = load_model()

# User input
user_text = st.text_area('Enter text:', 'This machine learning course is amazing!')

if st.button('Analyze'):
    result = classifier(user_text)[0]
    st.write(f"**Sentiment:** {result['label']}")
    st.write(f"**Confidence:** {result['score']:.3f}")

    if result['label'] == 'POSITIVE':
        st.success('Positive sentiment detected!')
    else:
        st.error('Negative sentiment detected.')
```

**Run the app:**
```bash
streamlit run app.py
```

### 9.2 Gradio Interface

**Gradio** creates quick ML demo interfaces with even less code.

```python
import gradio as gr
from transformers import pipeline

classifier = pipeline('sentiment-analysis')

def analyze_sentiment(text):
    result = classifier(text)[0]
    return f"{result['label']} (confidence: {result['score']:.3f})"

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label='Enter text', placeholder='Type something...'),
    outputs=gr.Textbox(label='Result'),
    title='Sentiment Analyzer',
    description='Enter text to analyze its sentiment.'
)

demo.launch()
```

### 9.3 Streamlit vs Gradio

| Feature | Streamlit | Gradio |
|---------|-----------|--------|
| Setup complexity | Low | Very low |
| Customization | High (full Python scripts) | Medium (function-based) |
| Layout control | More flexible | Simpler, more constrained |
| Sharing | Streamlit Cloud | Gradio share link |
| Best for | Full dashboards, multi-page apps | Quick ML demos |
| Install | `pip install streamlit` | `pip install gradio` |
| Run | `streamlit run app.py` | `demo.launch()` in script |

---

## 10. Quick Reference Tables

### 10.1 Generative Model Comparison

| Model Type | Input | Output | Training Needed | Difficulty |
|-----------|-------|--------|----------------|------------|
| TF-IDF + Classifier | Text | Labels | Yes (fast) | Beginner |
| Pre-trained Pipeline | Text | Text/Labels | No | Beginner |
| Fine-tuned Transformer | Text | Text/Labels | Yes (GPU needed) | Intermediate |
| Stable Diffusion | Text prompt | Images | No (pre-trained) | Beginner |
| Real-ESRGAN | Low-res image | High-res image | No (pre-trained) | Beginner |
| LLM API (OpenAI/Anthropic) | Text prompt | Text | No | Beginner |

### 10.2 API Quick Reference

| Provider | Install | Key Variable | Model Example |
|----------|---------|-------------|---------------|
| OpenAI | `pip install openai` | `OPENAI_API_KEY` | `gpt-4o` |
| Anthropic | `pip install anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| HuggingFace | `pip install transformers` | `HF_TOKEN` (optional) | `gpt2`, `bert-base-uncased` |

### 10.3 Diffusion Parameters Quick Reference

| Parameter | Low Value Effect | High Value Effect | Default |
|-----------|-----------------|------------------|---------|
| `num_inference_steps` | Faster, lower quality | Slower, higher quality | 50 |
| `guidance_scale` | More creative | Follows prompt closely | 7.5 |
| `temperature` (LLMs) | More focused/deterministic | More creative/random | 1.0 |

---

## 11. Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [HuggingFace Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [Real-ESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)

---

**Generative AI is a rapidly evolving field. Start with pre-trained models and APIs before attempting to train your own -- you can build impressive applications with just a few lines of code!**

---

[← Previous: NLP & Transformers](24_NLP_TRANSFORMERS_GUIDE.md) | [Index](README.md) | [Next: Building with LLMs →](26_LLM_PROMPTING_GUIDE.md)
