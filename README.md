# Retrieval-Augmented Question Answering System

## Overview

This is a personal side project where I explored building a Retrieval Augmented Generation (RAG) system to answer questions accurately using natural language processing. I started with a basic language model and then enhanced it by adding retrieval capabilities. The project involves two main phases: one using the LangChain framework and another building a custom RAG system from scratch. My goal was to experiment with question-answering systems, optimize retrieval performance, and dive deep into embeddings and document search techniques—all while having fun with some cool datasets and models.

## Features

- **Dual RAG Systems**: Built one RAG system with LangChain and another custom one without it.
- **Question Answering**: Generates answers using the `google/flan-t5-base` model.
- **Custom Retrieval**: Uses `sentence-transformers` for embeddings and FAISS for efficient document retrieval.
- **Parameter Tuning**: Experimented with chunk sizes, k-values, and prompts to boost performance.
- **Evaluation**: Measured success with F1 scores and exact match metrics.

## Datasets

I worked with two datasets to fuel this project:
- **Questions Dataset** (`questions-1.csv`): Contains 100 questions, like "Which presidential administration developed Safe Harbor policy?"
- **Passages Dataset** (`passages-1.csv`): Includes 525 text passages for context, such as "The 'Safe Harbor' agreement is a voluntary agreement between the private landowner and FWS..."

## Technical Details

- **Language**: Python
- **Models**:
  - Question Answering: `google/flan-t5-base` (Hugging Face)
  - Embeddings: `sentence-transformers/roberta-base-nli-stsb-mean-tokens`
- **Tools**:
  - LangChain (for the first system)
  - FAISS (for vector search in the custom system)
  - Sentence Transformers (for generating embeddings)
- **Purpose**: Enhance question-answering accuracy with retrieval-augmented generation.

## Experiments

### LangChain RAG System
I tested different configurations to see what works best:
- **Chunk Size**: Tried 500, 1000, and 2000. Surprisingly, performance stayed at F1=75.3 regardless—maybe the key info fits in smaller chunks!
- **K-Value**: Tested k=1, 3, 5. Bigger k (more retrieved docs) improved results slightly (F1=66.9 to 79.7).
- **Prompts**: Played with prompt order and style. Putting context before the question worked best (F1=75.7), while adding fancy modifiers didn’t help much.

**Best Config**: Chunk size=1000, k=5, prompt="Context: {context} Question: {question} Answer:"  
**Results**: Exact Match=68.0, F1=79.7

### Custom RAG System
Built my own RAG system from scratch and tinkered with it:
- **K-Value**: Tested k=1, 3, 4, 5. Found k=4 was the sweet spot (F1=61.4), but k=5 added noise (F1=57.4). Settled on k=3 for stability.
- **Prompts**: Tested various formats. Question-first prompts crushed it (F1=60.1), while context-first tanked (F1=23.3). Adding "answer:" hurt performance too.

**Best Config**: k=4, prompt="question: {question} context: {context}"  
**Results**: Exact Match=52.0, F1=61.4

## Key Findings
- **Chunk Size Oddity**: Bigger chunks didn’t improve results—possibly because critical info was short enough to fit anywhere.
- **K-Value Sweet Spot**: More retrieved docs help, but too many add noise (especially in the custom system).
- **Prompt Sensitivity**: The same model loved context-first prompts with LangChain but flipped to question-first without it. Prompt design is wild!

## Showcase

Since this project is all about experimentation, here’s a look at the key results from my tuning efforts. I’ve summarized the performance of both RAG systems with tables to highlight how different configurations stacked up. All scores are F1 metrics unless noted otherwise.

### LangChain RAG Highlights
Tested chunk sizes, k-values, and prompts to find the sweet spot:

| Chunk Size | K-Value | Prompt Style                          | F1 Score |
|------------|---------|---------------------------------------|----------|
| 500        | 3       | "Question: {q} Context: {c} Answer:"  | 75.3     |
| 1000       | 3       | "Question: {q} Context: {c} Answer:"  | 75.3     |
| 2000       | 3       | "Question: {q} Context: {c} Answer:"  | 75.3     |
| 1000       | 1       | "Context: {c} Question: {q} Answer:"  | 66.9     |
| 1000       | 5       | "Context: {c} Question: {q} Answer:"  | 79.7     |

*Best Config*: Chunk Size=1000, K=5, Prompt="Context: {context} Question: {question} Answer:"  
- Exact Match: 68.0  
- F1 Score: 79.7  

**Takeaway**: K-value mattered more than chunk size—bigger k boosted performance, but chunk size didn’t budge the needle.

### Custom RAG Highlights
Explored k-values and prompt variations in my DIY system:

| K-Value | Prompt Style                          | F1 Score |
|---------|---------------------------------------|----------|
| 1       | "question: {q} context: {c}"          | 56.8     |
| 3       | "question: {q} context: {c}"          | 60.1     |
| 4       | "question: {q} context: {c}"          | 61.4     |
| 5       | "question: {q} context: {c}"          | 57.4     |
| 3       | "context: {c} question: {q}"          | 23.3     |

*Best Config*: K=4, Prompt="question: {question} context: {context}"  
- Exact Match: 52.0  
- F1 Score: 61.4

## Why I Built This
I’m fascinated by how retrieval can supercharge language models, so I dove into RAG systems to see what I could create. It’s been a blast experimenting with embeddings, FAISS, and prompt tweaks—plus, I learned a ton about balancing accuracy and noise in NLP.

## Next Steps
- Test more prompts (prompt engineering is a rabbit hole!).
- Try other embedding models for better retrieval.
- Scale it up with bigger datasets.
