# Weekly Course Objectives and accompanying Jupyter Notebook Exercises

## Prerequisites

This course is designed for professionals with 2-4 years of experience as a software engineer, data scientist, machine learning engineer, or similar. You should have some background in data science basics, experience building data pipelines, and familiarity with analytics and visualization. You should have worked with basic machine learning models, understand text processing fundamentals and know of distributed computing frameworks like Spark. We expect you're already using version control and have collaborated on software projects. Some exposure to university-level mathematical concepts (e.g. differential equations, statistics, linear algebra) will help you grasp the concepts behind training and deploying language models at scale more quickly.

## Introduction: Understanding Language Models
### Week 1: LLM Fundamentals and Development Environment Setup
Resource Required: Google Colab (T4) GPU or better (RunPod 1x A10G/L40s recommended)
- **Notebook 1.1: Background and Introduction**
  - Introduction to Large Language Models
  - Overview of transformer architecture
  - History and evolution of language models
- **Notebook 1.2: Introducing PyTorch**
  - Explore PyTorch basics for LLM implementation
  - Tensor operations and autograd functionality
  - Building neural networks with PyTorch
- **Notebook 1.3: Tokenisation**
  - Understanding tokenization in language models
  - Byte-Pair Encoding (BPE) algorithm implementation
  - SEA-LION multilingual tokenization example and insights

  HOML (Jay Alammar)
  To Extend Later: Word2Vec and basic contrastive learning (HOML ch2); relate to training own Sentence Embedding model (HOML ch9) and also multimodality (HOML ch10), KV-Cache (HOML ch2)

## Pre-training Language Models
### Week 2: Transformer Architecture Deep Dive
- **Notebook 2.1: Introduce Backpropagation**
  - Build a tiny autograd engine
  - Explain loss.backward() as seen in lesson 1.2
- **Notebook 2.2: Simple Transformer to understand logits, softmax, sigmoid, logprobs, how LM Head works as final layer before generating next token**
  - Train the model on tiny Shakespeare dataset
  - Demonstrate how a model can "learn", somewhat semantic, way to generate text from a training dataset
  - Consider explaining progression from bigram (n-gram) to RNN to Transformer
- **Notebook 2.3: Introduce why self-attention, and how it evolved from RNN for sequence modelling**
  - Implement simplified versions of encoder-only (BERT-like) models
  - Compare performance of decoder-only (GPT-like) architectures
  - Experiment with encoder-decoder (T5-like) designs on translation tasks (E.g. NLLB)

### Week 3: Advanced Transformer Components
- **Notebook 3.1: GPT-2 to Llama Architecture**
  - Code attention mechanism from scratch (Q, K, V matrices)
  - Implement scaled dot-product attention with masking
- **Notebook 3.2: Explain Optimizations**
  - Code multi-head attention from scratch
  - Implement positional embeddings and encodings
  - Build feed-forward networks with different activation functions
- **Notebook 3.2: Explain Hyperparameters when training**
  - Implement and benchmark KV-cache for efficient inference
  - Visualize memory usage with and without KV-cache

### Week 4: Data Pipeline Engineering
Resource Required: 8xA100 GPU
- **Notebook 4.1: Synthetic Data Generation**
  - Design and implement synthetic data generation pipeline for generating high quality pretraining data based
  - Create template seed prompts with different content styles (textbook, blog, etc.) to ensure diversity and rich coverage in data generation
  - Analyze and visualize generated data, collect feedback on quality to improve generation pipeline
- **Notebook 4.2: Data Filtering**
  - Implement simple data filtering pipeline which involves deduplication, decontamination, and quality scoring
  - Understand how data filtering fits into the overall data generation pipeline
  - Measure and analyse quality impact of filtering steps to improve data pipeline
- **Notebook 4.3: Web-Scale PDF Processing**
  - Implement an educational example of a real world data pipeline for extracting high quality Indonesian texts from a PDF
  - Understand the challenges and complexities in building a scalable data pipeline for web scale data extraction
  - Understand the role of Google Translate and NLLB in multilingual data pipelines

## Evaluating Language Models
### Week 5: Inference Optimization and Deployment
Resource Required: Google Colab (T4) GPU or better (RunPod 1x A10G/L40s recommended)
- **Notebook 5.1: Sampling Parameters**
  - Explore different sampling strategies (greedy, temperature, top-k, top-p)
  - Implement beam search and repetition penalty techniques
  - Analyze effects of parameters on generation quality and diversity
- **Notebook 5.2: vLLM Sampling**
  - Benchmark vLLM against standard HF/PyTorch inference
  - Explore PagedAttention and continuous batching optimizations
  - Compare and contrast vLLM and Llama.cpp
- **Notebook 5.3: Quantization**
  - Convert models to optimized GGUF format for llama.cpp
  - Implement QLoRA for efficient fine-tuning
  - Analyze accuracy-performance tradeoffs

### Week 6: Evaluation
Resource Required: Google Colab (T4) GPU or better (RunPod 1x A10G/L40s recommended)
- **Notebook 6.1: Benchmark Development**
  - Create custom evaluation benchmarks for specific domains
  - Implement automated evaluation for math and coding tasks
- **Notebook 6.2: Evaluation Methods**
  - Compare reference-free vs. reference-based evaluation
  - Implement human-in-the-loop evaluation interfaces
  - Create visualizations for model performance analysis
- **Notebook 6.3: Lite-Eval Implementation**
  - Build lightweight evaluation pipelines for rapid testing
  - Analyze correlation between lite-eval and comprehensive benchmarks

## Post-training and Fine-tuning Language Models
### Week 7: Fine-tuning and Adaptation Methods
Resource Required: Google Colab (T4) GPU or better (RunPod 1x A10G/L40s recommended)
- **Notebook 7.1: Fine-tuning (SFT) for classification (using spam classification dataset)**
  - Why (when to) FineTune (Chip ch7)
  - Implement full fine-tuning on domain-specific datasets
  - Compare performance before and after fine-tuning
- **Notebook 7.2: Instruction Finetuning (QLora) in PyTorch**
  - Implement LoRA from scratch
  - Build adapter modules and prompt tuning mechanisms
  - Compare parameter counts and performance across methods
- **Notebook 7.3: DPO Preference Finetuning for reward models (LoRA)**
  - Create instruction tuning datasets and training loops
  - Implement simplified RLHF pipeline components

## The future of Language Models
### Week 8: Multimodality, RL
- **Notebook 8.1: Multimodal Integration**
  - Introduction to early and late fusion techniques
  - Build simple vision-language model components (PaliGemma walkthrough)
  - Auto-regressive image generation (GPT4o)
- **Notebook 8.2: Introduction to RL**
  - Create reward modeling components
  - Introduce RLHF
- **Notebook 8.3: Agents**
  - Build simple agent scaffold
  - Explore frontier research in agentic models