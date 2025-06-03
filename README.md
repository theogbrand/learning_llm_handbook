# Weekly Course Objectives and accompanying Jupyter Notebook Exercises

## Prerequisites

This course is designed for professionals with 2-4 years of experience as a software engineer, data scientist, machine learning engineer, or similar. You should have some background in data science basics, experience building data pipelines, and familiarity with analytics and visualization. You should have worked with basic machine learning models, understand text processing fundamentals and know of distributed computing frameworks like Spark. We expect you're already using version control and have collaborated on software projects. Some exposure to university-level mathematical concepts (e.g. differential equations, statistics, linear algebra) will help you grasp the concepts behind training and deploying language models at scale more quickly.

## Introduction: Understanding Language Models
### Week 1: LLM Fundamentals and Development Environment Setup
Resource Required: Google Colab (T4) GPU or better (RunPod 1x A10G/L40s recommended)
- **Notebook 1.1: Background and Introduction**
  - Set up a LLM development environment with GPU acceleration
  - Compare base models vs instruction-tuned models through hands-on experiments
  - Generate text completions and analyze model outputs for different use cases
- **Notebook 1.2: Introducing PyTorch**
  - Build neural networks from scratch using PyTorch's tensor operations
  - Train a complete image classifier on FashionMNIST dataset with 90%+ accuracy
  - Implement custom training loops with optimizers, loss functions, and DataLoaders
  - Apply transfer learning to adapt pre-trained models for new tasks
- **Notebook 1.3: Tokenization**
  - Implement Byte-Pair Encoding (BPE) tokenizer from scratch in pure Python
  - Handle UTF-8 encoding and Unicode text processing for multilingual applications
  - Analyze tokenization efficiency across languages using fertility rate metrics
  - Compare and choose between tokenization libraries (tiktoken, SentencePiece) for production use

## Pre-training Language Models
### Week 2: Transformer Architecture Deep Dive
- **Notebook 2.1: Backpropagation**
  - Build a minimal autograd engine (micrograd) to understand how PyTorch works internally
  - Implement computational graphs and automatic gradient calculation from scratch
  - Verify gradients manually using calculus and the chain rule
  - Train neural networks using your own automatic differentiation system
- **Notebook 2.2: Multilayer Perceptrons (MLPs)**
  - Create a character-level language model that generates Shakespeare-like text
  - Implement neural probabilistic language models following Bengio et al. (2003)
  - Train word embeddings and visualize learned semantic relationships
  - Generate coherent text using only MLPs (without attention)
- **Notebook 2.3: Attention Introduction**
  - Implement self-attention mechanism from first principles in NumPy and PyTorch
  - Build the Query, Key, Value computation pipeline for attention
  - Design multi-head attention layers with proper masking and scaling
  - Construct a complete mini-GPT model and generate text with transformers

### Week 3: Advanced Transformer Components
- **Notebook 3.1: GPT Architecture**
  - Build a complete GPT-2 model from scratch in PyTorch
  - Implement causal attention masks for autoregressive generation
  - Generate coherent text using temperature and top-k sampling strategies
  - Debug and visualize attention patterns in decoder-only architectures
- **Notebook 3.2: Model Training Optimizations**
  - Accelerate training by 3-5x using mixed precision and tensor cores
  - Implement Flash Attention to reduce memory usage by 10-20x
  - Profile and optimize PyTorch code with torch.compile
  - Diagnose and fix common GPU memory bottlenecks
- **Notebook 3.3: Model Training Hyperparameters**
  - Configure AdamW optimizer with gradient clipping for stable training
  - Design learning rate schedules that improve convergence speed
  - Scale training across multiple GPUs using Distributed Data Parallel
  - Evaluate your model's reasoning capabilities using HellaSwag benchmark
- **Notebook 3.4: From GPT-2 to LLaMA3**
  - Upgrade models with RMSNorm for 2x faster training
  - Implement Rotary Position Embeddings (RoPE) for better length generalization
  - Reduce memory usage by 3x with Group Query Attention (GQA)
  - Boost model performance using SwiGLU activation functions

### Week 4: Data Pipeline Engineering
Resource Required: Google Colab (T4) GPU or better
- **Notebook 4.1: Synthetic Data Generation**
  - Generate high-quality educational content using the Cosmopedia methodology
  - Design and execute topic clustering pipelines for diverse training data
  - Create content in multiple styles (textbook, blog, wikihow) for varied learning
  - Deploy and orchestrate local LLMs with Ollama for scalable generation
- **Notebook 4.2: Data Filtering**
  - Build MinHash deduplication pipelines to remove redundant data at scale
  - Implement decontamination checks to prevent benchmark leakage
  - Train and deploy BERT classifiers to score educational content quality
  - Design multi-stage filtering pipelines that balance quality and quantity
- **Notebook 4.3: Real-World Data Extraction**
  - Extract clean text from complex PDFs using Marker at production scale
  - Build OCR pipelines that handle scanned documents and images
  - Create multilingual processing pipelines with Google Translate and NLLB
  - Optimize data extraction workflows for web-scale document processing

## Evaluating Language Models
### Week 5: Inference Optimization and Deployment
Resource Required: Google Colab (T4) GPU or better (RunPod 1x A10G/L40s recommended)
- **Notebook 5.1: Sampling Techniques**
  - Build custom sampling functions to control LLM output quality and diversity
  - Implement temperature, top-k, and top-p sampling from scratch to understand generation mechanics
  - Design beam search algorithms for finding optimal text sequences
  - Create interactive visualizations to debug and optimize probability distributions
- **Notebook 5.2: Model Inference Serving**
  - Deploy production-ready LLM services with 10x throughput improvements
  - Configure PagedAttention to serve multiple users with minimal GPU memory
  - Optimize inference pipelines using continuous batching and Flash Attention
  - Build and deploy OpenAI-compatible APIs for seamless integration
- **Notebook 5.3: Quantization**
  - Reduce model size by 75% while maintaining 95%+ of original performance
  - Implement quantization algorithms (absmax, zero-point) to understand precision tradeoffs
  - Deploy models on edge devices using GGUF format and llama.cpp
  - Fine-tune quantized models with QLoRA for memory-efficient adaptation

### Week 6: Evaluation
Resource Required: Google Colab (T4) GPU or better (RunPod 1x A10G/L40s recommended)
- **Notebook 6.1: Introduction to Model Evaluation**
  - Design comprehensive evaluation frameworks for both MCQA and generative tasks
  - Introduce common evaluation methodologies such as loglikelihood_acc_norm and quasi_exact_match
  - Build automated testing pipelines using the lighteval framework
  - Prompt engineering to maximize model performance on benchmarks
- **Notebook 6.2: Synthetic Evaluation Dataset Generation**
  - Generate custom evaluation datasets tailored to your specific domain using GPT-4.1-mini
  - Implement rubric-based scoring systems for automated quality assessment
  - Detect and mitigate biases in synthetic datasets through statistical analysis
  - Build robust data pipelines with Pydantic for structured output validation
- **Notebook 6.3: LLM as Judge**
  - Implement automated model comparison systems using AlpacaEval methodology
  - Diagnose and correct length bias in LLM evaluations using statistical techniques
  - Build position-invariant evaluation protocols to ensure fair comparisons
  - Extract and analyze logprobs to create length-controlled evaluation metrics

## Post-training and Fine-tuning Language Models
### Week 7: Fine-tuning and Adaptation Methods
Resource Required: Google Colab (T4) GPU or better (RunPod 1x A10G/L40s recommended)
- **Notebook 7.1: Build a Custom Text Classifier from LLMs**
  - Transform any pre-trained LLM into a high-accuracy spam detector
  - Implement strategic layer freezing to preserve knowledge while adapting
  - Design and attach custom classification heads to language models
  - Analyze model performance using confusion matrices and F1 scores
- **Notebook 7.2: Create Instruction-Following AI Assistants**
  - Convert raw language models into helpful chat assistants using the Alpaca dataset
  - Design effective prompt templates that shape model behavior
  - Implement production-ready training with gradient accumulation and mixed precision
  - Evaluate and improve response quality through systematic testing
- **Notebook 7.3: Optimize Models with LoRA and Preference Learning**
  - Deploy parameter-efficient fine-tuning that reduces memory by 90%
  - Execute two-stage training pipelines combining SFT and DPO
  - Implement Direct Preference Optimization to align models with human preferences
  - Merge and compare multiple model versions for optimal performance

## The Future of Language Models
### Week 8: Multimodality, Reinforcement Learning, and Agents
- **Notebook 8.1: Build Vision-Language Models from Scratch**
  - Implement a complete Vision Transformer (ViT) for image understanding
  - Build contrastive learning systems evolving from CLIP to SigLIP
  - Engineer multimodal fusion layers using cross-attention mechanisms
  - Construct a PaliGemma-style model that processes images and text together
- **Notebook 8.2: Design Autonomous AI Agents**
  - Architect an example agent system built with LLMs and tools
  - Build functional agents that can calculate math and navigate Wikipedia
  - Implement the ReAct pattern to enable reasoning before acting
  - Design memory systems that allow agents to learn from experience
- **Notebook 8.3: Apply Reinforcement Learning to Language Models**
  - Implement reward modeling for language generation tasks
  - Build intuition for RLHF and PPO through hands-on experiments
  - Introduce GRPO (Group Relative Policy Optimization) for efficient training