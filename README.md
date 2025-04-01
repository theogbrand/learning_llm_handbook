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

## Training and Fine-Tuning Language Models
### Week 2: Transformer Architecture Deep Dive
- **Notebook 2.1: Attention Mechanism**
  - Code attention mechanism from scratch (Q, K, V matrices)
  - Implement scaled dot-product attention with masking
- **Notebook 2.2: Simple Transformer Implementation**
  - Build a mini transformer model in PyTorch
  - Train the model on a small text dataset
  - Implement n-gram character-level language modeling
  - Trace backpropagation steps manually and verify against PyTorch autograd
- **Notebook 2.3: Backpropagation Implementation**
  - Build a tiny autograd engine
  - Implement and train a simple neural network using micrograd

### Week 3: Advanced Transformer Components
- **Notebook 3.1: Transformer Block Implementation**
  - Code multi-head attention from scratch
  - Implement positional embeddings and encodings
  - Build feed-forward networks with different activation functions
- **Notebook 3.2: KV-Cache Optimization**
  - Implement and benchmark KV-cache for efficient inference
  - Visualize memory usage with and without KV-cache
- **Notebook 3.3: Architecture Comparisons**
  - Implement simplified versions of encoder-only (BERT-like) models
  - Compare performance of decoder-only (GPT-like) architectures
  - Experiment with encoder-decoder (T5-like) designs on translation tasks (E.g. NLLB)

### Week 4: Data Pipeline Engineering
Resource Required: 8xA100 GPU to train on FineWeb (Karpathy example)
- **Notebook 4.1: Data Pipeline Development**
  - Build data preprocessing pipelines for text datasets (OCR - give already OCR-ed data)
  - Implement efficient data loading with PyTorch DataLoader
  - Create data cleaning and normalization functions (Down/Upsampling, REGEX)
- **Notebook 4.2: Training Infrastructure**
  - Experiment with FineWeb dataset samples (Karpathy)
  - Set up distributed training environments
  - Implement mixed precision training
  - Experiment with gradient accumulation and large batch training
- **Notebook 4.3: Pre-training Methodologies**
  - Implement masked language modeling objectives
  - Build causal language modeling training loop

### Week 5: Fine-tuning and Adaptation Methods
Resource Required: Google Colab (T4) GPU or better (RunPod 1x A10G/L40s recommended)
- **Notebook 5.1: Full Fine-tuning**
  - Implement full fine-tuning on domain-specific datasets
  - Compare performance before and after fine-tuning
- **Notebook 5.2: Parameter-Efficient Methods**
  - Implement LoRA and QLoRA from scratch
  - Build adapter modules and prompt tuning mechanisms
  - Compare parameter counts and performance across methods
- **Notebook 5.3: Domain Adaptation**
  - Create instruction tuning datasets and training loops
  - Implement simplified RLHF pipeline components

## Evaluating Language Models
### Week 6: Inference Optimization and Deployment
Resource Required: Google Colab (T4) GPU or better (RunPod 1x A10G/L40s recommended)
- **Notebook 6.1: Inference Engines**
  - Benchmark vLLM against standard HF/PyTorch inference 
  - Implement various inference optimizations
- **Notebook 6.2: Quantization Techniques**
  - Apply INT8 and INT4 quantization to models (gguf lib?)
  - Analyse accuracy-performance tradeoffs
- **Notebook 6.3: Production Optimizations**
  - Implement KV-cache management strategies for production
  - Benchmark different GPU configurations and batch sizes

### Week 7: Evaluation
Resource Required: Google Colab (T4) GPU or better (RunPod 1x A10G/L40s recommended)
- **Notebook 7.1: Benchmark Development**
  - Create custom evaluation benchmarks for specific domains
  - Implement automated evaluation for math and coding tasks
- **Notebook 7.2: Evaluation Methods**
  - Compare reference-free vs. reference-based evaluation
  - Implement human-in-the-loop evaluation interfaces
  - Create visualizations for model performance analysis
- **Notebook 7.3: Lite-Eval Implementation**
  - Build lightweight evaluation pipelines for rapid testing
  - Analyze correlation between lite-eval and comprehensive benchmarks

## The future of Language Models
### Week 8: Multimodality, RL, RAG Systems
Resource Required: API keys for Pipecat (TBD)
- **Notebook 8.1: Multimodal Integration**
  - Introduction to early and late fusion techniques
  - Build simple vision-language model components (PaliGemma walkthrough)
  - Pipecat pipeline example
- **Notebook 8.2: Introduction to RL**
  - Create reward modeling components
  - Implement S1 (R1) loop
- **Notebook 8.3: RAG System Development**
  - Build retrieval-augmented generation pipelines
  - Implement vector stores and efficient retrieval mechanisms
  - Evaluate RAG systems against fine-tuned models
