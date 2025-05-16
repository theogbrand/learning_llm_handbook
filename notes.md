# Course Structure

Setting Up Jupyter Notebook with SLURM and uv: https://adam-streck.medium.com/creating-persistent-jupyter-notebooks-on-a-cluster-using-vs-code-slurm-and-conda-140b922a97a8

uv venv --python 3.12 --seed
uv pip install notebook
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install vllm
jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888

## Estimated course duration: 2 months
* Estimated time commitment: 320 hours. (8 hours per day * 5 days a week * 4 weeks per month * 2 months)

## Prerequisites

This course is designed for professionals with a solid background in data science basics, experience building data pipelines,
and familiarity with analytics and visualization. You should be comfortable with distributed computing frameworks like Spark,
have worked with basic machine learning models, and understand text processing fundamentals. We expect you're already using
version control and have collaborated on software projects. Some exposure to university-level mathematical concepts (e.g. differential equations, statistics, linear algebra) will help you grasp the concepts behind training and deploying language models at scale.


## Course Outline

1. Data science fundamentals refresher (optional, recommended)

2. LLM fundamentals
* General understanding of neural networks
    * Building Micrograd (understanding basic back propagation and autograd)
    * MNIST by hand, with and without PyTorch
    * Introduction to language modelling 
        * Kaparthy's n-gram makemore tutorial
        * Understanding Backpropagation (Partial derivatives, Chain rule, autograd)
        * Go through example by hand at least once and get the general idea of it
    * Understanding vanilla Transformer architecture (Attention is all you need paper)
        * Understanding transformer blocks (see illustrated transformer example)
            * f(Q,K,V) = softmax(QK^T/√d_k)V
        * KV-cache
        * Some light introduction to other architectures (encoder-only, decoder-only) and rationale to how and why decoder-only is current dominant architecture
    * Tokenizers
        * Kaparthy's video
        * Tiktoken, BPE, SentencePiece (older)
        * Understanding fertility rates
    * What are Embeddings?
    * Gentle introduction to multimodality
        * Pipeline approaches
        * fusion (early, embedded)

* Introduction to training large language models
    * Pretraining
        * train neural net with FineWeb, Karpathy video
    * Posttraining
        * Finetuning
        * RL (probably leave out)
    * Evaluation
        * Understanding nuance of every leaderboard
        * Evaluation types
        * Critique dataset curation strategies
    * Inference and deployment
        * What is vLLM
        * Picking and deploying a suitable GPU

Resources
* LLM under the hood crash course by William and Leslie: [GitHub Link](https://github.com/aisingapore/learning-buildgpt2)