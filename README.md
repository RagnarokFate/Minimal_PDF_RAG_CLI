# Minimal PDF RAG CLI

A minimal Retrieval-Augmented Generation (RAG) CLI built with Python, LangChain, and Google Gemini.

## Quick Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env with your API key
echo "GOOGLE_API_KEY=your_key_here" > .env
```

## Tech Stack
- **Language**: Python 3.10+
- **LLM**: `gemini-flash-latest` (Fast, large context)
- **Embeddings**: `gemini-embedding-001`
- **Vector Store**: ChromaDB (Local persistence)
- **Framework**: LangChain (Retrieval & Generation logic)

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment:
   Create a `.env` file in the root directory and add your Google API Key:
   ```text
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

### 1. Ingest a PDF
To extract text and store it in the local vector database:
```bash
python main.py --ingest data/attention_paper.pdf
```

### 2. Ask Questions
To query the ingested document:
```bash
python main.py --ask "What is the formula for scaled dot-product attention?"
```

## Data
This project has been verified using the **"Attention Is All You Need"** research paper (stored in `data/attention_paper.pdf`), which introduces the Transformer architecture.

```bash
curl -o data/attention_paper.pdf https://arxiv.org/pdf/1706.03762.pdf  
```

## Technical Details
- **Embeddings**: `gemini-embedding-001`
- **LLM**: `gemini-flash-latest`
- **Vector Store**: ChromaDB (Local persistence in `./chroma_db`)

## Design Notes (Q1)

**Tools & Models Choice**: 
I chose `gemini-flash-latest` for generation due to its high speed and large context window. For embeddings, `gemini-embedding-001` provides a reliable, natively integrated foundation for Google’s ecosystem. Following the assignment requirements, `ChromaDB` was selected as the vector store for its local persistence and zero-config setup, directly supporting the project's goal.

**Weaknesses**: 
The current approach relies on basic similarity search without reranking or hybrid retrieval (BM25 + Dense). The single-file architecture, while fast to implement, limits scalability as more features are added.

**Future Improvements**: 
Future enhancements would focus on modularizing the project structure into distinct mini-services for PDF ingestion and vector embedding. I would also implement customized templates for responses and a "self-correction" feature to ensure high-quality and relevant outputs.

## AI Tools Used (Q2)

This project was developed with the assistance of several AI-driven tools:
- **Gemini CLI**: Used as an interactive coding agent for executing tasks and managing the development lifecycle.
- **ChatGPT**: Assisted in the initial planning and architectural conceptualization of the RAG system.
- **Antigravity IDE**: Provided the primary development environment for code authoring and integrated testing.

## Tests

### Question Example #1 - What is the formula for scaled dot-product attention?

```python main.py --ask "What is the formula for scaled dot-product attention?" ```

Answer: Based on the provided document, the formula for scaled dot-product attention is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Question Example #2 - What is the context of the project?

```python main.py --ask "What is the context of the project?" ```

Answer: Based on the provided document, the context of the project is the development of a new model architecture called the **Transformer** for sequence modeling and transduction tasks, such as machine translation. 

Key aspects of the project's context include:
* **Goal:** To reduce sequential computation and allow for significantly more parallelization during training, which is a major constraint in existing recurrent models.
* **Innovation:** The architecture eschews recurrence and instead relies entirely on an **attention mechanism** to draw global dependencies between input and output.
* **Comparison to Existing Work:** It builds on the goal of reducing sequential computation seen in models like ByteNet and ConvS2S (which use convolutional neural networks), but aims to improve the efficiency of relating signals from distant positions.
* **Performance:** The project demonstrates that this architecture can reach a new state of the art in translation quality with relatively short training times (twelve hours on eight P100 GPUs).
* **Structure:** The model uses an encoder-decoder structure where the encoder maps an input sequence to continuous representations and the auto-regressive decoder generates an output sequence one element at a time.

### Question Example #3 - What's the results of the attentions? and How transformers achieves better BLEU scores than previous models?

```python main.py --ask "What's the results of the attentions? and How transformers achieves better BLEU scores than previous models?" ```

**Answer:** Based on the provided document, here are the results regarding the attention mechanisms and how the Transformer achieves superior BLEU scores:

**Results of the Attentions**
According to the experiments detailed in the analysis of Table 3:
* **Number of Heads:** While single-head attention is 0.9 BLEU worse than the best setting, model quality also drops off if there are too many heads.
* **Key Size ($d_k$):** Reducing the attention key size hurts model quality. This suggests that determining compatibility is difficult and that the dot product may benefit from a more sophisticated compatibility function.
* **Functionality:** The model relies solely on attention mechanisms, dispensing with recurrence and convolutions entirely to determine dependencies.

**How Transformers Achieve Better BLEU Scores**
The Transformer achieves better BLEU scores (reaching 28.4 on English-to-German and 41.8 on English-to-French) through the following methods and architectural choices:
* **Architectural Efficiency:** By dispensing with recurrence and convolutions in favor of attention, the model is more parallelizable and requires significantly less time and cost to train (e.g., the big model was trained for only 3.5 days on eight GPUs).
* **Model Scaling:** The document notes that "bigger models are better," with the "Transformer (big)" model significantly outperforming the base model and previous state-of-the-art ensembles.
* **Regularization via Dropout:** Residual dropout is applied to the output of each sub-layer and to the sums of embeddings/positional encodings to avoid over-fitting.
* **Regularization via Label Smoothing:** During training, label smoothing ($\epsilon_{ls} = 0.1$) is used. While it hurts perplexity, it improves accuracy and the final BLEU score.
* **Optimization via Checkpoint Averaging:** The base model averaged the last 5 checkpoints, while the big model averaged the last 20 checkpoints.
* **Optimization via Beam Search:** The use of beam search with a beam size of 4 and a length penalty of 0.6 during inference.
* **Positional Encoding:** The model uses positional encodings to account for the order of the sequence since it lacks recurrence.
