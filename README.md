# Retrieval-Augmented Generation System for NLP Research Papers

A comprehensive RAG (Retrieval-Augmented Generation) pipeline specialized for Natural Language Processing research paper discovery and question-answering, built as part of CSE 575 Statistical Machine Learning at Arizona State University.

## 📋 Project Overview

This project implements a complete RAG system that enables users to ask natural language questions about NLP research and receive citation-backed answers derived from recent academic publications. The system combines semantic search with large language models to demonstrate how AI can support knowledge discovery in rapidly evolving research areas.

## 🎯 Key Features

- **Multi-Model Embedding Support**: SciBERT, MiniLM-L6, MPNet-Base, Instructor-Base
- **Hybrid Retrieval**: Combines dense (FAISS) and sparse (BM25) retrieval methods
- **Dual Generation Models**: Flan-T5-Base and TinyLlama-1.1B-Chat
- **Citation Tracking**: Automatic citation generation for retrieved papers
- **Comprehensive Evaluation**: Retrieval quality, generation metrics (ROUGE, BLEU), and scalability analysis
- **Advanced Visualizations**: Heatmaps, radar charts, performance comparisons

## 🔬 Research Questions

1. **RQ1**: How do domain-specific embeddings (SciBERT) compare to general-purpose models for NLP paper retrieval?
2. **RQ2**: What is the optimal weighting (α) for hybrid retrieval combining dense and sparse methods?
3. **RQ3**: How does retrieval quality impact generation quality across different LLMs?
4. **RQ4**: How does FAISS scale with increasing dataset sizes?

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or local environment with GPU
- Google Drive account (for storing data)

### Required Packages

```bash
pip install arxiv pymupdf sentence-transformers rank-bm25 faiss-cpu transformers torch rouge-score nltk accelerate matplotlib seaborn pandas numpy
```

## 📂 Project Structure

```
RAG_NLP_Project/
├── papers/
│   ├── pdfs/              # Downloaded arXiv papers
│   └── texts/             # Extracted text files
├── embeddings/
│   ├── scibert_embeddings.npy
│   ├── minilm_embeddings.npy
│   ├── mpnet_embeddings.npy
│   └── instructor_embeddings.npy
├── rag_results/
│   ├── rq1_embedding_comparison.csv
│   ├── rq2_hybrid_retrieval.csv
│   ├── rq3_generation_quality.csv
│   ├── rq4_scalability.csv
│   ├── generation_with_citations.csv
│   └── citation_analysis.csv
├── visualizations/
│   ├── heatmap_embedding_performance.png
│   ├── generation_quality_comparison.png
│   ├── hybrid_retrieval_performance.png
│   ├── scalability_analysis.png
│   ├── radar_chart_comparison.png
│   └── comprehensive_embedding_analysis.png
└── document_chunks.pkl
```

## 💻 Usage

### 1. Data Collection

```python
# Download 500 NLP papers from arXiv
query = 'cat:cs.CL OR cat:cs.AI OR ti:natural language processing OR ti:transformer'
papers_metadata = download_arxiv_papers(query, max_results=500)
```

### 2. Initialize Retrievers

```python
# Load pre-generated embeddings
scibert_embeddings = np.load('embeddings/scibert_embeddings.npy')

# Initialize FAISS index
scibert_retriever = DenseRetriever(scibert_embeddings, chunks)

# Initialize BM25
bm25_retriever = BM25Retriever(chunks)

# Create hybrid retriever
hybrid_retriever = HybridRetriever(scibert_retriever, bm25_retriever, alpha=0.5)
```

### 3. Query the System

```python
# Initialize citation-enhanced RAG
citation_rag = CitationRAG(
    retriever=hybrid_retriever,
    encoder=scibert_model,
    generator=flan_t5_generator,
    model_name="Flan-T5-Base"
)

# Ask a question
result = citation_rag.query_with_citations(
    "What are attention mechanisms in transformer models?",
    k=5
)

print(f"Answer: {result['answer']}")
print(f"Citations: {result['citations']}")
```

## 📊 Experimental Results

### RQ1: Embedding Model Comparison

| Model | Avg Retrieval Score | Avg Unique Papers |
|-------|---------------------|-------------------|
| SciBERT | **Highest** | **Best diversity** |
| MPNet | High | Good diversity |
| MiniLM | Moderate | Moderate diversity |
| Instructor | Lower | Lower diversity |

**Finding**: SciBERT (domain-specific) significantly outperforms general-purpose embeddings for NLP research papers.

### RQ2: Hybrid Retrieval Optimization

| Strategy | α | Avg Score | Diversity |
|----------|---|-----------|-----------|
| Dense | 1.0 | High | Moderate |
| **Hybrid** | **0.5** | **Highest** | **Highest** |
| BM25 | 0.0 | Lower | Lower |

**Finding**: Hybrid retrieval (α=0.5) provides optimal balance between semantic and keyword matching.

### RQ3: Generation Quality

| Model | ROUGE-1 | ROUGE-2 | BLEU | Avg Length |
|-------|---------|---------|------|------------|
| Flan-T5-Base | Moderate | Moderate | Moderate | Concise |
| **TinyLlama-1.1B** | **Higher** | **Higher** | **Higher** | **Detailed** |

**Finding**: TinyLlama produces more coherent and detailed responses despite being a smaller model.

### RQ4: FAISS Scalability

| Papers | Chunks | Indexing Time | Query Time |
|--------|--------|---------------|------------|
| 50 | ~2,000 | <10ms | <2ms |
| 100 | ~4,000 | <15ms | <3ms |
| 200 | ~8,000 | <25ms | <4ms |
| 350 | ~14,000 | <40ms | <5ms |
| 500 | ~20,000 | <55ms | <6ms |

**Finding**: FAISS demonstrates excellent scalability with sub-linear query time growth.

## 📈 Evaluation Metrics

- **Retrieval Quality**: Cosine similarity scores, unique paper diversity
- **Generation Quality**: ROUGE-1, ROUGE-2, ROUGE-L, BLEU
- **Citation Accuracy**: Papers cited per answer, citation relevance
- **Scalability**: Indexing time, query latency

## 🎨 Visualizations

The project includes 11 professional visualizations:
1. Embedding performance heatmap
2. Generation quality comparison
3. Generation quality heatmap (query-level)
4. Answer length comparison
5. Hybrid retrieval performance
6. Scalability analysis (dual plots)
7. Radar chart comparison
8. Comprehensive embedding analysis (4-plot grid)
9. Generation examples table
10. Summary statistics table

## 🔧 Key Components

### Dense Retriever (FAISS)
- Uses cosine similarity with normalized embeddings
- Efficient nearest-neighbor search
- Supports multiple embedding models

### BM25 Retriever
- Classic sparse retrieval based on term frequency
- Effective for keyword-based queries
- Complements dense retrieval

### Hybrid Retriever
- Weighted combination of dense and sparse scores
- Configurable α parameter (0=BM25, 1=Dense)
- Optimal at α=0.5

### Generation Models
- **Flan-T5-Base**: 250M parameters, encoder-decoder architecture
- **TinyLlama-1.1B**: 1.1B parameters, decoder-only architecture
- Both optimized for GPU memory constraints

## 📝 Dataset

- **Source**: arXiv (categories: cs.CL, cs.AI)
- **Size**: 500 papers
- **Topics**: Natural Language Processing, Transformers, Language Models
- **Chunking**: 512-token chunks with 50-token overlap
- **Total Chunks**: ~20,000

## ⚙️ Configuration

```python
# Model Configuration
EMBEDDING_MODEL = "allenai/scibert_scivocab_uncased"
GENERATION_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Retrieval Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
HYBRID_ALPHA = 0.5

# Generation Configuration
MAX_LENGTH = 256
TEMPERATURE = 0.7
NUM_BEAMS = 4
```

## 🚧 Known Limitations

1. **GPU Memory**: Large models (e.g., Llama-2-7B) exceeded Colab free tier memory
2. **Generation Quality**: Smaller models may produce less sophisticated responses
3. **Dataset Size**: Limited to 500 papers due to download and processing constraints
4. **Evaluation**: Manual citation accuracy checking required for full validation

## 🔮 Future Work

- Expand to 1000+ papers for improved coverage
- Implement re-ranking mechanisms
- Add conversational memory for multi-turn dialogues
- Deploy as web application with Gradio/Streamlit
- Integrate GPT-4 API for improved generation

## 📚 References

1. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
2. Beltagy, I., Lo, K., & Cohan, A. (2020). SciBERT: A pretrained language model for scientific text. EMNLP.
3. Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
4. Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR.
5. Izacard, G., & Grave, E. (2021). Leveraging passage retrieval with generative models for open-domain question answering. EACL.

## 📄 License

This project is developed for academic purposes as part of CSE 575 at Arizona State University.

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the me through the email addresses mentioned below.

## 📧 Contact

For inquiries about this project, please reach out to:
- Chandana Vinay Kumar: cvinayku@asu.edu
- Project Repository: http://github.com/Chandana0127/Retrieval-Augmented-Generation-for-NLP-Research-Papers

---
