# RAG Wiki Chatbot

Hệ thống chatbot QA (Question Answering) sử dụng Retrieval-Augmented Generation (RAG) với nguồn dữ liệu từ Wikipedia tiếng Việt.

## 🎯 Tổng quan

Dự án này xây dựng một hệ thống hỏi đáp thông minh dựa trên:
- **Retrieval**: Tìm kiếm thông tin liên quan từ Wikipedia
- **Augmented**: Bổ sung ngữ cảnh từ tài liệu tìm được
- **Generation**: Sinh câu trả lời bằng LLM (Large Language Model)

### ✨ Tính năng nổi bật

- ✅ **Multi-topic support**: Download và xây dựng knowledge base từ nhiều chủ đề
- ✅ **Hybrid search**: Kết hợp Dense (semantic) và BM25 (keyword) search
- ✅ **Reranking**: Sử dụng Cross-Encoder để cải thiện độ chính xác
- ✅ **Multi-LLM support**: OpenAI, Gemini, Anthropic
- ✅ **Source citation**: Trích dẫn nguồn Wikipedia với URL đầy đủ
- ✅ **Caching**: Tự động cache embeddings để tối ưu hiệu suất

## 📐 Kiến trúc

```
Query → Embedding → Vector Search → Reranking → LLM Generation → Answer + Sources
                     ↓
                  Wikipedia
                  Knowledge Base
```

## 📁 Cấu trúc thư mục

```
rag_wiki/
├── README.md                    # File này
├── PHAN_TICH_COMPONENT.md       # Phân tích chi tiết từng component
├── requirements.txt             # Các thư viện cần thiết
├── run_pipeline.py              # Chạy toàn bộ pipeline
├── .env                         # API keys 
│
├── data/                        # Dữ liệu
│   ├── raw_wiki/               # Wikipedia dump gốc
│   ├── parsed/                 # Văn bản đã parse
│   ├── chunks/                 # Chunks + embeddings
│   ├── faiss_index/            # Vector database
│   └── .embedding_cache/       # Cache embeddings
│
├── scripts/                     # ETL Scripts
│   ├── download_wiki.py        # Tải Wikipedia
│   ├── parse_wikitext.py       # Parse wikitext
│   └── chunk_text.py           # Chia thành chunks
│
├── src/                        # Source code chính
│   ├── config.yaml             # Cấu hình hệ thống
│   ├── qa_pipeline.py          # Pipeline chính
│   ├── indexer/                # Embedding & indexing
│   │   ├── embedder.py
│   │   └── indexer.py
│   ├── retriever/              # Retrieval & reranking
│   │   ├── retriever.py
│   │   └── reranker.py
│   └── generator/              # LLM generation
│       ├── generator.py
│       └── prompt_templates.md
│
└── notebooks/                   # Jupyter notebooks demo
```

## 🚀 Quickstart

### 1. Yêu cầu hệ thống

- Python 3.8+
- 8GB RAM (tối thiểu)
- 10GB dung lượng ổ đĩa

### 2. Cài đặt

```bash
# Clone repo
git clone https://github.com/BTL4w/rag-wiki-chatbot.git
cd rag_wiki

# Cài đặt thư viện
pip install -r requirements.txt
```

### 3. Cấu hình API Keys

Tạo file `.env` trong thư mục gốc:

```env
# Chọn một trong các LLM providers
OPENAI_API_KEY=your_openai_key_here
# hoặc
GEMINI_API_KEY=your_gemini_key_here
# hoặc
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 4. Download & Build Knowledge Base

**Option 1: Chạy toàn bộ pipeline (Khuyến nghị)**

```bash
# Download 50 bài về AI và chạy toàn bộ pipeline
python run_pipeline.py --topic "artificial intelligence machine learning" --limit 50
```

**Option 2: Chạy từng bước**

```bash
# Bước 1: Download Wikipedia
python scripts/download_wiki.py --topic "artificial intelligence" --limit 50

# Bước 2: Parse văn bản
python scripts/parse_wikitext.py

# Bước 3: Chia thành chunks
python scripts/chunk_text.py

# Bước 4: Tạo embeddings
python -m src.indexer.embedder --input data/chunks/chunks.jsonl --output data/chunks/embeddings.npy

# Bước 5: Build vector index
python -m src.indexer.indexer --embeddings data/chunks/embeddings.npy --chunks data/chunks/chunks.jsonl
```

### 5. Chạy Chatbot

**Mode Interactive**:

```bash
python -m src.qa_pipeline
```

**Mode Single Question**:

```bash
python -m src.qa_pipeline --question "Học máy là gì?"
```

**Output mẫu**:

```
💬 Bạn: Học máy là gì?

🤖 Bot: Học máy hay máy học (machine learning) là một lĩnh vực của trí tuệ nhân tạo 
liên quan đến việc nghiên cứu và xây dựng các kĩ thuật cho phép các hệ thống "học" 
tự động từ dữ liệu để giải quyết những vấn đề cụ thể (Nguồn: [1]).

📚 Nguồn (5):
  [1] Học máy - Introduction
      🔗 https://vi.wikipedia.org/wiki/Học_máy
  [2] Trí tuệ nhân tạo - Học hỏi
      🔗 https://vi.wikipedia.org/wiki/Trí_tuệ_nhân_tạo
  [3] Học máy - Tính phổ quát
      🔗 https://vi.wikipedia.org/wiki/Học_máy

⏱️ Thời gian: 1.52s
```

## 📖 Sử dụng trong Code Python

```python
from src import QAPipeline

# Khởi tạo pipeline
pipeline = QAPipeline("src/config.yaml")
pipeline.load_components()

# Đặt câu hỏi
response = pipeline.query("Thủ đô của Việt Nam là gì?")

print(response['answer'])
print(response['sources'])
print(f"Time: {response['total_time']:.2f}s")
```

## ⚙️ Cấu hình

### Chọn Embedding Model

```yaml
# src/config.yaml
embedding:
  provider: "sentence-transformers"
  model_name: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  dimension: 768
  batch_size: 32
```

**Models khuyên dùng cho tiếng Việt**:
- `paraphrase-multilingual-mpnet-base-v2` (768 dim, khuyến nghị)
- `LaBSE` (768 dim, multilingual)
- OpenAI `text-embedding-3-small` (1536 dim, API)

### Chọn LLM Provider

```yaml
# src/config.yaml
generation:
  provider: "openai"  # openai, gemini, anthropic
  model_name: "gpt-4o-mini"
  temperature: 0.1
  max_completion_tokens: 500
```

### Tối ưu Retrieval

```yaml
# src/config.yaml
retrieval:
  top_k: 50              # Số chunks ban đầu
  final_k: 5             # Số chunks sau rerank
  use_reranker: false    # Bật/tắt reranking
  hybrid_search: false   # Bật BM25 + dense search
```

## 🎯 Use Cases

### Use Case 1: Multi-Topic Knowledge Base

```bash
# Download nhiều topics
python scripts/download_wiki.py --topic "artificial intelligence" --limit 50
python scripts/download_wiki.py --topic "Việt Nam" --limit 50
python scripts/download_wiki.py --topic "lịch sử" --limit 30

# Process tất cả
python scripts/parse_wikitext.py
python scripts/chunk_text.py
python -m src.indexer.embedder --input data/chunks/chunks.jsonl --output data/chunks/embeddings.npy
python -m src.indexer.indexer --embeddings data/chunks/embeddings.npy --chunks data/chunks/chunks.jsonl

# Chatbot có thể trả lời cả 3 topics!
python -m src.qa_pipeline
```

### Use Case 2: Domain-Specific Chatbot

```bash
# Chỉ download về Y tế
python run_pipeline.py --topic "y tế sức khỏe" --limit 100

# Chatbot chuyên về Y tế
python -m src.qa_pipeline
```

### Use Case 3: Custom Wikipedia Download

```bash
# Download theo category
python scripts/download_wiki.py --category "Machine learning" --limit 50

# Download specific pages (tạo file pages.txt)
python scripts/download_wiki.py --pages-file pages.txt
```

## 🛠️ Scripts Tiện Ích

### Clean Data

```bash
# Chỉ xóa dữ liệu
clean_data.bat

# Xóa + chạy lại pipeline
clean_and_retry.bat
```

### Debug Mode

```bash
# Chạy với debug info
python -m src.qa_pipeline --question "..." --debug
```

Output:
```
🔍 Debug:
  Retrieved: 50 chunks
  Reranked: 5 chunks
  Retrieval: 0.050s
  Rerank: 0.200s
  Generation: 1.500s
```



## 🔄 Workflow Tổng Quát

```
┌─────────────────────────────────────────────────────────┐
│                  PHASE 1: SETUP (1 lần)                 │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   Download           Parse             Chunk
   Wikipedia        Wikitext            Text
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   Create            Build Vector      Save to
   Embeddings          Index            Disk
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              PHASE 2: QUERY (mỗi câu hỏi)              │
└─────────────────────────────────────────────────────────┘
                          │
        User Question     │
              │           │
              ▼           ▼
        Embed Query → Vector Search
              │           │
              └─────┬─────┘
                    │
                    ▼
              Reranking (optional)
                    │
                    ▼
            Build Context + Prompt
                    │
                    ▼
              LLM Generate
                    │
                    ▼
         Answer + Sources + URLs
```



## 🎓 Học thêm

### Concepts chính

- **RAG**: Retrieval-Augmented Generation
- **Embedding**: Vector representation of text
- **FAISS**: Vector database for similarity search
- **Cross-Encoder**: Reranking model
- **BM25**: Keyword-based ranking algorithm

### Architecture Patterns

- **Pipeline Pattern**: Data flows through stages
- **Caching Pattern**: Cache expensive operations
- **Multi-provider Pattern**: Swap components easily

