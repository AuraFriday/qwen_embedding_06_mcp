# Qwen Local Embeddings — 1024-Dimensional Vector Generation

An MCP server for generating Qwen3 embedding vectors

> **Local model. No API calls. Automatic caching.** Generate embeddings using Qwen3-Embedding-0.6B without internet dependency.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/AuraFriday/mcp-link-server)

---

## Benefits

### 1. 🔒 Complete Privacy & Offline Operation
**Not cloud API — local inference.** Your text never leaves your machine. No API keys required. No internet needed after initial model download. **Complete data privacy.**

### 2. ⚡ Automatic Caching & Speed
**Not repeated computation — intelligent caching.** Every embedding is cached locally in SQLite. Identical text? Instant retrieval. No re-computation. Thread-safe concurrent access.

### 3. 🌍 Multilingual Excellence
**Not English-only — 100+ languages.** State-of-the-art multilingual embeddings from Alibaba's Qwen team. Chinese, Spanish, Arabic, Hindi, and 96+ more languages. Same quality across all.

---

## Why This Tool Matters

**Embeddings are fundamental to modern AI.** Semantic search, RAG systems, similarity matching, clustering, classification — all require embeddings. But most embedding APIs have problems:

**Privacy concerns:** Your data goes to third-party servers.  
**Cost:** API calls add up fast for large datasets.  
**Latency:** Network round-trips slow everything down.  
**Dependency:** Internet required. API downtime breaks your app.

**This tool solves all of that.** Local inference. No API calls. No internet dependency. Automatic caching makes repeated queries instant. And it's **free** — no per-request costs.

---

## Real-World Disaster: When Google Killed Our Legal System

**This tool exists because cloud embedding providers will destroy your work.**

### The Nightmare

We built a complex legal RAG system. Months of work. Hundreds of millions of embeddings. Critical legal research infrastructure.

**We chose Google Gemini embeddings.** Seemed safe. Big company. Reliable service. Right?

**Wrong.**

### The Hidden Horrors

**What Google doesn't tell you on their website:**

1. **Undocumented Rate Limits**: Hit them constantly. No warning. No documentation. Just failures.

2. **No Opt-Out**: Want to pay more for higher limits? Too bad. Need guaranteed throughput for production? Tough luck. Rate limits are mandatory and non-negotiable.

3. **Zero Support**: Support requests went unanswered. Weeks of silence. We were on our own.

4. **Complex Recovery Required**: Spent over a week building exponential backoff, retry logic, lost embedding recovery systems. Just to work around their undocumented limitations.

5. **Data Sovereignty Nightmare**: Weeks spent navigating legal requirements for using cloud embeddings with sensitive legal data. Privacy policies. Data retention. Compliance documentation.

### Then They Killed It

**Six months later, Google shut down the service.**

Hundreds of millions of embeddings. Months of work. Critical legal infrastructure. **Gone.**

Not deprecated. Not sunset with migration path. **Shut down.**

**Every cloud provider will do this to you eventually.** Google, OpenAI, Anthropic — doesn't matter. When the service isn't profitable enough, they kill it. Your work disappears.

### The Solution

**Qwen3-Embedding-0.6B running locally.**

- **Equal quality** to Google's expensive embeddings
- **Never shuts down** (runs on your hardware)
- **Zero rate limits** (it's your computer)
- **Complete privacy** (data never leaves your machine)
- **No support tickets** (it just works)
- **Free forever** (no API costs)
- **Optimized for modern hardware** (runs well on wide range of systems)

**This is the model our SQLite tool uses** for its built-in embedding generation. Battle-tested on hundreds of millions of embeddings. Reliable. Dependable. Yours.

**Learn from our pain.** Don't build on cloud embeddings. They will betray you.

---

## Real-World Story: Building a Private Knowledge Base

**The Scenario:** Developer building a personal knowledge management system with 50,000 documents. Needs semantic search. Tried OpenAI embeddings.

**The Problem:**
- **Cost:** 50,000 docs × $0.0001/doc = $5 initial cost, plus ongoing costs for new docs
- **Privacy:** All document content sent to OpenAI servers
- **Latency:** 50,000 API calls took 6 hours
- **Dependency:** Internet required for every query

**With Qwen Local Embeddings:**

```python
# Generate embeddings for 50,000 documents
for doc in documents:
    embedding = qwen_embedding_0_6b.generate(
        text=doc.content,
        tool_unlock_token="<token>"
    )
    store_in_vector_db(doc.id, embedding)

# Result:
# - Cost: $0 (completely free)
# - Privacy: All data stays local
# - Speed: After caching, re-indexing takes seconds instead of hours
# - Offline: Works without internet after initial model download
```

**The result:** Developer built a completely private, offline-capable knowledge base with zero ongoing costs. Re-indexing after updates? Instant, thanks to caching.

---

## The Complete Feature Set

### Local Model Inference

**No API Calls:**
- Model: Qwen/Qwen3-Embedding-0.6B (596M parameters)
- Dimensions: 1024 (supports user-defined 32-1024)
- Context Length: Up to 32K tokens
- Languages: 100+ languages supported

**Why local matters:** Complete privacy. No API costs. No internet dependency. No rate limits.

### Automatic Caching System

**Intelligent Cache:**
```python
# First call: Generates embedding (takes ~100ms)
embedding1 = generate(text="Machine learning is amazing")

# Second call with same text: Cache hit (takes ~1ms)
embedding2 = generate(text="Machine learning is amazing")

# Result: 100x faster on cache hits
```

**Cache Features:**
- SQLite-based persistent storage
- Thread-safe concurrent access
- Exact text matching for cache hits
- WAL mode for better concurrency
- Automatic cache management

**Why caching matters:** Re-processing identical text is wasteful. Cache makes repeated queries instant.

### Automatic Dependency Management

**Zero Configuration:**
- First run: Auto-downloads model (~600MB)
- Auto-installs `sentence-transformers>=2.7.0`
- Auto-installs `transformers>=4.51.0`
- No manual setup required

**Why auto-install matters:** Users don't need to understand Python dependencies. It just works.

### Multilingual Support

**100+ Languages:**
- English, Chinese, Spanish, Arabic, Hindi, French, German, Japanese, Korean, Portuguese, Russian, Italian, Turkish, Vietnamese, Thai, Indonesian, Polish, Dutch, Romanian, Greek, Czech, Swedish, Hungarian, Hebrew, Finnish, Norwegian, Danish, Bulgarian, Slovak, Lithuanian, Slovenian, Croatian, Serbian, Ukrainian, Estonian, Latvian, Icelandic, Irish, Maltese, Welsh, Basque, Galician, Catalan, and 60+ more.

**Why multilingual matters:** Global applications need global language support. Qwen delivers state-of-the-art performance across all languages.

---

## Usage Examples

### Basic Embedding Generation

```json
{
  "input": {
    "operation": "generate",
    "text": "The quick brown fox jumps over the lazy dog",
    "tool_unlock_token": "<your_token>"
  }
}
```

**Returns:**
```json
[0.0234, -0.0567, 0.0891, ..., 0.0123]
```
(1024 floating-point numbers)

### Multilingual Example

```json
{
  "input": {
    "operation": "generate",
    "text": "机器学习是人工智能的一个子集",
    "tool_unlock_token": "<your_token>"
  }
}
```

**Works perfectly** — Same quality for Chinese as English.

### Integration with SQLite Vector Search

```python
# Generate embedding
embedding = qwen_embedding_0_6b.generate(
    text="Find documents about machine learning",
    tool_unlock_token="<token>"
)

# Use with sqlite vector search
results = sqlite.execute(
    sql="""
        SELECT title, content, 
               vec_distance_cosine(embedding, :query_embedding) AS similarity
        FROM documents
        ORDER BY similarity
        LIMIT 10
    """,
    bindings={"query_embedding": embedding},
    database="knowledge.db",
    tool_unlock_token="<token>"
)
```

**Why this matters:** Qwen embeddings integrate seamlessly with the `sqlite` tool's vector search capabilities for powerful semantic search.

---

## Technical Architecture

### Model Details
- **Architecture:** Qwen3-Embedding-0.6B
- **Parameters:** 596 million
- **Output Dimensions:** 1024 (configurable 32-1024)
- **Context Length:** 32K tokens
- **Training:** Multilingual training on 100+ languages

### Caching Strategy
- **Storage:** SQLite database with WAL mode
- **Key:** Exact text match (primary key)
- **Thread Safety:** Concurrent read/write support
- **Location:** User data directory (cross-platform)
- **Persistence:** Survives server restarts

### Dependency Management
- **Auto-Install:** First run installs dependencies
- **Versions:** `sentence-transformers>=2.7.0`, `transformers>=4.51.0`
- **Model Download:** Automatic via HuggingFace Hub
- **Storage:** HuggingFace cache directory

---

## Limitations & Considerations

### First Run
- **Model Download:** ~600MB download on first use
- **Time:** 2-5 minutes depending on connection speed
- **Storage:** Model stored in HuggingFace cache (~1GB total)

### Performance
- **Cold Start:** ~100ms per embedding (first time)
- **Cache Hit:** ~1ms per embedding (cached)
- **Batch Processing:** Not currently optimized for batching

### Hardware
- **CPU:** Works on any CPU (no GPU required)
- **GPU:** Will use GPU if available (faster)
- **Memory:** ~2GB RAM for model

### Comparison to Gemini
- **Dimensions:** 1024 (Qwen) vs 3072 (Gemini)
- **Privacy:** Complete (Qwen) vs API call (Gemini)
- **Cost:** Free (Qwen) vs API costs (Gemini)
- **Speed:** Faster with cache (Qwen) vs network latency (Gemini)
- **Quality:** Excellent (Qwen) vs Slightly better (Gemini)

**When to use Qwen:** Privacy matters, offline operation needed, cost is a concern, or multilingual support required.  
**When to use Gemini:** Need highest possible quality and have API key available.

---

## Why This Tool is Unmatched

**1. Complete Privacy**  
Your data never leaves your machine. No API calls. No third-party servers.

**2. Zero Cost**  
No API fees. No per-request charges. Completely free after initial download.

**3. Offline Operation**  
Works without internet after initial model download. Perfect for air-gapped environments.

**4. Automatic Caching**  
Intelligent cache makes repeated queries instant. 100x faster on cache hits.

**5. Multilingual Excellence**  
State-of-the-art performance across 100+ languages. Not English-only.

**6. Zero Configuration**  
Auto-downloads model. Auto-installs dependencies. Just works.

**7. Thread-Safe**  
Concurrent access supported. Multiple requests handled safely.

**8. SQLite Integration**  
Works seamlessly with the `sqlite` tool's vector search for powerful semantic search.

**9. Cross-Platform**  
Works on Windows, macOS, Linux. Same code everywhere.

**10. Open Source Model**  
Based on Alibaba's open-source Qwen3 model. Transparent and auditable.

---

## Powered by MCP-Link

This tool is part of the [MCP-Link Server](https://github.com/AuraFriday/mcp-link-server).

### Get MCP-Link

Download: [GitHub Releases](https://github.com/AuraFriday/mcp-link-server/releases/latest)

---

## License & Copyright

Copyright © 2025 Christopher Nathan Drake

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

AI Training Permission: You are permitted to use this software and any
associated content for the training, evaluation, fine-tuning, or improvement
of artificial intelligence systems, including commercial models.

SPDX-License-Identifier: Apache-2.0

Part of the Aura Friday MCP-Link Server project.

