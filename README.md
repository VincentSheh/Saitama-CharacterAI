# Saitama-CharacterAI

## 1. Project Overview and How to Run

This repository implements a **persona-driven conversational AI system** inspired by Character.AI, using **Saitama from *One Punch Man*** as the target character.  

The system combines **persona control**, **multi-tier memory**, and **retrieval-augmented generation (RAG)** to produce concise, in-character responses grounded in canonical knowledge. At the same time we implement Tool Usage in the form of News however, e acknowledge that **web search and news-style tool usage is not a natural fit for Saitama’s character**, as his personality emphasizes laziness and limited curiosity about external facts.

A more suitable character for extensive tool-based interactions would be **Deadpool**, have the tendency to break the fourth wall align better with external information retrieval and real-world commentary. Exploring such a character is left as **future work**.


### Repository Structure
```
.
├── chat_rag.py              # Main chat loop (persona + memory + RAG)
├── persona.md               # System prompt defining Saitama’s persona and rules
├── openrouter_client.py     # OpenRouter API wrapper for LLM calls
├── knowledge/               # Canon knowledge sources for RAG
│   ├── abilities.md
│   ├── relationship.md
│   └── synopsis.md
├── rag/
│   ├── build_kb.py          # Builds vector index from knowledge files
│   ├── rag_store.py         # FAISS-based RAG implementation
│   └── artifacts/           # Persisted RAG index
├── memory/
│   ├── ltm_summary.py       # Long-term summary-based memory storage
│   ├── summary_updater.py   # Updates long-term summaries using the LLM
│   └── artifacts/           # Persisted long-term memory
└── assets/                  # (Optional) demo assets or figures
```

### Key Components

**Persona (`persona.md`)**  
Defines Saitama’s speaking style, behavioral constraints, scope control, and inner monologue rules.  
This prompt enforces persona consistency across all responses.

**Retrieval-Augmented Generation (RAG)**  
Canonical One Punch Man knowledge is stored as markdown documents under `knowledge/`, obtained from the One Punch Man Wiki https://onepunchman.fandom.com/wiki/One-Punch_Man_Wiki.  
These documents are chunked, embedded, and indexed using FAISS to support similarity-based retrieval during inference.

**Short-Term Memory**  
Implemented as a rolling window of the last 15 dialogue turns.  
This enables conversational coherence while keeping context size bounded.

**Long-Term Memory**  
Implemented as a summary-based memory.  
The system maintains a persistent user summary and conversation summary, updated incrementally after each turn to retain durable context without memory explosion.

**LLM Backend**  
All generation and summarization are performed via **OpenRouter**, allowing flexible model selection.

### How to Run

#### 1. Install dependencies
```bash
pip install faiss-cpu sentence-transformers requests numpy
```
#### 2. Set OpenRouter API key
```bash
export OPENROUTER_API_KEY="your_api_key_here"
export TAVILY_API_KEY="your_api_key_here"
```
#### 3. Build the knowledge base (one-time)
```bash
python rag/build_kb.py
```
#### 4. Start the chat interface
```bash
python chat_rag.py
```

## 2. Implementation

### A. Memory

#### Short-Term Memory
Short-term memory is implemented as a **rolling window** using a fixed-length deque.  
The system retains the **last N=10 user–assistant turns**, which are injected into the prompt as recent conversational context.  
When the window exceeds its capacity, older turns are automatically discarded to control context length and inference cost.

#### Long-Term Memory
Long-term memory is implemented as a **summary-based memory mechanism**.  
Instead of storing individual memory items, the system maintains:
- a **user summary** capturing durable preferences, constraints, and goals
- a **conversation summary** capturing the high-level progression of the dialogue

These summaries are updated incrementally after each turn using the language model and are persisted to disk.  
This design avoids memory noise and uncontrolled growth while preserving relevant long-term context.


### B. Persona

Persona consistency is enforced through a **strict system prompt** that defines:
- speaking style (short, blunt, deadpan)
- behavioral constraints (no bragging, no therapy, no meta explanations)
- scope control for out-of-universe or unsupported questions

An **inner monologue mechanism** is incorporated by requiring a short parenthetical thought after each response, reinforcing character authenticity without affecting factual grounding.


### C. Retrieval-Augmented Generation (RAG)

RAG is used to ground responses in **One Punch Man canonical knowledge**.  
Markdown-based knowledge documents are chunked, embedded using a sentence-transformer model, and stored in a FAISS vector index.  
For each user query, the system retrieves the most relevant knowledge chunks and injects them into the prompt as contextual grounding.  
If no relevant context is found, the model is instructed to respond with uncertainty rather than hallucinate.


## 3. Demo Examples

### Canonical knowledge grounding
Demonstrates retrieval-augmented generation where responses are grounded in One Punch Man canon while preserving Saitama’s tone and behavior.
```
You: Who is Genos to you?

Saitama: Genos is my disciple. He's a cyborg and a good fighter. He wants to get stronger. (I guess he's kind of a friend too.)

You: If I become your disciple, I’ll give you Wagyu A5 every day

Saitama: I prefer to relax instead. No need for a disciple. (Wagyu A5 sounds nice, but naps are better.)
```

**Personality Expression:**  
Demonstrates persona enforcement through concise, indifferent phrasing and an inner monologue that reinforces Saitama’s character.
```
You: what is martial arts to you

Saitama: Martial arts is just a way to fight. It's about technique and discipline. I prefer simple punches. (All that technique seems complicated.)
```

**Memory recall:**  
Demonstrates both intra-conversation and inter-conversation memory by recalling a previously mentioned offer. (The Prompt below is tested on both cases)
```
You: what did I offered you previously

Saitama: You offered me wagyu A5 daily. (I still prefer my naps, though.)
```

**Out-of-scope handling:**  
Demonstrates strict scope control where the system refuses unrelated tasks while remaining in character.
```
You:  Can you write a C program for Hello World?

Saitama: Not sure. I don't do coding.
```

**Tool Usage: News:**  
Demonstrates optional tool usage for factual, out-of-universe queries when explicitly requested.
```
You: Where is Shueisha headquartered?

Saitama: Shueisha is headquartered in Tokyo, Japan. (That was a simple question.)

You: When is the release date for One Punch man season 3

Saitama: One Punch Man Season 3 will premiere on October 12, 2025. (Finally, some news.)
```

