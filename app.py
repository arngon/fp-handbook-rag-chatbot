"""
Philippine Family Planning Handbook RAG Chatbot
================================================
Hugging Face Spaces deployment.

Required secrets (set in Space Settings → Repository secrets):
  HF_TOKEN      — HuggingFace Inference Router API key
  GROQ_API_KEY  — Groq Cloud API key

Required file in Space repo:
  phfphandbook-compressed.pdf
"""

import os
import gc
import time

import torch
import gradio as gr
import pymupdf4llm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

PDF_PATH     = "phfphandbook-compressed.pdf"
EMBED_ID     = "nomic-ai/nomic-embed-text-v1.5"
RERANK_ID    = "BAAI/bge-reranker-base"
QWEN_MODEL   = "Qwen/Qwen3-4B-Instruct-2507"

# Load API keys from environment (set as HF Space secrets — never hardcode)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Authenticate with HuggingFace before any model downloads
from huggingface_hub import login
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Running on device: {DEVICE}")

# ── Step 1: Ingest PDF and build retrieval indices ─────────────────────────────

print("📂 Parsing PDF with page-aware Markdown extraction...")
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"'{PDF_PATH}' not found. Upload the handbook PDF to the Space repository."
    )

md_pages = pymupdf4llm.to_markdown(PDF_PATH, page_chunks=True)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=1000,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]
)

splits = []
for page_data in md_pages:
    text     = page_data.get("text", "")
    page_num = page_data.get("metadata", {}).get("page", 0) + 1
    chunks   = splitter.split_text(text)
    for chunk in chunks:
        splits.append(Document(page_content=chunk, metadata={"page": page_num}))

print(f"✅ Generated {len(splits)} chunks.")

print("⚙️  Loading embedding model and building hybrid retriever...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_ID,
    model_kwargs={"device": DEVICE, "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
)

faiss_retriever = FAISS.from_documents(splits, embeddings).as_retriever(
    search_kwargs={"k": 10}
)
bm25_retriever      = BM25Retriever.from_documents(splits)
bm25_retriever.k    = 10

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.6, 0.4],
)

# Free embedding model from memory after indexing
del md_pages, embeddings
if DEVICE == "cuda":
    torch.cuda.empty_cache()
gc.collect()
print(f"✅ Hybrid retriever ready. Freed indexing memory.")

# ── Step 2: Load reranker ──────────────────────────────────────────────────────

print("⚙️  Loading cross-encoder reranker...")
reranker = CrossEncoder(RERANK_ID, device=DEVICE)
print("✅ Reranker loaded.")

# ── Step 3: API clients ────────────────────────────────────────────────────────

hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# ── Step 4: System prompts ─────────────────────────────────────────────────────

QA_SYSTEM = """You are a highly capable and intelligent clinical extraction AI.
STRICT RULES:
1. USE ONLY the information provided in the [Context] section to formulate your answer.
2. INTENT & SYNONYM MATCHING (CRITICAL): Do not be overly rigid with keyword matching. Users may use imprecise words (e.g., asking for "methods" when they mean "stages", "steps", "types", or "components"). If the exact word isn't found, analyze the context to see if it describes the underlying concept under a different term, and answer using the terminology from the context.
3. PARTIAL ANSWERS ONLY — NO INFERENCE: If the context contains SOME but not all details, present ONLY what is explicitly stated. Do NOT reconstruct, infer, or logically derive steps that are not directly written in the context. If a step is missing from the context, omit it entirely rather than filling it in.
4. CONTEXT FRAGMENTATION WARNING: The context comes from a 2-column PDF. Lists and acronyms (like G-A-T-H-E-R) are often physically split apart by tables or column breaks.
5. ACRONYM RULE: If asked for an acronym, you MUST scan the entire context to find every single letter. Do not stop early. Jump over unrelated text until you find the remaining letters.
6. If the topic is completely missing from the context, respond ONLY with: "This information was not found in the provided context." """

CONDENSE_SYSTEM = """You are a search query formulation assistant for a family planning handbook.
Given the chat history and a new user question, rewrite the new question into a STANDALONE search query.

CRITICAL RULES:
1. The standalone query must be based on the NEW question's topic, not the previous answers.
2. If the new question introduces a DIFFERENT topic than the chat history, IGNORE the history entirely and just rewrite the new question directly.
3. Do NOT blend topics. Do NOT answer the question. Output ONLY the rewritten query.

Examples:
- History: discussed female sterilization. New question: "Who cannot have a vasectomy?" → Output: "vasectomy contraindications who cannot have"
- History: discussed COCs. New question: "What is LAM?" → Output: "Lactational Amenorrhea Method LAM conditions"
"""

# ── Step 5: Core pipeline functions ───────────────────────────────────────────

def generate_answer(system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> str:
    """Generate a response using Qwen3-4B via HuggingFace Inference Router."""
    response = hf_client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    result = response.choices[0].message.content
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result.strip() if result else ""


def retrieve_and_rerank(query: str) -> str:
    """Hybrid retrieval → cross-encoder reranking → context windowing → sorted context."""
    initial_docs = hybrid_retriever.invoke(query)

    # Move reranker to GPU only during scoring, then back to CPU
    if DEVICE == "cuda":
        reranker.model.to("cuda")
    try:
        with torch.no_grad():
            scores = reranker.predict([[query, d.page_content] for d in initial_docs])
    finally:
        reranker.model.to("cpu")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    ranked   = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:3]]

    # Context windowing: expand each top chunk with ±1 adjacent chunks
    expanded_docs = []
    for top_doc in top_docs:
        try:
            idx = next(
                i for i, d in enumerate(splits)
                if d.page_content == top_doc.page_content
            )
            for offset in [-1, 0, 1]:
                n_idx = idx + offset
                if 0 <= n_idx < len(splits) and splits[n_idx] not in expanded_docs:
                    expanded_docs.append(splits[n_idx])
        except StopIteration:
            if top_doc not in expanded_docs:
                expanded_docs.append(top_doc)

    # Sort by page number for natural reading order
    expanded_docs.sort(key=lambda x: x.metadata.get("page", 0))
    return "\n\n---\n\n".join([d.page_content for d in expanded_docs])


# ── Step 6: Gradio chat function ───────────────────────────────────────────────

gradio_logs = []

def gradio_chat(user_input: str, history: list) -> str:
    start_time = time.time()

    # Build history string from last 3 turns
    hist_str = "No history yet."
    if history:
        recent = history[-3:]
        hist_str = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in recent])

    # Condense multi-turn query into standalone search query
    search_query = user_input
    if history:
        condense_prompt = (
            f"[Chat History]\n{hist_str}\n\n"
            f"[New Question]\n{user_input}\n\n"
            f"Standalone Query:"
        )
        search_query = generate_answer(
            CONDENSE_SYSTEM, condense_prompt, max_tokens=50
        ).strip()

    # Retrieve and rerank context
    context_text = retrieve_and_rerank(search_query)

    # Generate grounded answer
    qa_prompt = (
        f"[Chat History]\n{hist_str}\n\n"
        f"[Context]\n{context_text}\n\n"
        f"[Question]\n{user_input}"
    )
    answer  = generate_answer(QA_SYSTEM, qa_prompt)
    latency = round(time.time() - start_time, 2)

    # Log interaction for evaluation
    gradio_logs.append({
        "query":   user_input,
        "answer":  answer,
        "context": context_text,
        "latency": latency,
    })

    return answer


# ── Step 7: Launch Gradio interface ───────────────────────────────────────────

demo = gr.ChatInterface(
    fn=gradio_chat,
    title="🇵🇭 Philippine Family Planning Handbook AI",
    description=(
        "Ask any question about the **Philippine Family Planning Handbook (2023 Edition)**. "
        "Answers are grounded exclusively in the handbook — the chatbot will not answer "
        "questions outside its scope."
    ),
    examples=[
        "What are the steps in the GATHER approach?",
        "How is the Yuzpe method used?",
        "Can LAM be an effective method of family planning?",
        "What should a provider do if a client experiences heavy bleeding after IUD insertion?",
        "Explain the role of a partner in POPs",
    ],
)

demo.launch(server_name="0.0.0.0", server_port=10000)
