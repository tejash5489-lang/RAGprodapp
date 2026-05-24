import logging
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
import inngest
import inngest.fast_api
from groq import Groq
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
)


# ─── FUNCTION 1: Ingest PDF ───────────────────────────────────────────────────

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):

    async def _load():
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return {"chunks": chunks, "source_id": source_id}

    async def _upsert():
        chunks = chunks_and_src["chunks"]
        source_id = chunks_and_src["source_id"]
        vecs = embed_texts(chunks)
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]
        payloads = [{"source_id": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return {"ingested": len(chunks)}

    chunks_and_src = await ctx.step.run("load-and-chunk", _load)
    result = await ctx.step.run("embed-and-upsert", _upsert)
    return result


# ─── FUNCTION 2: Query PDF ────────────────────────────────────────────────────

@inngest_client.create_function(
    fn_id="RAG: Search",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    async def _search():
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return {"contexts": found["contexts"], "sources": found["sources"]}

    found = await ctx.step.run("embed-and-search", _search)

    async def _llm_answer():
        context_block = "\n\n".join(f"- {c}" for c in found["contexts"])
        prompt = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer concisely using the context above."
        )
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    answer = await ctx.step.run("llm-answer", _llm_answer)

    return {
        "answer": answer,
        "sources": found["sources"],
        "num_contexts": len(found["contexts"]),
    }


# ─── Serve ────────────────────────────────────────────────────────────────────

inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf, rag_query_pdf_ai],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}