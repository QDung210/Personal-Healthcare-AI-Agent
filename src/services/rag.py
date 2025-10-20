"""
Retrieval tools used to generate text or query documents
"""
import src.config as config  
from dataclasses import dataclass
from qdrant_client import QdrantClient
from src.models.model import EMBEDDER, QDRANT_CLIENT, RAG_AGENT
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict
from pydantic import BaseModel
import textwrap
import re


@dataclass
class Deps:
    qdrant: QdrantClient
    embedder: SentenceTransformer

DEPS = Deps(qdrant=QDRANT_CLIENT, embedder=EMBEDDER)

class SourceModel(BaseModel):
    title: str
    url: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Optional[str]] = {}
    snippet: Optional[str] = None

class RAGResult(BaseModel):
    answer: str
    sources: List[SourceModel]


def remove_urls_from_text(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def deduplicate_sources(docs: List[dict], max_sources: int = 3) -> List[dict]:
    seen_titles = {}
    
    # Group by title, keep highest score
    for doc in docs:
        title = doc.get("title", "Không có tiêu đề")
        score = doc.get("score", 0)
        
        if title not in seen_titles:
            seen_titles[title] = doc
        else:
            # Keep the one with higher score
            existing_score = seen_titles[title].get("score", 0)
            if score > existing_score:
                seen_titles[title] = doc
    
    # Convert back to list and sort by score
    unique_docs = list(seen_titles.values())
    unique_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Return top max_sources
    return unique_docs[:max_sources]


def retrieve_documents(deps: Deps, query: str, limit: int = 15):
    print(f"📊 Search query gửi đến Qdrant: '{query}'")
    qvec = deps.embedder.encode(query).tolist()
    hits = deps.qdrant.query_points(
        collection_name="medical_collection",
        query=qvec,
        limit=limit, 
        with_payload=True,
    ).points

    docs = []
    for hit in hits:
        payload = getattr(hit, "payload", {}) or {}
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        content = payload.get("page_content", "") or metadata.get("text", "") or metadata.get("content", "")
        if not content:
            continue
        score = getattr(hit, "score", None)
        title = metadata.get("title", "Không có tiêu đề")
        docs.append({
            "title": title,
            "content": content[:1500],
            "score": round(score, 3) if score else None,
            "metadata": metadata
        })
    context = "\n\n".join(f"[Nguồn {i+1}] {d['content']}" for i, d in enumerate(docs))
    return context, docs

async def search_medical_info(query: str, chat_history: str = "", stream: bool = False):
    print(f"Đang tìm kiếm thông tin về: {query}")
    search_query = query
    if chat_history:
        lines = chat_history.strip().split('\n')
        for line in reversed(lines):
            if line.startswith("Người dùng:") and not line.endswith(query):
                previous_question = line.replace("Người dùng:", "").strip()
                if len(query.split()) <= 10 and any(kw in query.lower() for kw in ["còn", "thêm", "nữa", "khác", "nào", "nào khác", "gì nữa"]):
                    search_query = f"{previous_question} {query}"
                    print(f"Mở rộng tìm kiếm: {search_query}")
                break
    
    context, docs = retrieve_documents(DEPS, search_query, limit=15)  
    filtered_docs = [d for d in docs if d.get('score', 0) >= 0.6]
    
    if not filtered_docs:
        filtered_docs = [d for d in docs if d.get('score', 0) >= 0.5]
        if not filtered_docs:
            if stream:
                async def error_stream():
                    yield "Không tìm thấy thông tin liên quan đủ chính xác trong cơ sở dữ liệu. Vui lòng hỏi cách khác hoặc đặt lịch khám để được bác sĩ tư vấn trực tiếp."
                return error_stream()
            else:
                return RAGResult(
                    answer="Không tìm thấy thông tin liên quan đủ chính xác trong cơ sở dữ liệu. Vui lòng hỏi cách khác hoặc đặt lịch khám để được bác sĩ tư vấn trực tiếp.",
                    sources=[]
                )
    unique_docs = deduplicate_sources(filtered_docs, max_sources=3)
    
    context = "\n\n".join(
        f"[Nguồn {i+1}] {remove_urls_from_text(d['content'])}" 
        for i, d in enumerate(filtered_docs[:10])
    ) 

    docs = unique_docs
    
    if not docs:
        if stream:
            async def error_stream():
                yield "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu. Vui lòng hỏi cách khác hoặc đặt lịch khám để được bác sĩ tư vấn trực tiếp."
            return error_stream()
        else:
            return RAGResult(
                answer="Không tìm thấy thông tin liên quan trong cơ sở dữ liệu. Vui lòng hỏi cách khác hoặc đặt lịch khám để được bác sĩ tư vấn trực tiếp.",
                sources=[]
            )

    context_section = f"\nLỊCH SỬ HỘI THOẠI:\n{chat_history}\n" if chat_history else ""
    
    prompt = textwrap.dedent(f"""
        Bạn là bác sĩ y tế chuyên nghiệp. Dựa trên thông tin sau, hãy trả lời câu hỏi "{query}" bằng tiếng Việt một cách chi tiết, dễ hiểu.
        {context_section}
        QUAN TRỌNG: 
        - Nếu câu hỏi hiện tại đề cập đến "còn", "thêm", "nữa", "khác" thì hãy dựa vào lịch sử hội thoại để hiểu người dùng đang hỏi tiếp về chủ đề nào.
        - SỬ DỤNG MARKDOWN để format câu trả lời đẹp mắt
        - Trả lời nội dung một cách tự nhiên, mượt mà, có cấu trúc rõ ràng.
        DỮ LIỆU Y TẾ:
        {context}
        
        Trả lời:
    """)

    sources_out = []
    for i, d in enumerate(docs, start=1):
        m = d.get("metadata", {}) or {}
        url = m.get("url") or m.get("source_url") or m.get("link") or None
        
        snippet = (d.get("content")[:300] + "...") if d.get("content") else ""
        sources_out.append(SourceModel(
            title=d.get("title", f"Nguồn {i}"),
            url=url if url else "N/A",
            score=d.get("score"),
            metadata={k: str(v) for k, v in m.items()},
            snippet=snippet
        ))

    if stream:
        async def stream_response():
            async with RAG_AGENT.run_stream(prompt, deps=DEPS) as result:
                async for chunk in result.stream_text(delta=True):
                    yield chunk
            if sources_out:
                yield "\n\n"
                for source in sources_out:
                    if source.url and source.url != "N/A":
                        yield f"[ARTICLE]\n"
                        yield f"title: {source.title}\n"
                        yield f"url: {source.url}\n"
                        yield f"category: Y TẾ\n"
                        yield f"[/ARTICLE]\n\n"
        
        return stream_response()
    
    # NON-STREAMING MODE (backward compatible)
    else:
        result = await RAG_AGENT.run(prompt, deps=DEPS)
        answer = str(result.output)
        return RAGResult(answer=answer, sources=sources_out)

