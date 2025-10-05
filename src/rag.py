from dataclasses import dataclass
from qdrant_client import QdrantClient
from model import EMBEDDER, QDRANT_CLIENT, RAG_AGENT
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict
from pydantic import BaseModel
import textwrap


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


def retrieve_documents(deps: Deps, query: str, limit: int = 5):
    """Tìm kiếm tài liệu y tế từ Qdrant"""
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
        print(f"   ✓ Tìm thấy: '{title[:60]}...' (score: {round(score, 3) if score else 'N/A'})")
        docs.append({
            "title": title,
            "content": content[:1500],
            "score": round(score, 3) if score else None,
            "metadata": metadata
        })
    context = "\n\n".join(f"[Nguồn {i+1}] {d['content']}" for i, d in enumerate(docs))
    return context, docs

async def search_medical_info(query: str, chat_history: str = "") -> RAGResult:
    print(f"🔍 Đang tìm kiếm thông tin về: {query}")
    
    # Nếu có lịch sử chat, tạo query mở rộng để tìm kiếm tốt hơn
    search_query = query
    if chat_history:
        # Trích xuất chủ đề chính từ lịch sử (câu hỏi gần nhất của user)
        lines = chat_history.strip().split('\n')
        for line in reversed(lines):
            if line.startswith("Người dùng:") and not line.endswith(query):
                previous_question = line.replace("Người dùng:", "").strip()
                # Nếu câu hỏi hiện tại ngắn và có từ như "còn", "thêm", "nữa" thì kết hợp với câu trước
                if len(query.split()) <= 10 and any(kw in query.lower() for kw in ["còn", "thêm", "nữa", "khác", "nào", "nào khác", "gì nữa"]):
                    search_query = f"{previous_question} {query}"
                    print(f"🔍 Mở rộng tìm kiếm: {search_query}")
                break
    
    context, docs = retrieve_documents(DEPS, search_query, limit=5)  # Tăng lên 5 để có nhiều lựa chọn
    
    # Lọc docs theo độ liên quan (chỉ giữ lại score >= 0.6)
    filtered_docs = [d for d in docs if d.get('score', 0) >= 0.6]
    
    if not filtered_docs:
        # Nếu không có doc nào đủ điểm, thử giảm xuống 0.5
        filtered_docs = [d for d in docs if d.get('score', 0) >= 0.5]
        if not filtered_docs:
            return RAGResult(
                answer=" Không tìm thấy thông tin liên quan đủ chính xác trong cơ sở dữ liệu. Vui lòng hỏi cách khác hoặc đặt lịch khám để được bác sĩ tư vấn trực tiếp.",
                sources=[]
            )
    
    # Rebuild context từ filtered docs
    context = "\n\n".join(f"[Nguồn {i+1}] {d['content']}" for i, d in enumerate(filtered_docs))
    docs = filtered_docs  # Sử dụng docs đã lọc
    
    if not docs:
        return RAGResult(
            answer=" Không tìm thấy thông tin liên quan trong cơ sở dữ liệu. Vui lòng hỏi cách khác hoặc đặt lịch khám để được bác sĩ tư vấn trực tiếp.",
            sources=[]
        )

    # Tạo prompt cho RAG với lịch sử chat
    context_section = f"\nLỊCH SỬ HỘI THOẠI:\n{chat_history}\n" if chat_history else ""
    
    prompt = textwrap.dedent(f"""
        Bạn là bác sĩ y tế chuyên nghiệp. Dựa trên thông tin sau, hãy trả lời câu hỏi "{query}" bằng tiếng Việt một cách chi tiết, dễ hiểu.
        {context_section}
        QUAN TRỌNG: Nếu câu hỏi hiện tại đề cập đến "còn", "thêm", "nữa", "khác" thì hãy dựa vào lịch sử hội thoại để hiểu người dùng đang hỏi tiếp về chủ đề nào.
        
        Hãy trích dẫn nguồn bằng cách viết [Nguồn 1], [Nguồn 2] khi sử dụng thông tin từ tài liệu.
        
        DỮ LIỆU Y TẾ:
        {context}
        
        Trả lời:
    """)

    # Gọi RAG Agent để sinh văn bản
    result = await RAG_AGENT.run(prompt, deps=DEPS)
    answer = str(result.output)

    # Format sources
    sources_out = []
    for i, d in enumerate(docs, start=1):
        m = d.get("metadata", {}) or {}
        url = m.get("url") or m.get("source_url") or m.get("link") or "N/A"
        snippet = (d.get("content")[:300] + "...") if d.get("content") else ""
        sources_out.append(SourceModel(
            title=d.get("title", f"Nguồn {i}"),
            url=url,
            score=d.get("score"),
            metadata={k: str(v) for k, v in m.items()},
            snippet=snippet
        ))
    
    return RAGResult(answer=answer, sources=sources_out)

