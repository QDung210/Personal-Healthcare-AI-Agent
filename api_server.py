"""
FastAPI server
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import src.config as config
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional
import json
import asyncio
import re
import tempfile
import os
from src.services.rag import search_medical_info
from src.services.appointment_booking import extract_information, update_appointment_info, appointment_info
from src.services.classification import classify_image
from src.models.model import ROUTER_MODEL
from pydantic_ai import Agent

app = FastAPI(title="Medical AI Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def route_task_async(user_input: str) -> str:
    """
    Sử dụng Bedrock model để xác định task: RAG, BOOKING, hoặc CLASSIFICATION
    """
    prompt = f"""Bạn là một AI Router. Phân tích câu nói của người dùng và xác định task phù hợp.

CÓ 3 TASK:
1. "rag" - Tìm kiếm thông tin y tế (triệu chứng, bệnh, điều trị, sức khỏe)
2. "booking" - Đặt lịch khám bệnh (đặt lịch, book, hẹn khám, thông tin bệnh nhân)
3. "classification" - Phân loại ảnh bệnh (phân loại ảnh, phân loại bệnh, tìm bệnh qua ảnh)

QUAN TRỌNG:
- Nếu hỏi về triệu chứng, bệnh lý, sức khỏe → "rag"
- Nếu muốn đặt lịch, cung cấp thông tin cá nhân → "booking"
- Nếu muốn phân loại bệnh thông qua ảnh → "classification"

VÍ DỤ:
- "ho khan kéo dài" → {{"task": "rag"}}
- "đau đầu có nguy hiểm không" → {{"task": "rag"}}
- "tôi muốn đặt lịch khám" → {{"task": "booking"}}
- "tên tôi là Nguyễn Văn A" → {{"task": "booking"}}
- "tôi muốn biết mình bị bệnh qua ảnh" → {{"task": "classification"}}

Người dùng: {user_input}

Trả về ONLY JSON (không thêm text nào khác):
{{"task": "rag hoặc booking hoặc classification"}}"""
    
    try:
        agent = Agent(ROUTER_MODEL, system_prompt="Bạn là AI Router chính xác.")
        result = await agent.run(prompt)
        text = result.output
        
        json_blocks = re.findall(r'\{[\s\S]*?\}', text)
        if json_blocks:
            result_json = json.loads(json_blocks[-1])
            task = result_json.get("task", "rag").lower()
            print(f"🔍 [Router] Detected task: {task}")
            return task
    except Exception as e:
        print(f"⚠️ Router error: {e}")
    
    # Fallback
    if any(kw in user_input.lower() for kw in ["đặt lịch", "book", "hẹn", "tên tôi", "tôi tên"]):
        return "booking"
    return "rag"


@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...),
    messages: str = Form("[]"),
    files: List[UploadFile] = File(None)
):
    """
    Main chat endpoint - handles RAG, BOOKING, and CLASSIFICATION tasks
    """
    try:
        print(f"\n Received message: {message}")
        print(f" Files attached: {len(files) if files else 0}")

        chat_history = json.loads(messages) if messages else []
        # Determine task
        task = await route_task_async(message)
        print(f"🎯 Task: {task}")
        
        # CLASSIFICATION MODE - requires image
        if task == "classification":
            if not files or len(files) == 0:
                return JSONResponse({
                    "task": "classification",
                    "message": " Vui lòng đính kèm ảnh X-quang ngực để phân loại bệnh.",
                    "error": "no_image"
                })
            
            # Save uploaded file temporarily
            image_file = files[0]
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"xray_{asyncio.get_event_loop().time()}.jpg")
            
            with open(temp_path, "wb") as f:
                content = await image_file.read()
                f.write(content)
            
            print(f"💾 Saved image to: {temp_path}")
            
            try:
                # Classify image
                result = await classify_image(temp_path)
                
                # Format response
                probabilities = result.get("Kết quả phân loại", [])
                analysis = result.get("Phân tích cụ thể", "")
                related_info = result.get("Thông tin liên quan", [])
                
                # Build formatted message
                response_message = " **KẾT QUẢ PHÂN LOẠI**\n\n"
                response_message += " **Xác suất các bệnh:**\n"
                
                for item in probabilities:
                    if item["prob"] >= 0.5:
                        response_message += f"• {item['class']}: {item['prob']*100:.2f}%\n"
                
                response_message += f"\n **Phân tích:**\n{analysis}\n"
                
                # Add related articles in special format
                if related_info and len(related_info) > 0:
                    response_message += "\n **Thông tin tham khảo:**\n\n"
                    for i, info in enumerate(related_info[:3]):
                        title = info.get('title', 'Không có tiêu đề')
                        url = info.get('url', '')
                        response_message += f"[ARTICLE]\n"
                        response_message += f"title: {title}\n"
                        response_message += f"url: {url}\n"
                        response_message += f"category: Y TẾ\n"
                        response_message += f"[/ARTICLE]\n\n"
                
                return JSONResponse({
                    "task": "classification",
                    "message": response_message,
                    "classification_result": result
                })
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # BOOKING MODE
        elif task == "booking":
            try:
                json_text = extract_information(message)
                result = update_appointment_info(json_text)
                
                # Check missing fields
                missing = []
                if not result.get('patient_name'):
                    missing.append("tên bệnh nhân")
                if not result.get('time'):
                    missing.append("thời gian khám")
                if not result.get('appointment_type'):
                    missing.append("loại khám")
                if not result.get('hospital'):
                    missing.append("bệnh viện")
                
                response_message = "Tôi đã cập nhật thông tin đặt lịch. "
                if missing:
                    response_message += f"Còn thiếu: {', '.join(missing)}. Vui lòng cung cấp thêm."
                else:
                    response_message += "Đã đủ thông tin! Vui lòng kiểm tra và xác nhận."
                
                return JSONResponse({
                    "task": "booking",
                    "message": response_message,
                    "booking_data": {
                        "patient_name": result.get('patient_name') or '',
                        "time": result.get('time') or '',
                        "appointment_type": result.get('appointment_type') or '',
                        "hospital": result.get('hospital') or ''
                    }
                })
                
            except Exception as e:
                print(f" Booking error: {e}")
                return JSONResponse({
                    "task": "booking",
                    "message": f"Lỗi khi xử lý đặt lịch: {str(e)}"
                })
        
        # RAG MODE (default)
        else:
            # Build chat history text
            history_text = ""
            for msg in chat_history[-4:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                role_name = "Người dùng" if role == "user" else "Trợ lý"
                history_text += f"{role_name}: {content}\n"
            
            # Get RAG response (collect stream)
            full_response = ""
            stream_gen = await search_medical_info(message, chat_history=history_text, stream=True)
            
            async for chunk in stream_gen:
                full_response += chunk
            
            return JSONResponse({
                "task": "rag",
                "message": full_response
            })
    
    except Exception as e:
        print(f" Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {
                "task": "rag",
                "message": "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.",
                "error": str(e)
            },
            status_code=500
        )


@app.get("/")
async def root():
    return {"message": "Medical AI Backend is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    print(" Starting Medical AI Backend Server...")
    print(" Server will be available at: http://localhost:8000")
    print(" Frontend should connect to: http://localhost:8000/api/chat")
    uvicorn.run(app, host="0.0.0.0", port=8000)
