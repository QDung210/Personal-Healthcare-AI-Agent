import asyncio
import json
import re
from rag import search_medical_info
from appointment_booking import extract_information, update_appointment_info, appointment_info
from model import ROUTER_MODEL, ROUTER_TOKENIZER
from langchain.memory import ConversationBufferMemory


def route_task(user_input: str) -> str:
    """
    Sử dụng model function calling để xác định task: RAG hoặc BOOKING
    
    Returns:
        "rag" hoặc "booking"
    """
    prompt = f"""Bạn là một AI Router. Phân tích câu nói của người dùng và xác định task phù hợp.

CÓ 2 TASK:
1. "rag" - Tìm kiếm thông tin y tế (triệu chứng, bệnh, điều trị, sức khỏe)
2. "booking" - Đặt lịch khám bệnh (đặt lịch, book, hẹn khám, thông tin bệnh nhân)

QUAN TRỌNG:
- Nếu hỏi về triệu chứng, bệnh lý, sức khỏe → "rag"
- Nếu muốn đặt lịch, cung cấp thông tin cá nhân → "booking"

VÍ DỤ:
- "ho khan kéo dài" → {{"task": "rag"}}
- "đau đầu có nguy hiểm không" → {{"task": "rag"}}
- "tôi muốn đặt lịch khám" → {{"task": "booking"}}
- "tên tôi là Nguyễn Văn A" → {{"task": "booking"}}
- "tôi muốn khám vào ngày mai" → {{"task": "booking"}}

Người dùng: {user_input}

Trả về JSON:
{{"task": "rag hoặc booking"}}

JSON:"""
    
    try:
        inputs = ROUTER_TOKENIZER(prompt, return_tensors="pt").to(ROUTER_MODEL.device)
        outputs = ROUTER_MODEL.generate(**inputs, max_new_tokens=50, temperature=0.1)
        text = ROUTER_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON
        json_blocks = re.findall(r'\{[\s\S]*?\}', text)
        if json_blocks:
            result = json.loads(json_blocks[-1])
            task = result.get("task", "rag").lower()
            print(f"🔍 [Router] Detected task: {task}")
            return task
    except Exception as e:
        print(f" Router error: {e}")
    
    # Fallback: detect từ khóa
    if any(kw in user_input.lower() for kw in ["đặt lịch", "book", "hẹn", "tên tôi", "tôi tên"]):
        return "booking"
    return "rag"

async def chat_loop():
    print("🤖 Chào mừng đến với Trợ lý Y tế AI!")
    print("💬 Tôi có thể giúp bạn:")
    print("   • Tìm kiếm thông tin y tế (triệu chứng, bệnh lý, điều trị)")
    print("   • Đặt lịch khám bệnh")
    print("\n📝 Gõ 'exit' để thoát")
    print("📝 Gõ 'xem lịch' để xem thông tin đặt lịch hiện tại")
    print("📝 Gõ 'reset' để đặt lịch mới")
    print("📝 Gõ 'xong' hoặc 'hoàn tất' để kết thúc đặt lịch\n")
    print("=" * 60)
    
    current_task = None  # Task hiện tại: None, "rag", hoặc "booking"
    
    # Khởi tạo memory để lưu lịch sử chat
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output"
    )
    
    while True:
        user_input = input("\n👤 Bạn: ").strip()
        
        if user_input.lower() == "exit":
            print("\nCảm ơn bạn đã sử dụng dịch vụ. Chúc bạn khỏe mạnh!")
            break
        
        if not user_input:
            continue
        

        if user_input.lower() == "xem lịch":
            print(f"\nThông tin đặt lịch hiện tại:")
            print(f"   Tên: {appointment_info.get('patient_name') or 'chưa có'}")
            print(f"   Thời gian: {appointment_info.get('time') or 'chưa có'}")
            print(f"   Loại khám: {appointment_info.get('appointment_type') or 'chưa có'}")
            print(f"   Bệnh viện: {appointment_info.get('hospital') or 'chưa có'}")
            continue
        
        if user_input.lower() == "reset":
            for key in appointment_info.keys():
                appointment_info[key] = None
            current_task = None
            print("\nĐã reset thông tin đặt lịch!")
            continue
        
        # Check nếu user muốn hoàn tất booking
        if user_input.lower() in ["xong", "hoàn tất", "ok", "đồng ý", "xác nhận"]:
            if current_task == "booking":
                # Kiểm tra đã đủ thông tin chưa
                if appointment_info.get('patient_name') and appointment_info.get('time'):
                    print("\nĐã hoàn tất đặt lịch!")
                    print("Phòng khám sẽ liên hệ xác nhận trong 24h. Cảm ơn bạn!")
                    current_task = None  # Thoát khỏi luồng booking
                else:
                    print("\nVẫn còn thiếu thông tin. Vui lòng cung cấp đầy đủ!")
            continue
        
        if current_task is None:
            current_task = route_task(user_input)
            print(f"📍 Chế độ: {'Tìm kiếm y tế' if current_task == 'rag' else 'Đặt lịch khám'}")
        
        if current_task == "booking":
            try:
                # Extract thông tin từ câu nói
                json_text = extract_information(user_input)
                result = update_appointment_info(json_text)
                
                print(f"\nĐã cập nhật thông tin:")
                if result.get('patient_name'):
                    print(f"   Tên: {result['patient_name']}")
                if result.get('time'):
                    print(f"   Thời gian: {result['time']}")
                if result.get('appointment_type'):
                    print(f"   Loại khám: {result['appointment_type']}")
                if result.get('hospital'):
                    print(f"   Bệnh viện: {result['hospital']}")
                
                # Check missing info
                missing = []
                if not result.get('patient_name'):
                    missing.append("tên bệnh nhân")
                if not result.get('time'):
                    missing.append("thời gian khám")
                
                if missing:
                    print(f"\n Còn thiếu: {', '.join(missing)}")
                    print(" Vui lòng cung cấp thêm thông tin")
                    print(" Hoặc gõ 'xong' để hoàn tất (nếu đã đủ)")
                else:
                    print("\n Đã đủ thông tin!")
                    print("Gõ 'xong' để xác nhận, hoặc tiếp tục cập nhật thông tin")
                
                # GIỮ LUỒNG BOOKING - không tự động thoát
                
            except Exception as e:
                print(f"\n Lỗi khi xử lý đặt lịch: {e}")
        
        elif current_task == "rag":
            try:
                # Lấy lịch sử chat để cung cấp context
                chat_history = memory.load_memory_variables({}).get("chat_history", [])
                
                # Format lịch sử chat thành text
                history_text = ""
                if chat_history:
                    for msg in chat_history[-4:]:  # Chỉ lấy 4 tin nhắn gần nhất để tránh quá dài
                        role = "Người dùng" if hasattr(msg, 'type') and msg.type == "human" else "Trợ lý"
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        history_text += f"{role}: {content}\n"
                
                # Gọi RAG với context từ lịch sử
                result = await search_medical_info(user_input, chat_history=history_text)
                
                print(f"\n🤖 Trợ lý: {result.answer}")
                
                # Lưu vào memory
                memory.save_context(
                    {"input": user_input},
                    {"output": result.answer}
                )
                
                # Chỉ hiển thị nguồn nếu có và độ liên quan >= 0.6
                if result.sources and any(s.score and s.score >= 0.6 for s in result.sources):
                    print("\n" + "=" * 60)
                    print(" **Nguồn tham khảo:**")
                    for i, s in enumerate(result.sources, 1):
                        if s.score and s.score >= 0.6:  # Chỉ hiển thị nguồn có độ liên quan cao
                            print(f"\n[{i}] {s.title}")
                            if s.url and s.url != "N/A":
                                print(f"     {s.url}")
                            if s.score:
                                print(f"     Độ liên quan: {s.score}")
                            if s.snippet:
                                print(f"     {s.snippet[:150]}...")
                    print("=" * 60)
                
                # Sau khi trả lời RAG, reset task để có thể nhận task mới
                current_task = None
            
            except Exception as e:
                print(f"\n Lỗi: {e}")
                print("Vui lòng thử lại hoặc hỏi cách khác.")
                current_task = None

if __name__ == "__main__":
    asyncio.run(chat_loop())
