import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import src.config as config  
import asyncio
import json
import re
from src.services.rag import search_medical_info
from src.services.appointment_booking import extract_information, update_appointment_info, appointment_info
from src.services.classification import classify_image
from src.models.model import ROUTER_MODEL
from langchain.memory import ConversationBufferMemory
from pydantic_ai import Agent


async def route_task_async(user_input: str) -> str:
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
- "tôi muốn khám vào ngày mai" → {{"task": "booking"}}
- "tôi muốn biết mình bị bệnh qua ảnh" → {{"task": "classification"}} 
- "trong ảnh thì tôi đang bị bệnh gì thế" → {{"task": "classification"}} 

Người dùng: {user_input}

Trả về ONLY JSON (không thêm text nào khác):
{{"task": "rag hoặc booking hoặc classification"}}"""
    
    try:
        # Use Pydantic AI Agent with Bedrock model
        agent = Agent(ROUTER_MODEL, system_prompt="Bạn là AI Router chính xác.")
        result = await agent.run(prompt)
        text = result.output
        
        # Extract JSON
        json_blocks = re.findall(r'\{[\s\S]*?\}', text)
        if json_blocks:
            result_json = json.loads(json_blocks[-1])
            task = result_json.get("task", "rag").lower()
            print(f"🔍 [Router] Detected task: {task}")
            return task
    except Exception as e:
        print(f" Router error: {e}")
    
    # Fallback
    if any(kw in user_input.lower() for kw in ["đặt lịch", "book", "hẹn", "tên tôi", "tôi tên"]):
        return "booking"
    return "rag"

async def chat_loop():
    print("🤖 Chào mừng đến với Trợ lý Y tế AI!")
    print("💬 Tôi có thể giúp bạn:")
    print("   • Tìm kiếm thông tin y tế (triệu chứng, bệnh lý, điều trị)")
    print("   • Đặt lịch khám bệnh")
    print("   • Phân loại bệnh từ ảnh X-quang ngực")
    print("\n📝 Gõ 'exit' để thoát")
    print("📝 Gõ 'xem lịch' để xem thông tin đặt lịch hiện tại")
    print("📝 Gõ 'reset' để đặt lịch mới")
    print("📝 Gõ 'xong' hoặc 'hoàn tất' để kết thúc đặt lịch")
    print("📝 Gõ 'phân loại ảnh' và nhập đường dẫn ảnh để phân loại bệnh\n")
    print("=" * 60)
    
    current_task = None  
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
        if user_input.lower() in ["xong", "hoàn tất", "ok", "đồng ý", "xác nhận"]:
            if current_task == "booking":
                missing = []
                if not appointment_info.get('patient_name'):
                    missing.append("tên bệnh nhân")
                if not appointment_info.get('time'):
                    missing.append("thời gian khám")
                if not appointment_info.get('appointment_type'):
                    missing.append("loại khám")
                if not appointment_info.get('hospital'):
                    missing.append("bệnh viện")
                
                if not missing:
                    print(f" Thông tin đặt lịch:")
                    print(f"   • Tên bệnh nhân: {appointment_info['patient_name']}")
                    print(f"   • Thời gian: {appointment_info['time']}")
                    print(f"   • Loại khám: {appointment_info['appointment_type']}")
                    print(f"   • Bệnh viện: {appointment_info['hospital']}")
                    current_task = None
                else:
                    print("\n Vẫn còn thiếu thông tin:")
                    for item in missing:
                        print(f"   ✗ {item}")
                    print("\n Vui lòng cung cấp đầy đủ thông tin trước khi hoàn tất!")
            continue
        
        if current_task is None:
            current_task = await route_task_async(user_input)
            print(f" Chế độ: {'Tìm kiếm y tế' if current_task == 'rag' else 'Đặt lịch khám' if current_task == 'booking' else 'Phân loại ảnh'}")
        
        if current_task == "classification":
            try:
                print("\n Vui lòng nhập đường dẫn đến ảnh X-quang ngực:")
                image_path = input("   Đường dẫn: ").strip()
                
                if not image_path:
                    print(" Đường dẫn không hợp lệ!")
                    current_task = None
                    continue
                
                print("\n Đang phân tích ảnh...")
                result = await classify_image(image_path)
                
                print("\n" + "=" * 60)
                print(" KẾT QUẢ PHÂN LOẠI")
                print("=" * 60)
                print("\n Xác suất các bệnh:")
                for item in result["Kết quả phân loại"]:
                    if item["prob"] >= 0.5:  
                        print(f"   • {item['class']}: {item['prob']*100:.2f}%")
                print("\n Phân tích cụ thể:")
                print(f"   {result['Phân tích cụ thể']}")
                thong_tin = result.get("Thông tin liên quan", [])
                if thong_tin and isinstance(thong_tin, list) and len(thong_tin) > 0:
                    print("\n Thông tin tham khảo:")
                    count = min(3, len(thong_tin))
                    for i in range(count):
                        info = thong_tin[i]
                        if isinstance(info, dict):
                            title = info.get('title', 'Không có tiêu đề')
                            description = info.get('description', 'Không có mô tả')
                            url = info.get('url', '')
                            
                            print(f"\n   {i+1}. {title}")
                            if description and isinstance(description, str):
                                if len(description) > 200:
                                    desc_preview = description[0:200] + "..."
                                else:
                                    desc_preview = description
                                print(f"      {desc_preview}")
                            if url:
                                print(f"      🔗 {url}")
                
                print("\n" + "=" * 60)
                current_task = None
                
            except FileNotFoundError:
                print(f"\n Không tìm thấy file ảnh: {image_path}")
                print(" Vui lòng kiểm tra lại đường dẫn!")
                current_task = None
            except Exception as e:
                print(f"\n Lỗi khi phân loại ảnh: {e}")
                current_task = None
        
        elif current_task == "booking":
            try:
                json_text = extract_information(user_input)
                result = update_appointment_info(json_text)
                
                print(f"\nĐã cập nhật thông tin:")
                if result.get('patient_name'):
                    print(f"   ✓ Tên: {result['patient_name']}")
                else:
                    print(f"   ✗ Tên: chưa có")
                    
                if result.get('time'):
                    print(f"   ✓ Thời gian: {result['time']}")
                else:
                    print(f"   ✗ Thời gian: chưa có")
                    
                if result.get('appointment_type'):
                    print(f"   ✓ Loại khám: {result['appointment_type']}")
                else:
                    print(f"   ✗ Loại khám: chưa có")
                    
                if result.get('hospital'):
                    print(f"   ✓ Bệnh viện: {result['hospital']}")
                else:
                    print(f"   ✗ Bệnh viện: chưa có")
                
                # Check missing info - CHECK ALL 4 FIELDS
                missing = []
                if not result.get('patient_name'):
                    missing.append("tên bệnh nhân")
                if not result.get('time'):
                    missing.append("thời gian khám")
                if not result.get('appointment_type'):
                    missing.append("loại khám")
                if not result.get('hospital'):
                    missing.append("bệnh viện")
                
                if missing:
                    print(f"\n Còn thiếu: {', '.join(missing)}")
                    print(" Vui lòng cung cấp thêm thông tin")
                    print(" Hoặc gõ 'xong' để hoàn tất (nếu muốn bỏ qua các thông tin trên)")
                else:
                    print("\n Đã đủ thông tin!")
                    print(" Gõ 'xong' để xác nhận, hoặc tiếp tục cập nhật thông tin")
            except Exception as e:
                print(f"\n Lỗi khi xử lý đặt lịch: {e}")
        
        elif current_task == "rag":
            try:
                chat_history = memory.load_memory_variables({}).get("chat_history", [])
                history_text = ""
                if chat_history:
                    for msg in chat_history[-4:]:  
                        role = "Người dùng" if hasattr(msg, 'type') and msg.type == "human" else "Trợ lý"
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        history_text += f"{role}: {content}\n"
                print("\n Trợ lý: ", end="", flush=True)
                
                full_response = ""
                stream_gen = await search_medical_info(user_input, chat_history=history_text, stream=True)
                
                async for chunk in stream_gen:
                    print(chunk, end="", flush=True)
                    full_response += chunk
                
                print()  

                memory.save_context(
                    {"input": user_input},
                    {"output": full_response}
                )

                current_task = None
            
            except Exception as e:
                print(f"\n Lỗi: {e}")
                print("Vui lòng thử lại hoặc hỏi cách khác.")
                current_task = None
        
if __name__ == "__main__":
    asyncio.run(chat_loop())
