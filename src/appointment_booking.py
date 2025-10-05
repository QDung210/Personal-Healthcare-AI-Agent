from model import ROUTER_MODEL, ROUTER_TOKENIZER
import re
import json

appointment_info = {
    "patient_name": None,
    "appointment_type": None,
    "time": None,
    "hospital": None
}

def extract_information(user_input: str) -> str:
    prompt = f"""Bạn là một AI Agent chỉ output JSON dưới dạng function call.
Hãy trích thông tin từ câu người dùng và trả kết quả JSON có format:
{{
    "patient_name": null,
    "appointment_type": null,
    "time": null,
    "hospital": null
}}
Nếu như trong input của người dùng có thông tin nào chưa đề cập thì cứ để là null.
Tuyệt đối không được thêm thông tin lạ hoặc thông tin không có trong câu nói của người dùng 

Ví dụ 1:
Người dùng nhập input là: Tôi tên là Đỗ Quốc Dũng. Thì trả về là 
{{
    "patient_name": "Đỗ Quốc Dũng",
    "appointment_type": null,
    "time": null,
    "hospital": null
}}
Tuyệt đối không điền trường "appointment_type" và "hospital" và "time" vì trong câu input của người dùng không có
Ví dụ 2:
Người dùng nhập input là: Tôi tên là Đỗ Quốc Dũng, tôi muốn đặt lịch khám vào sáng mai. Thì trả về là 
{{
    "patient_name": "Đỗ Quốc Dũng",
    "appointment_type": null,
    "time": "Sáng mai",
    "hospital": null
}}
Tuyệt đối không điền trường "appointment_type" và "hospital" vì trong câu input của người dùng không có
Người dùng: {user_input}

JSON:"""
    
    # Gọi model sinh output
    inputs = ROUTER_TOKENIZER(prompt, return_tensors="pt").to(ROUTER_MODEL.device)
    outputs = ROUTER_MODEL.generate(**inputs, max_new_tokens=256)
    
    # Giải mã text model sinh ra
    text = ROUTER_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    
    # Tìm block JSON cuối cùng
    json_blocks = re.findall(r'\{[\s\S]*?\}', text)
    json_text = json_blocks[-1].strip()
    return json_text

def update_appointment_info(json_text: str) -> dict:
    data = json.loads(json_text)

    for key in appointment_info.keys():
        if data.get(key) not in [None, "null", "None", ""]:
            appointment_info[key] = data[key]

    return appointment_info

