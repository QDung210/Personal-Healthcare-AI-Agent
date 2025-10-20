# 🏥 Medical AI Assistant - Full Stack Application

## 📌 Tổng quan

Ứng dụng Trợ lý Y tế AI kết hợp **Backend Python (FastAPI)** và **Frontend Next.js**, hỗ trợ 3 chức năng chính:

1. **RAG (Retrieval-Augmented Generation)**: Hỏi đáp y tế thông minh
2. **Booking**: Đặt lịch khám bệnh tự động
3. **Classification**: Phân loại bệnh từ ảnh X-quang ngực

---

## 🚀 CÁCH CHẠY NHANH

### 🎯 Bước 1: Chạy Backend
```bash
python api_server.py
```
✅ Backend sẽ chạy tại: http://localhost:8000

### 🎯 Bước 2: Chạy Frontend (Terminal mới)
```bash
cd Frontend
pnpm install    # Chỉ cần chạy lần đầu
pnpm dev
```
✅ Frontend sẽ chạy tại: http://localhost:3000

### 🎯 Bước 3: Mở trình duyệt
Truy cập: **http://localhost:3000**

---

## 📦 Cài đặt lần đầu

### Backend (Python):
```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Tạo file .env với AWS credentials
# Xem mẫu trong FRONTEND_SETUP_GUIDE.md
```

### Frontend (Node.js):
```bash
cd Frontend

# Cài đặt pnpm (nếu chưa có)
npm install -g pnpm

# Cài đặt dependencies
pnpm install
```

---

## 💡 Cách sử dụng

### 1️⃣ RAG - Hỏi đáp y tế
```
👤: "Đau đầu kéo dài có nguy hiểm không?"
🤖: [Trả lời chi tiết với nguồn tham khảo từ database y khoa]
```

### 2️⃣ Booking - Đặt lịch khám
```
👤: "Tôi muốn đặt lịch khám"
🤖: [Mở form đặt lịch bên phải]

👤: "Tên tôi là Nguyễn Văn A"
🤖: [Cập nhật tên]

👤: "Khám tim mạch vào 9h sáng mai tại Bạch Mai"
🤖: [Cập nhật đầy đủ thông tin]

→ Click "Xác nhận đặt lịch" để hoàn tất
```

### 3️⃣ Classification - Phân loại ảnh X-quang
```
1. Click nút 📎 (Attach file)
2. Chọn ảnh X-quang ngực (.jpg, .png)
3. Gõ: "Phân tích ảnh này"
🤖: [Hiển thị kết quả phân loại với xác suất từng loại bệnh]
```

---

## 🔧 Cấu trúc Project

```
PROJECT/
│
├── 🔥 api_server.py          # FastAPI Backend Server
├── main.py                   # CLI version (standalone)
├── requirements.txt          # Python dependencies
├── .env                      # AWS & Qdrant credentials
│
├── src/
│   ├── models/              # AI Models (Bedrock, Router)
│   ├── services/            # RAG, Booking, Classification
│   └── utils/
│
└── Frontend/
    ├── app/
    │   ├── api/chat/route.ts    # API Proxy to Backend
    │   └── page.tsx             # Main page
    │
    ├── components/
    │   ├── medical-chatbot.tsx  # Chat UI
    │   └── booking-form.tsx     # Booking Form UI
    │
    └── package.json
```

---

## 🌐 API Endpoints

### Backend (Python - FastAPI)
- `GET /` - Health check
- `GET /health` - Health status
- `POST /api/chat` - Main chat endpoint

**Request:**
```json
FormData {
  "message": "user message",
  "messages": "[chat history]",
  "files": [File objects]
}
```

**Response:**
```json
{
  "task": "rag" | "booking" | "classification",
  "message": "AI response",
  "booking_data": {...},        // if task=booking
  "classification_result": {...} // if task=classification
}
```

---

## 🎨 Tech Stack

### Backend:
- **FastAPI** - Web framework
- **AWS Bedrock** - LLM (Claude Sonnet)
- **Qdrant** - Vector database
- **TensorFlow** - Image classification
- **LangChain** - RAG framework
- **Pydantic AI** - AI agents

### Frontend:
- **Next.js 15** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Shadcn UI** - UI components
- **Lucide Icons** - Icons

---

## 📸 Screenshots

### Chat Interface
- RAG mode: Hỏi đáp y tế với sources
- Booking mode: Form đặt lịch bên phải
- Classification mode: Kết quả phân loại ảnh

---

## 🐛 Troubleshooting

### Lỗi kết nối Backend
```bash
# Kiểm tra backend đang chạy
curl http://localhost:8000/health

# Hoặc mở trình duyệt
http://localhost:8000
```

### Frontend không load
```bash
cd Frontend
rm -rf node_modules
pnpm install
pnpm dev
```

### Port bị chiếm
```bash
# Thay đổi port backend (api_server.py)
uvicorn.run(app, host="0.0.0.0", port=8001)

# Thay đổi port frontend
pnpm dev -- -p 3001
```

---

## 📚 Tài liệu chi tiết

- **FRONTEND_SETUP_GUIDE.md** - Hướng dẫn chi tiết setup
- **QUICK_START.md** - Quick start guide
- **start_servers.bat** - Script tự động chạy servers (Windows)

---

## 🔐 Environment Variables

File `.env`:
```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

---

## ✅ Features

- ✅ Real-time chat interface
- ✅ AI-powered task routing (RAG/Booking/Classification)
- ✅ File upload support (images for classification)
- ✅ Dynamic booking form
- ✅ Article extraction and display
- ✅ Responsive design
- ✅ Error handling
- ✅ CORS configured
- ✅ Type-safe frontend

---

## 🚀 Production Deployment

### Backend:
```bash
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend:
```bash
cd Frontend
pnpm build
pnpm start
```

---

## 📝 License

MIT License - Dự án học tập

---

## 👨‍💻 Author

Medical AI Assistant - Full Stack Application

---

**Happy Coding! 🎉**
