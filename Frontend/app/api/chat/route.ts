import { type NextRequest, NextResponse } from "next/server"

// Backend API URL
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const message = (formData.get("message") as string) || ""
    const messagesStr = (formData.get("messages") as string) || "[]"
    const messages = JSON.parse(messagesStr)

    // Extract uploaded files
    const fileEntries = formData.getAll("files")
    const files = fileEntries.filter((entry): entry is File => entry instanceof File)

    if (files.length > 0) {
      console.log(
        "[Frontend] Sending files to backend:",
        files.map((f) => ({ name: f.name, type: f.type, size: f.size })),
      )
    }

    // Create FormData to send to Python backend
    const backendFormData = new FormData()
    backendFormData.append("message", message)
    backendFormData.append("messages", messagesStr)
    
    // Append files if any
    files.forEach((file) => {
      backendFormData.append("files", file)
    })

    console.log("[Frontend] Sending request to backend:", `${BACKEND_URL}/api/chat`)

    // Call Python backend
    const backendResponse = await fetch(`${BACKEND_URL}/api/chat`, {
      method: "POST",
      body: backendFormData,
    })

    if (!backendResponse.ok) {
      throw new Error(`Backend error! status: ${backendResponse.status}`)
    }

    const data = await backendResponse.json()
    console.log("[Frontend] Received response from backend:", data)

    return NextResponse.json(data)
  } catch (error) {
    console.error("[Frontend] Chat API error:", error)
    return NextResponse.json(
      {
        task: "rag",
        message: "Xin lỗi, không thể kết nối đến server backend. Vui lòng kiểm tra xem backend đã chạy chưa (python api_server.py)",
      },
      { status: 500 },
    )
  }
}
