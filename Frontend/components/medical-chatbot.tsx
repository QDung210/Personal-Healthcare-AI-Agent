"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Send, Paperclip, X, FileText, ImageIcon, ExternalLink } from "lucide-react"
import BookingForm from "@/components/booking-form"
import { MarkdownRenderer } from "@/components/markdown-renderer"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  attachments?: Array<{
    name: string
    type: string
    url: string
  }>
  articles?: Array<{
    url: string
    title: string
    category?: string
  }>
}

type BookingData = {
  patient_name: string
  appointment_type: string
  time: string
  hospital: string
}

type ChatMode = "default" | "booking"

export default function MedicalChatbot() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [mode, setMode] = useState<ChatMode>("default")
  const [bookingData, setBookingData] = useState<BookingData | null>(null)
  const [attachments, setAttachments] = useState<File[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async () => {
    if ((!input.trim() && attachments.length === 0) || isLoading) return

    const messageAttachments = attachments.map((file) => ({
      name: file.name,
      type: file.type,
      url: URL.createObjectURL(file),
    }))

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
      attachments: messageAttachments.length > 0 ? messageAttachments : undefined,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setAttachments([])
    setIsLoading(true)

    try {
      const formData = new FormData()
      formData.append("message", input)
      formData.append("messages", JSON.stringify(messages))
      attachments.forEach((file) => {
        formData.append("files", file)
      })

      const response = await fetch("/api/chat", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      if (data.task === "booking") {
        setMode("booking")
        if (data.booking_data) {
          // Sanitize booking data to ensure no null values
          const sanitizedBookingData: BookingData = {
            patient_name: data.booking_data.patient_name || "",
            appointment_type: data.booking_data.appointment_type || "",
            time: data.booking_data.time || "",
            hospital: data.booking_data.hospital || "",
          }
          setBookingData(sanitizedBookingData)
        }
      } else if (data.task === "rag") {
        setMode("default")
      } else if (data.task === "classification") {
        // Classification mode - keep default view
        setMode("default")
      }

      const messageContent = data.message || data.response || ""
      const articles = extractArticles(messageContent)

      // Create assistant message with empty content first (for streaming effect)
      const assistantMessageId = (Date.now() + 1).toString()
      const assistantMessage: Message = {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
        articles: articles.length > 0 ? articles : undefined,
      }

      setMessages((prev) => [...prev, assistantMessage])

      // Simulate streaming effect by adding text gradually
      let currentIndex = 0
      const streamInterval = setInterval(() => {
        if (currentIndex < messageContent.length) {
          const chunkSize = Math.min(3, messageContent.length - currentIndex)
          const chunk = messageContent.slice(currentIndex, currentIndex + chunkSize)
          currentIndex += chunkSize

          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId ? { ...msg, content: msg.content + chunk } : msg,
            ),
          )
        } else {
          clearInterval(streamInterval)
        }
      }, 20) // Add characters every 20ms for smooth effect
    } catch (error) {
      console.error("[v0] Error sending message:", error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleBookingConfirm = () => {
    setMode("default")
    setBookingData(null)
  }

  const handleBookingCancel = () => {
    setMode("default")
    setBookingData(null)
  }

  const handleBookingUpdate = (data: BookingData) => {
    // Ensure all fields are strings, never null or undefined
    const sanitizedData: BookingData = {
      patient_name: data.patient_name || "",
      appointment_type: data.appointment_type || "",
      time: data.time || "",
      hospital: data.hospital || "",
    }
    setBookingData(sanitizedData)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    const validFiles = files.filter((file) => {
      const isImage = file.type.startsWith("image/")
      const isPDF = file.type === "application/pdf"
      const isWord =
        file.type === "application/msword" ||
        file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      return isImage || isPDF || isWord
    })
    setAttachments((prev) => [...prev, ...validFiles])
  }

  const removeAttachment = (index: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== index))
  }

  const extractArticles = (content: string): Array<{ url: string; title: string; category?: string }> => {
    if (!content || typeof content !== "string") {
      return []
    }

    const articleRegex = /\[ARTICLE\]([\s\S]*?)\[\/ARTICLE\]/g
    const articleMatches = [...content.matchAll(articleRegex)]

    if (articleMatches.length > 0) {
      return articleMatches.map((match) => {
        const articleContent = match[1]
        const titleMatch = articleContent.match(/title:\s*(.+)/i)
        const urlMatch = articleContent.match(/url:\s*(.+)/i)
        const categoryMatch = articleContent.match(/category:\s*(.+)/i)

        return {
          url: urlMatch ? urlMatch[1].trim() : "",
          title: titleMatch ? titleMatch[1].trim() : "Bài viết",
          category: categoryMatch ? categoryMatch[1].trim() : "Y TẾ",
        }
      })
    }

    const urlRegex = /(https?:\/\/[^\s]+)/g
    const urls = content.match(urlRegex) || []

    return urls.map((url, index) => {
      const urlIndex = content.indexOf(url)
      const beforeUrl = content.substring(Math.max(0, urlIndex - 100), urlIndex).trim()

      const titleMatch = beforeUrl.match(/([^.!?\n]+)$/)
      const title = titleMatch ? titleMatch[1].trim() : `Bài viết ${index + 1}`

      return {
        url,
        title: title || `Bài viết ${index + 1}`,
        category: "Y TẾ",
      }
    })
  }

  const ArticleCard = ({ article }: { article: { url: string; title: string; category?: string } }) => {
    // Get screenshot from Google PageSpeed Insights API (free)
    const screenshotUrl = `https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url=${encodeURIComponent(article.url)}&screenshot=true`;
    
    // Fallback: Extract domain for favicon
    let domain = '';
    try {
      const urlObj = new URL(article.url);
      domain = urlObj.hostname;
    } catch (e) {
      domain = '';
    }
    
    const faviconUrl = domain ? `https://www.google.com/s2/favicons?domain=${domain}&sz=128` : null;
    
    return (
      <a
        href={article.url}
        target="_blank"
        rel="noopener noreferrer"
        className="block bg-card border border-border rounded-lg overflow-hidden hover:shadow-md transition-shadow group"
      >
        <div className="aspect-video bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-950 dark:to-indigo-950 flex items-center justify-center relative overflow-hidden">
          {faviconUrl ? (
            <div className="flex flex-col items-center justify-center gap-2">
              <img 
                src={faviconUrl} 
                alt={domain}
                className="w-12 h-12 rounded-lg shadow-sm"
                onError={(e) => {
                  // Fallback to icon if favicon fails
                  e.currentTarget.style.display = 'none';
                }}
              />
              <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                <ExternalLink className="w-4 h-4 text-primary" />
              </div>
            </div>
          ) : (
            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
              <ExternalLink className="w-6 h-6 text-primary" />
            </div>
          )}
        </div>
        <div className="p-3">
          {article.category && (
            <span className="text-xs font-semibold text-primary uppercase tracking-wide">{article.category}</span>
          )}
          <h3 className="text-sm font-medium text-foreground mt-1 line-clamp-2 group-hover:text-primary transition-colors">
            {article.title}
          </h3>
        </div>
      </a>
    )
  }

  return (
    <div className="flex h-screen bg-background">
      <div
        className={`flex flex-col transition-all duration-300 ${mode === "booking" ? "w-1/2" : "w-full max-w-4xl mx-auto"}`}
      >
        <div className="border-b border-border bg-card">
          <div className="px-6 py-4">
            <h1 className="text-2xl font-semibold text-foreground">Trợ lý Y tế AI</h1>
            <p className="text-sm text-muted-foreground mt-1">Hỗ trợ tư vấn sức khỏe 24/7</p>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="max-w-md space-y-4">
                <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto">
                  <svg className="w-8 h-8 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                    />
                  </svg>
                </div>
                <h2 className="text-xl font-semibold text-foreground">Xin chào! Tôi có thể giúp bạn:</h2>
                <ul className="text-left space-y-2 text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span>Tìm kiếm thông tin y tế (triệu chứng, bệnh lý, điều trị)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span>Đặt lịch khám bệnh</span>
                  </li>
                </ul>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message) => {
                // Clean content: Cắt bỏ toàn bộ phần từ "📚 Nguồn tham khảo" trở đi
                let cleanContent = message.content;
                
                // 1. Cắt từ "📚 Nguồn tham khảo:" đến hết (bao gồm tất cả URLs phía sau)
                cleanContent = cleanContent.replace(/📚\s*Nguồn tham khảo:?[\s\S]*/gi, '');
                
                // 2. Cắt từ "Nguồn tham khảo:" không có emoji (fallback)
                cleanContent = cleanContent.replace(/Nguồn tham khảo:?[\s\S]*/gi, '');
                
                // 3. Remove [ARTICLE] blocks
                cleanContent = cleanContent.replace(/\[ARTICLE\][\s\S]*?\[\/ARTICLE\]/g, '');
                
                // 4. Remove numbered URLs (1. Title\n   https://...)
                cleanContent = cleanContent.replace(/\d+\.\s*[^\n]*\n\s*https?:\/\/[^\s]+/g, '');
                
                // 5. Remove standalone URLs on their own line
                cleanContent = cleanContent.replace(/^\s*https?:\/\/[^\s]+\s*$/gm, '');
                
                // 6. Clean up multiple newlines
                cleanContent = cleanContent.replace(/\n{3,}/g, '\n\n').trim();
                
                return (
                  <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[80%] rounded-lg px-4 py-3 ${
                        message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted text-foreground"
                      }`}
                    >
                      {message.attachments && message.attachments.length > 0 && (
                        <div className="mb-2 space-y-2">
                          {message.attachments.map((attachment, idx) => (
                            <div key={idx} className="flex items-center gap-2">
                              {attachment.type.startsWith("image/") ? (
                                <div className="rounded overflow-hidden max-w-xs">
                                  <img
                                    src={attachment.url || "/placeholder.svg"}
                                    alt={attachment.name}
                                    className="w-full h-auto object-cover"
                                  />
                                </div>
                              ) : (
                                <div className="flex items-center gap-2 bg-background/20 rounded px-2 py-1">
                                  <FileText className="w-4 h-4" />
                                  <span className="text-xs">{attachment.name}</span>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                      
                      {/* Render message content with markdown */}
                      <MarkdownRenderer 
                        content={cleanContent} 
                        className="text-sm leading-relaxed"
                      />

                      {/* Article cards at the bottom */}
                      {message.articles && message.articles.length > 0 && (
                        <div className="mt-4 pt-3 border-t border-foreground/10">
                          <div className="grid grid-cols-3 gap-3">
                            {message.articles.slice(0, 3).map((article, idx) => (
                              <ArticleCard key={idx} article={article} />
                            ))}
                          </div>
                        </div>
                      )}
                      
                      <span className="text-xs opacity-70 mt-1 block">
                        {message.timestamp.toLocaleTimeString("vi-VN", {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                    </div>
                  </div>
                );
              })}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-muted rounded-lg px-4 py-3">
                    <div className="flex gap-1">
                      <div
                        className="w-2 h-2 rounded-full bg-foreground/40 animate-bounce"
                        style={{ animationDelay: "0ms" }}
                      />
                      <div
                        className="w-2 h-2 rounded-full bg-foreground/40 animate-bounce"
                        style={{ animationDelay: "150ms" }}
                      />
                      <div
                        className="w-2 h-2 rounded-full bg-foreground/40 animate-bounce"
                        style={{ animationDelay: "300ms" }}
                      />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="border-t border-border bg-card px-6 py-4">
          {attachments.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-2">
              {attachments.map((file, index) => (
                <div key={index} className="flex items-center gap-2 bg-muted rounded-lg px-3 py-2 text-sm">
                  {file.type.startsWith("image/") ? (
                    <ImageIcon className="w-4 h-4 text-muted-foreground" />
                  ) : (
                    <FileText className="w-4 h-4 text-muted-foreground" />
                  )}
                  <span className="text-foreground max-w-[150px] truncate">{file.name}</span>
                  <button
                    onClick={() => removeAttachment(index)}
                    className="text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="flex gap-2">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*,.pdf,.doc,.docx"
              onChange={handleFileSelect}
              className="hidden"
            />
            <Button
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              size="icon"
              variant="outline"
              className="shrink-0"
            >
              <Paperclip className="w-4 h-4" />
            </Button>
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Nhập tin nhắn của bạn..."
              className="flex-1 bg-background"
              disabled={isLoading}
            />
            <Button
              onClick={handleSendMessage}
              disabled={isLoading || (!input.trim() && attachments.length === 0)}
              size="icon"
              className="shrink-0"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {mode === "booking" && (
        <div className="w-1/2 border-l border-border bg-card overflow-y-auto">
          <BookingForm
            bookingData={bookingData}
            onConfirm={handleBookingConfirm}
            onCancel={handleBookingCancel}
            onUpdate={handleBookingUpdate}
          />
        </div>
      )}
    </div>
  )
}
