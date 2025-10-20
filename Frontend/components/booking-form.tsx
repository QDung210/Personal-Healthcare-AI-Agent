"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card } from "@/components/ui/card"
import { Download, X, Calendar, Clock, Hospital, User } from "lucide-react"

type BookingData = {
  patient_name: string
  appointment_type: string
  time: string
  hospital: string
}

type BookingFormProps = {
  bookingData: BookingData | null
  onConfirm: () => void
  onCancel: () => void
  onUpdate: (data: BookingData) => void
}

export default function BookingForm({ bookingData, onConfirm, onCancel, onUpdate }: BookingFormProps) {
  const [formData, setFormData] = useState<BookingData>({
    patient_name: "",
    appointment_type: "",
    time: "",
    hospital: "",
  })

  useEffect(() => {
    if (bookingData) {
      setFormData({
        patient_name: bookingData.patient_name || "",
        appointment_type: bookingData.appointment_type || "",
        time: bookingData.time || "",
        hospital: bookingData.hospital || "",
      })
    }
  }, [bookingData])

  const handleInputChange = (field: keyof BookingData, value: string) => {
    const newData = { ...formData, [field]: value }
    setFormData(newData)
    onUpdate(newData)
  }

  const handleDownload = () => {
    // Create appointment ticket content
    const ticketContent = `
╔════════════════════════════════════════════════╗
║         PHIẾU ĐẶT LỊCH KHÁM BỆNH              ║
╚════════════════════════════════════════════════╝

Thông tin bệnh nhân:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Họ và tên:        ${formData.patient_name}
Loại khám:        ${formData.appointment_type}
Thời gian:        ${formData.time}
Bệnh viện:        ${formData.hospital}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ghi chú:
• Vui lòng có mặt trước 15 phút
• Mang theo CMND/CCCD và sổ khám bệnh (nếu có)
• Liên hệ hotline để thay đổi lịch hẹn

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ngày tạo: ${new Date().toLocaleDateString("vi-VN")}
Mã phiếu: ${Date.now().toString().slice(-8)}

    `.trim()

    // Create and download file
    const blob = new Blob([ticketContent], { type: "text/plain;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = `phieu-kham-${Date.now()}.txt`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)

    // Confirm and close
    onConfirm()
  }

  const isFormComplete = formData.patient_name && formData.appointment_type && formData.time && formData.hospital

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b border-border px-6 py-4 flex items-center justify-between bg-card">
        <h2 className="text-xl font-semibold text-foreground">Phiếu Đặt Lịch Khám</h2>
        <Button variant="ghost" size="icon" onClick={onCancel}>
          <X className="w-5 h-5" />
        </Button>
      </div>

      {/* Form Content */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <Card className="p-6 space-y-6 bg-background border-border">
          {/* Patient Name */}
          <div className="space-y-2">
            <Label htmlFor="patient_name" className="flex items-center gap-2 text-foreground">
              <User className="w-4 h-4" />
              Họ và tên bệnh nhân
            </Label>
            <Input
              id="patient_name"
              value={formData.patient_name}
              onChange={(e) => handleInputChange("patient_name", e.target.value)}
              placeholder="Nhập họ và tên"
              className="bg-card"
            />
          </div>

          {/* Appointment Type */}
          <div className="space-y-2">
            <Label htmlFor="appointment_type" className="flex items-center gap-2 text-foreground">
              <Calendar className="w-4 h-4" />
              Loại khám
            </Label>
            <Input
              id="appointment_type"
              value={formData.appointment_type}
              onChange={(e) => handleInputChange("appointment_type", e.target.value)}
              placeholder="Ví dụ: Khám tổng quát, Khám chuyên khoa..."
              className="bg-card"
            />
          </div>

          {/* Time */}
          <div className="space-y-2">
            <Label htmlFor="time" className="flex items-center gap-2 text-foreground">
              <Clock className="w-4 h-4" />
              Thời gian
            </Label>
            <Input
              id="time"
              value={formData.time}
              onChange={(e) => handleInputChange("time", e.target.value)}
              placeholder="Ví dụ: 9h sáng, 14h chiều..."
              className="bg-card"
            />
          </div>

          {/* Hospital */}
          <div className="space-y-2">
            <Label htmlFor="hospital" className="flex items-center gap-2 text-foreground">
              <Hospital className="w-4 h-4" />
              Bệnh viện
            </Label>
            <Input
              id="hospital"
              value={formData.hospital}
              onChange={(e) => handleInputChange("hospital", e.target.value)}
              placeholder="Nhập tên bệnh viện"
              className="bg-card"
            />
          </div>

          {/* Notes */}
          <div className="pt-4 border-t border-border">
            <h3 className="font-semibold text-foreground mb-3">Ghi chú quan trọng:</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="text-primary mt-0.5">•</span>
                <span>Vui lòng có mặt trước 15 phút để làm thủ tục</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-0.5">•</span>
                <span>Mang theo CMND/CCCD và sổ khám bệnh (nếu có)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-0.5">•</span>
                <span>Liên hệ hotline để thay đổi hoặc hủy lịch hẹn</span>
              </li>
            </ul>
          </div>
        </Card>
      </div>

      {/* Actions */}
      <div className="border-t border-border px-6 py-4 bg-card">
        <div className="flex gap-3">
          <Button variant="outline" onClick={onCancel} className="flex-1 bg-transparent">
            Hủy
          </Button>
          <Button onClick={handleDownload} disabled={!isFormComplete} className="flex-1 gap-2">
            <Download className="w-4 h-4" />
            Xác nhận & Tải phiếu
          </Button>
        </div>
      </div>
    </div>
  )
}
