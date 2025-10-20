"use client"

import React, { type ReactElement } from 'react'

interface MarkdownRendererProps {
  content: string
  className?: string
}

export function MarkdownRenderer({ content, className = "" }: MarkdownRendererProps) {
  // Simple markdown parser
  const renderMarkdown = (text: string) => {
    const lines = text.split('\n')
    const elements: ReactElement[] = []
    let key = 0

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]
      
      // Skip empty lines
      if (!line.trim()) {
        elements.push(<br key={`br-${key++}`} />)
        continue
      }

      // H2 (## Title)
      if (line.startsWith('## ')) {
        const text = line.replace('## ', '')
        elements.push(
          <h2 key={`h2-${key++}`} className="text-lg font-bold mt-4 mb-2">
            {renderInline(text)}
          </h2>
        )
        continue
      }

      // H3 (### Title)
      if (line.startsWith('### ')) {
        const text = line.replace('### ', '')
        elements.push(
          <h3 key={`h3-${key++}`} className="text-base font-semibold mt-3 mb-2">
            {renderInline(text)}
          </h3>
        )
        continue
      }

      // Bullet points (- or *)
      if (line.match(/^\s*[-*]\s+/)) {
        const text = line.replace(/^\s*[-*]\s+/, '')
        elements.push(
          <div key={`bullet-${key++}`} className="flex gap-2 ml-4 mb-1">
            <span className="text-primary mt-1">•</span>
            <span>{renderInline(text)}</span>
          </div>
        )
        continue
      }

      // Numbered list (1. 2. 3.)
      if (line.match(/^\s*\d+\.\s+/)) {
        const match = line.match(/^\s*(\d+)\.\s+(.*)/)
        if (match) {
          const number = match[1]
          const text = match[2]
          elements.push(
            <div key={`num-${key++}`} className="flex gap-2 ml-4 mb-1">
              <span className="font-semibold">{number}.</span>
              <span>{renderInline(text)}</span>
            </div>
          )
          continue
        }
      }

      // Regular paragraph
      elements.push(
        <div key={`p-${key++}`} className="mb-2">
          {renderInline(line)}
        </div>
      )
    }

    return elements
  }

  const renderInline = (text: string) => {
    const parts: (string | ReactElement)[] = []
    let remaining = text
    let keyInline = 0

    // Bold (**text**)
    const boldRegex = /\*\*([^*]+)\*\*/g
    let lastIndex = 0
    let match

    while ((match = boldRegex.exec(text)) !== null) {
      // Add text before match
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index))
      }
      // Add bold text
      parts.push(
        <strong key={`bold-${keyInline++}`} className="font-semibold">
          {match[1]}
        </strong>
      )
      lastIndex = match.index + match[0].length
    }

    // Add remaining text
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex))
    }

    return parts.length > 0 ? parts : text
  }

  return (
    <div className={`markdown-content ${className}`}>
      {renderMarkdown(content)}
    </div>
  )
}
