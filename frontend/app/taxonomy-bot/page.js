'use client'
import React, { useState, useRef, useEffect } from 'react'
import { Plus, SendHorizontal, X, Image, FileText, Menu, Edit3, Trash2, MessageSquare, ChevronRight, Sparkles } from 'lucide-react'

const MarineChatbot = () => {
  const [conversations, setConversations] = useState([
    { id: 1, title: 'New Conversation', messages: [], timestamp: new Date() }
  ])
  const [currentConvId, setCurrentConvId] = useState(1)
  const [input, setInput] = useState('')
  const [attachments, setAttachments] = useState([])
  const [showFileMenu, setShowFileMenu] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const fileInputRef = useRef(null)
  const imageInputRef = useRef(null)
  const messagesEndRef = useRef(null)

  // Matches your backend endpoint exactly
  const API_BASE_URL = 'http://localhost:8001/taxonomyChat'

  const currentConv = conversations.find(c => c.id === currentConvId)
  const messages = currentConv?.messages || []

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isTyping])

  const createNewConversation = () => {
    const newConv = {
      id: Date.now(),
      title: 'New Conversation',
      messages: [],
      timestamp: new Date()
    }
    setConversations([newConv, ...conversations])
    setCurrentConvId(newConv.id)
  }

  const deleteConversation = (id, e) => {
    e.stopPropagation()
    if (conversations.length === 1) return
    const filtered = conversations.filter(c => c.id !== id)
    setConversations(filtered)
    if (currentConvId === id) {
      setCurrentConvId(filtered[0].id)
    }
  }

  const updateConversationTitle = (id, firstMessage) => {
    setConversations(convs =>
      convs.map(c =>
        c.id === id && c.title === 'New Conversation'
          ? { ...c, title: firstMessage.slice(0, 30) + (firstMessage.length > 30 ? '...' : '') }
          : c
      )
    )
  }

  const handleFileSelect = (type) => {
    if (type === 'image') {
      imageInputRef.current?.click()
    } else {
      fileInputRef.current?.click()
    }
    setShowFileMenu(false)
  }

  const handleFileChange = (e, type) => {
    const files = Array.from(e.target.files)
    const newAttachments = files.map(file => ({
      file,
      type,
      preview: type === 'image' ? URL.createObjectURL(file) : null,
      name: file.name
    }))
    setAttachments([...attachments, ...newAttachments])
  }

  const removeAttachment = (index) => {
    const newAttachments = [...attachments]
    if (newAttachments[index].preview) {
      URL.revokeObjectURL(newAttachments[index].preview)
    }
    newAttachments.splice(index, 1)
    setAttachments(newAttachments)
  }

  const handleSend = async () => {
    if (!input.trim() && attachments.length === 0) return

    // 1. Prepare User Message
    const userMessage = {
      id: Date.now(),
      text: input,
      sender: 'user',
      attachments: [...attachments],
      timestamp: new Date()
    }

    // 2. Prepare Placeholder Bot Message (for streaming)
    const botMessageId = Date.now() + 1
    const botMessage = {
      id: botMessageId,
      text: '', // Start empty
      sender: 'bot',
      timestamp: new Date(),
      data: null
    }

    // 3. Update State with User Message AND Empty Bot Message
    setConversations(convs =>
      convs.map(c =>
        c.id === currentConvId
          ? { ...c, messages: [...c.messages, userMessage, botMessage], timestamp: new Date() }
          : c
      )
    )

    if (currentConv.messages.length === 0) {
      updateConversationTitle(currentConvId, input)
    }

    // 4. Reset Inputs
    const messageText = input
    const currentAttachments = [...attachments]
    setInput('')
    setAttachments([])
    setIsTyping(true)

    try {
      const formData = new FormData()
      // Backend expects 'user_input' based on: user_input: str = Form(...)
      formData.append('user_input', messageText)
      
      currentAttachments.forEach((attachment) => {
        formData.append('files', attachment.file)
      })

      // 5. Fetch with Streaming
      const response = await fetch(API_BASE_URL, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      // 6. Handle the Stream
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let done = false

      while (!done) {
        const { value, done: doneReading } = await reader.read()
        done = doneReading
        
        if (value) {
          const chunk = decoder.decode(value, { stream: true })
          
          // Update the specific bot message in the state with the new chunk
          setConversations(prevConvs => 
            prevConvs.map(conv => {
              if (conv.id === currentConvId) {
                const updatedMessages = conv.messages.map(msg => {
                  if (msg.id === botMessageId) {
                    return { ...msg, text: msg.text + chunk }
                  }
                  return msg
                })
                return { ...conv, messages: updatedMessages }
              }
              return conv
            })
          )
        }
      }

    } catch (error) {
      console.error('Error sending message:', error)
      
      // Update the bot message to show error
      setConversations(convs =>
        convs.map(c =>
          c.id === currentConvId
            ? { 
                ...c, 
                messages: c.messages.map(msg => 
                  msg.id === botMessageId 
                    ? { ...msg, text: "I'm having trouble connecting right now. Please try again.", isError: true }
                    : msg
                )
              }
            : c
        )
      )
    } finally {
      setIsTyping(false)
    }
  }

  const examplePrompts = [
    { icon: "🐋", text: "What are the largest marine mammals?", gradient: "from-blue-500 to-cyan-500" },
    { icon: "🐠", text: "Identify this fish species", gradient: "from-purple-500 to-pink-500" },
    { icon: "🌊", text: "Explain coral reef ecosystems", gradient: "from-teal-500 to-emerald-500" },
    { icon: "🦈", text: "Tell me about shark behavior", gradient: "from-indigo-500 to-blue-500" }
  ]

  return (
    <div className="flex h-screen bg-[#0f172a] text-gray-100 overflow-hidden">
      
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 bg-[#1e293b] border-r border-gray-800 flex flex-col overflow-hidden`}>
        <div className="p-4 border-b border-gray-800">
          <button
            onClick={createNewConversation}
            className="w-full flex items-center justify-center space-x-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg px-4 py-3 transition-all duration-200 shadow-lg hover:shadow-cyan-500/50"
          >
            <Edit3 className="w-4 h-4" />
            <span className="font-medium">New Chat</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {conversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => setCurrentConvId(conv.id)}
              className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer transition-all duration-200 ${
                currentConvId === conv.id
                  ? 'bg-gray-800 shadow-lg'
                  : 'hover:bg-gray-800/50'
              }`}
            >
              <div className="flex items-center space-x-3 flex-1 min-w-0">
                <MessageSquare className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                <span className="text-sm truncate text-gray-300">{conv.title}</span>
              </div>
              {conversations.length > 1 && (
                <button
                  onClick={(e) => deleteConversation(conv.id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded transition-opacity"
                >
                  <Trash2 className="w-3.5 h-3.5 text-red-400" />
                </button>
              )}
            </div>
          ))}
        </div>

        <div className="p-4 border-t border-gray-800 text-xs text-gray-500">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <span>Marine Species AI</span>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        
        {/* Top Bar */}
        <div className="h-14 border-b border-gray-800 flex items-center px-4 bg-[#1e293b]/50 backdrop-blur">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="ml-4">
            <h2 className="text-sm font-medium text-gray-300">Marine Species Assistant</h2>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center p-8 max-w-3xl mx-auto">
              <div className="mb-8 relative">
                <div className="w-16 h-16 bg-gradient-to-br from-cyan-500 to-blue-500 rounded-2xl flex items-center justify-center shadow-2xl">
                  <Sparkles className="w-8 h-8 text-white" />
                </div>
                <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-emerald-500 rounded-full border-2 border-[#0f172a]"></div>
              </div>
              
              <h1 className="text-3xl font-bold mb-3 bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                Welcome to Marine AI
              </h1>
              <p className="text-gray-400 text-center mb-12 max-w-md">
                Your intelligent assistant for marine biology, species identification, and ocean ecosystems
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
                {examplePrompts.map((prompt, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setInput(prompt.text)
                      setTimeout(() => handleSend(), 100) // Small delay to allow state to settle
                    }}
                    className="group relative overflow-hidden bg-gray-800/50 hover:bg-gray-800 border border-gray-700 hover:border-gray-600 rounded-xl p-4 text-left transition-all duration-200 hover:scale-[1.02]"
                  >
                    <div className={`absolute inset-0 bg-gradient-to-br ${prompt.gradient} opacity-0 group-hover:opacity-5 transition-opacity`}></div>
                    <div className="relative flex items-start space-x-3">
                      <span className="text-2xl">{prompt.icon}</span>
                      <span className="text-sm text-gray-300 flex-1">{prompt.text}</span>
                      <ChevronRight className="w-4 h-4 text-gray-600 group-hover:text-cyan-400 transition-colors" />
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto w-full px-4 py-8 space-y-6">
              {messages.map((message, index) => (
                <div key={index} className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} animate-slideIn`}>
                  <div className={`flex items-start space-x-3 max-w-[85%] ${message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      message.sender === 'user' 
                        ? 'bg-gradient-to-br from-purple-500 to-pink-500' 
                        : 'bg-gradient-to-br from-cyan-500 to-blue-500'
                    }`}>
                      {message.sender === 'user' ? (
                        <span className="text-xs font-bold">You</span>
                      ) : (
                        <Sparkles className="w-4 h-4" />
                      )}
                    </div>
                    
                    <div className={`flex-1 flex-row-reverse`}>
                      <div className={`inline-block rounded-2xl px-4 py-3 ${
                        message.sender === 'user'
                          ? 'bg-gradient-to-br from-cyan-600 to-blue-600 text-white'
                          : message.isError
                          ? 'bg-red-500/20 border border-red-500/50 text-red-200'
                          : 'bg-gray-800 text-gray-100'
                      }`}>
                        {message.attachments && message.attachments.length > 0 && (
                          <div className="mb-3 space-y-2">
                            {message.attachments.map((att, i) => (
                              <div key={i} className="bg-black/20 rounded-lg p-2 flex items-center space-x-2">
                                {att.type === 'image' ? (
                                  <>
                                    <img src={att.preview} alt={att.name} className="w-12 h-12 object-cover rounded" />
                                    <span className="text-xs truncate">{att.name}</span>
                                  </>
                                ) : (
                                  <>
                                    <FileText className="w-6 h-6 text-blue-400" />
                                    <span className="text-xs truncate">{att.name}</span>
                                  </>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                        
                        {/* Only show text if it exists (avoids empty bubbles during initial load) */}
                        {message.text && (
                          <p className="whitespace-pre-wrap text-sm leading-relaxed text-left">
                            {message.text.split(/(\*\*.*?\*\*)/g).map((part, i) => {
                              if (part.startsWith('**') && part.endsWith('**')) {
                                return <strong key={i} className="font-bold text-white">{part.slice(2, -2)}</strong>;
                              }
                              return part;
                            })}
                          </p>
                        )}
                       
                        {/* Show typing indicator INSIDE bubble if empty and typing */}
                        {(!message.text && isTyping && message.sender === 'bot') && (
                           <span className="inline-flex space-x-1">
                             <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                             <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                             <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                           </span>
                        )}

                      </div>
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-800 bg-[#1e293b]/50 backdrop-blur">
          <div className="max-w-3xl mx-auto p-4">
            {attachments.length > 0 && (
              <div className="mb-3 flex flex-wrap gap-2">
                {attachments.map((att, index) => (
                  <div key={index} className="relative group">
                    <div className="bg-gray-800 rounded-lg p-2 pr-8 flex items-center space-x-2 border border-gray-700">
                      {att.type === 'image' ? (
                        <>
                          <img src={att.preview} alt={att.name} className="w-10 h-10 object-cover rounded" />
                          <span className="text-xs text-gray-300 truncate max-w-[100px]">{att.name}</span>
                        </>
                      ) : (
                        <>
                          <FileText className="w-6 h-6 text-cyan-400" />
                          <span className="text-xs text-gray-300 truncate max-w-[100px]">{att.name}</span>
                        </>
                      )}
                    </div>
                    <button
                      onClick={() => removeAttachment(index)}
                      className="absolute -top-2 -right-2 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-lg"
                    >
                      <X className="w-3 h-3 text-white" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            
            <div className="flex items-end space-x-3">
              <div className="relative">
                <button
                  onClick={() => setShowFileMenu(!showFileMenu)}
                  className="w-10 h-10 bg-gray-800 hover:bg-gray-700 rounded-lg flex items-center justify-center transition-colors border border-gray-700"
                >
                  <Plus className="w-5 h-5" />
                </button>
                
                {showFileMenu && (
                  <div className="absolute bottom-full mb-2 left-0 bg-gray-800 rounded-xl shadow-2xl overflow-hidden border border-gray-700 animate-slideUp">
                    <button
                      onClick={() => handleFileSelect('image')}
                      className="flex items-center space-x-3 px-4 py-3 hover:bg-gray-700 transition-colors w-full text-left"
                    >
                      <Image className="w-4 h-4 text-cyan-400" />
                      <span className="text-sm">Upload Image</span>
                    </button>
                    <button
                      onClick={() => handleFileSelect('file')}
                      className="flex items-center space-x-3 px-4 py-3 hover:bg-gray-700 transition-colors w-full text-left"
                    >
                      <FileText className="w-4 h-4 text-emerald-400" />
                      <span className="text-sm">Upload Document</span>
                    </button>
                  </div>
                )}
              </div>

              <input type="file" ref={imageInputRef} onChange={(e) => handleFileChange(e, 'image')} accept="image/*" className="hidden" multiple />
              <input type="file" ref={fileInputRef} onChange={(e) => handleFileChange(e, 'file')} accept=".pdf,.doc,.docx,.txt" className="hidden" multiple />

              <div className="flex-1 relative">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleSend()
                    }
                  }}
                  placeholder="Ask about marine species..."
                  rows={1}
                  className="w-full bg-gray-800 text-white rounded-xl px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-cyan-500 border border-gray-700 placeholder-gray-500 resize-none max-h-32"
                  style={{ minHeight: '44px' }}
                />
              </div>

              <button
                onClick={handleSend}
                disabled={!input.trim() && attachments.length === 0}
                className="w-10 h-10 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:from-gray-700 disabled:to-gray-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center transition-all duration-200 shadow-lg hover:shadow-cyan-500/50 disabled:shadow-none"
              >
                <SendHorizontal className="w-5 h-5" />
              </button>
            </div>

            <p className="text-xs text-gray-500 mt-3 text-center">
              Marine AI can make mistakes. Verify important information.
            </p>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-slideIn { animation: slideIn 0.3s ease-out; }
        .animate-slideUp { animation: slideUp 0.2s ease-out; }
      `}</style>
    </div>
  )
}

export default MarineChatbot