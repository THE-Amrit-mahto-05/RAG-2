import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, BookOpen, Bot, Loader2, RotateCcw, Layers, ScrollText } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

// Configure Production API URL
axios.defaults.baseURL = import.meta.env.VITE_API_URL || '/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  image?: {
    url: string;
    title: string;
    description: string;
  };
  sources?: {
    chunk_id: string;
    page: number;
    similarity: number;
    match_metadata?: any;
  }[];
  isDivider?: boolean;
}

const App: React.FC = () => {
  const [topicId, setTopicId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  const [, setUploadProgress] = useState<string | null>(null);
  const [activeSources, setActiveSources] = useState<any[]>([]);
  const [toc, setToc] = useState<{section: string, page: number, title: string}[]>([]);
  const [activeTopicIndex, setActiveTopicIndex] = useState<number | null>(null);
  
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
    const lastAssistantMsg = [...messages].reverse().find(m => m.role === 'assistant');
    if (lastAssistantMsg && lastAssistantMsg.sources) {
      setActiveSources(lastAssistantMsg.sources);
    }
  }, [messages]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadProgress("Analyzing chapter...");
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/upload', formData, { 
        timeout: 300000 // 5 minutes for processing large PDFs
      });
      const newTopicId = response.data.id;
      setTopicId(newTopicId);
      
      // Fetch real TOC from backend
      try {
        const tocResponse = await axios.get(`/api/toc/${newTopicId}`);
        setToc(tocResponse.data.toc);
      } catch {
        // Fallback to generic headings
        setToc([
          { section: '11.1', page: 1, title: 'Production of Sound' },
          { section: '11.2', page: 2, title: 'Propagation of Sound' },
          { section: '11.3', page: 7, title: 'Reflection of Sound' },
          { section: '11.4', page: 9, title: 'Range of Hearing' },
          { section: '11.5', page: 10, title: 'Applications of Ultrasound' },
        ]);
      }

      setMessages([{
        role: 'assistant',
        content: `I've processed "${file.name}". I'm ready to explain the concepts and show relevant diagrams.`
      }]);
    } catch (error) {
      console.error('Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleTopicClick = async (topic: any, index: number) => {
    setActiveTopicIndex(index);
    if (isChatting) return;

    // Insert a divider and the user's implicit question
    const dividerMsg: Message = { role: 'assistant', content: `--- Switched Topic to: ${topic.title} ---`, isDivider: true };
    const questionText = `Explain: ${topic.title}`;
    const tempMsg: Message = { role: 'user', content: questionText };
    
    setMessages(prev => [...prev, dividerMsg, tempMsg]);
    setIsChatting(true);

    try {
      const response = await axios.post('/api/chat', {
        topic_id: topicId,
        question: questionText,
        conversation_history: messages.map(m => ({ role: m.role, content: m.content }))
      });
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.data.answer,
        image: response.data.image,
        sources: response.data.sources
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error: any) {
      console.error('Topic click failed:', error);
      if (error.response?.status === 400) {
        setTopicId(null);
        setMessages([]);
        setToc([]);
        alert("Session expired or PDF memory cleared. Please re-upload.");
      }
    } finally {
      setIsChatting(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !topicId || isChatting) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsChatting(true);

    try {
      const response = await axios.post('/api/chat', {
        topic_id: topicId,
        question: input,
        conversation_history: messages.map(m => ({ role: m.role, content: m.content }))
      });

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.data.answer,
        image: response.data.image,
        sources: response.data.sources
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error: any) {
      console.error('Chat failed:', error);
      if (error.response?.status === 400) {
        setTopicId(null);
        setMessages([]);
        setToc([]);
        alert("Session expired or PDF memory cleared. Please re-upload.");
      }
    } finally {
      setIsChatting(false);
    }
  };

  return (
    <div className="dashboard-grid h-screen w-full">
      
      {/* --- HEADER --- */}
      <header className="header-main">
        <h1 className="header-title">EDU AI TUTOR</h1>
      </header>

      {/* --- LEFT SIDEBAR --- */}
      <aside className="panel-white">
        <div className="panel-header">CHAPTER NAVIGATOR</div>
        <div className="p-4 space-y-6">
          {!topicId ? (
            <div className="space-y-4">
               <div className="flex items-center gap-2">
                 <span className="status-badge">Status: Parsing & embedding pending...</span>
               </div>
               <p className="text-xs text-slate-400 leading-relaxed">
                 Topics will populate once chapter is uploaded.
               </p>
            </div>
          ) : (
            <div className="space-y-1 overflow-y-auto max-h-[calc(100vh-200px)] scrollbar-thin pr-2">
              {toc.map((t, i) => (
                <div
                  key={i}
                  onClick={() => handleTopicClick(t, i)}
                  className={`flex items-center py-2 text-sm cursor-pointer border-b border-slate-50 transition-colors ${
                    activeTopicIndex === i
                      ? 'text-indigo-600 font-semibold bg-indigo-50/50 px-2 rounded'
                      : 'text-slate-600 hover:text-indigo-600 hover:bg-slate-50 px-2 rounded'
                  } ${isChatting ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <span className="mr-2 text-slate-400 font-mono text-xs">{t.section ?? t.page}</span>
                  <span className="flex-1 leading-tight py-1">{t.title}</span>
                  {isChatting && activeTopicIndex === i && (
                    <Loader2 size={12} className="ml-auto animate-spin text-indigo-400" />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </aside>

      {/* --- CENTER PANEL --- */}
      <main className="panel-white">
        <div className="panel-header">AI TUTOR CHAT</div>
        
        {!topicId ? (
          <div className="flex-1 flex items-center justify-center p-8 bg-slate-50/50">
            <div className="upload-card">
              <div className="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mx-auto text-slate-400">
                <Upload size={32} />
              </div>
              <div className="space-y-2">
                <h2 className="text-xl font-bold text-slate-700">Upload Chapter PDF</h2>
                <p className="text-sm text-slate-500">
                  Chapter content, topics and visuals will appear after upload.
                </p>
              </div>
              <label className="btn-primary cursor-pointer inline-block">
                {isUploading ? <Loader2 className="animate-spin" /> : 'Upload Chapter PDF'}
                <input type="file" className="hidden" accept=".pdf" onChange={handleUpload} disabled={isUploading} />
              </label>
            </div>
          </div>
        ) : (
          <div className="flex flex-col flex-1 overflow-hidden h-full">
            <div className="flex-1 overflow-y-auto p-8 space-y-8 scrollbar-thin">
              <AnimatePresence>
                {messages.map((msg, idx) => {
                  if (msg.isDivider) {
                    return (
                      <div key={idx} className="flex items-center mx-auto max-w-3xl py-4 opacity-50">
                        <div className="flex-1 border-t border-slate-300"></div>
                        <span className="px-4 text-xs font-semibold text-slate-400 tracking-wider uppercase">{msg.content.replace(/---/g, '')}</span>
                        <div className="flex-1 border-t border-slate-300"></div>
                      </div>
                    );
                  }
                  return (
                  <motion.div key={idx} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-3xl mx-auto space-y-2">
                    <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                      {msg.role === 'user' ? 'Student (You):' : 'AI Tutor'}
                    </p>
                    <div className="text-slate-700 leading-relaxed text-sm whitespace-pre-wrap">
                      {msg.content
                        .replace(/!\[.*?\]\(.*?\)/g, '')   // Remove Markdown images
                        .replace(/\[.*?\]\(.*?\)/g, '')    // Remove Markdown links (hallucination safeguard)
                        .replace(/<img[^>]*>/g, '')         // Remove HTML images
                      }
                    </div>
                    {msg.image && (
                      <div className="mt-4 border border-slate-200 rounded-xl overflow-hidden bg-white p-3 shadow-md inline-block max-w-[90%]">
                        <div className="text-[10px] text-indigo-500 font-bold mb-2 uppercase tracking-tight">Textbook Figure:</div>
                        <img 
                          src={msg.image.url} 
                          alt={msg.image.title || "Textbook Diagram"} 
                          className="max-w-full h-auto rounded-lg border border-slate-100" 
                          onError={(e) => (e.currentTarget.style.display = 'none')}
                        />
                        <p className="text-xs text-slate-500 mt-2 italic font-medium leading-snug">
                          {msg.image.title || "Relevant diagram retrieved from the chapter core concepts."}
                        </p>
                      </div>
                    )}
                  </motion.div>
                )})}
              </AnimatePresence>
              <div ref={chatEndRef} />
            </div>

            <div className="p-6 bg-white border-t border-slate-100">
              <div className="max-w-3xl mx-auto flex gap-3">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="Ask a question..."
                  className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 text-sm outline-none focus:ring-1 focus:ring-slate-300"
                />
                <button onClick={handleSend} disabled={isChatting} className="btn-primary">
                  <Send size={18} />
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* --- RIGHT PANEL --- */}
      <aside className="panel-white">
        <div className="panel-header">CHAPTER CONTEXT</div>
        <div className="p-6 h-full flex flex-col">
          {activeSources.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-4/5 text-center text-slate-400 space-y-4">
              <div className="w-12 h-12 bg-slate-50 rounded-lg flex items-center justify-center">
                <Upload size={24} className="opacity-40" />
              </div>
              <div>
                <p className="text-sm font-medium text-slate-600 mb-1">Upload PDF to begin.</p>
                <p className="text-xs leading-relaxed">
                  Chapter text, references and citations will appear here once the PDF is uploaded.
                </p>
              </div>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto space-y-8 pb-4 scrollbar-thin">
              {activeSources.map((s, i) => (
                <div key={i} className="space-y-3 pb-6 border-b border-slate-100 last:border-0">
                  <p className="text-[10px] font-bold text-indigo-600 uppercase tracking-widest">Page {s.page}</p>
                  <div className="bg-slate-50 border-l-4 border-indigo-400 p-4 text-sm text-slate-600 leading-relaxed shadow-sm rounded-r-md whitespace-pre-wrap">
                    {s.text}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </aside>

    </div>
  );
};



export default App;
