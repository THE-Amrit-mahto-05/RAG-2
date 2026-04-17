import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, BookOpen, Bot, Loader2, RotateCcw, Layers, ScrollText } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

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
    text?: string; // We will expand this to show the raw text
  }[];
}

const App: React.FC = () => {
  const [topicId, setTopicId] = useState<string | null>(() => localStorage.getItem('edulevel_topic_id'));
  const [messages, setMessages] = useState<Message[]>(() => {
    const saved = localStorage.getItem('edulevel_messages');
    return saved ? JSON.parse(saved) : [];
  });
  
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  const [, setUploadProgress] = useState<string | null>(null);
  const [activeSources, setActiveSources] = useState<any[]>([]);
  const [toc, setToc] = useState<{page: number, title: string}[]>([]);
  
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (topicId) localStorage.setItem('edulevel_topic_id', topicId);
    else localStorage.removeItem('edulevel_topic_id');
  }, [topicId]);

  useEffect(() => {
    localStorage.setItem('edulevel_messages', JSON.stringify(messages));
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
      const response = await axios.post('/api/upload', formData);
      setTopicId(response.data.id);
      
      const pages = [
        { page: 1, title: 'What is Sound?' },
        { page: 2, title: 'Production of Sound' },
        { page: 3, title: 'Propagation of Sound' },
        { page: 4, title: 'Musical Instruments' }
      ];
      setToc(pages);

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
    } catch (error) {
      console.error('Chat failed:', error);
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
            <div className="space-y-1">
              {toc.map((t, i) => (
                <div key={i} className="flex items-center py-2 text-sm text-slate-600 hover:text-indigo-600 cursor-pointer border-b border-slate-50">
                  <span className="mr-2 text-slate-400 font-medium">{t.page}.</span>
                  <span>{t.title}</span>
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
                  Sound chapter content, topics and visuals will appear after upload.
                </p>
              </div>
              <label className="btn-primary cursor-pointer inline-block">
                {isUploading ? <Loader2 className="animate-spin" /> : 'Upload Sound Chapter'}
                <input type="file" className="hidden" accept=".pdf" onChange={handleUpload} disabled={isUploading} />
              </label>
            </div>
          </div>
        ) : (
          <div className="flex flex-col flex-1 overflow-hidden h-full">
            <div className="flex-1 overflow-y-auto p-8 space-y-8 scrollbar-thin">
              <AnimatePresence>
                {messages.map((msg, idx) => (
                  <motion.div key={idx} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-3xl mx-auto space-y-2">
                    <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                      {msg.role === 'user' ? 'Student (You):' : 'AI Tutor'}
                    </p>
                    <div className="text-slate-700 leading-relaxed text-sm">
                      {msg.content}
                    </div>
                    {msg.image && (
                      <img src={msg.image.url} alt={msg.image.title} className="max-w-md rounded-lg border border-slate-100 shadow-sm mt-4" />
                    )}
                  </motion.div>
                ))}
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
            <div className="flex-1 overflow-y-auto space-y-6 scrollbar-thin">
              {activeSources.map((s, i) => (
                <div key={i} className="space-y-2">
                  <p className="text-[10px] font-bold text-indigo-600 uppercase">Page {s.page}</p>
                  <div className="bg-blue-50/50 border-l-2 border-indigo-400 p-3 text-xs text-slate-600 leading-relaxed">
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
