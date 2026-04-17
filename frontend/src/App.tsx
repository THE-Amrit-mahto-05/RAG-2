import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, BookOpen, Bot, User, CheckCircle2, Loader2, Image as ImageIcon, RotateCcw, ChevronRight, Hash, Layers, Layout, ScrollText, Binary } from 'lucide-react';
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
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
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
    // Update right panel context when messages change
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
    setUploadProgress("Deconstructing chapter...");
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/upload', formData);
      setTopicId(response.data.id);
      
      // Auto-generate a basic TOC for navigation
      const pages = Array.from({length: Math.ceil(response.data.chunk_count/5)}, (_, i) => ({
        page: i + 1,
        title: `Section ${i + 1}`
      }));
      setToc(pages);

      setMessages([{
        role: 'assistant',
        content: `Dashboard ready. I've indexed "${file.name}" with my neural search engine. \n\nI'm ready to explain complex concepts and show you relevant diagrams. What should we start with?`
      }]);
    } catch (error) {
      setUploadProgress("Upload failed.");
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
    <div className="relative h-screen w-full overflow-hidden">
      {/* Background Aura */}
      <div className="aura-bg">
        <div className="aura-blob blob-1" />
        <div className="aura-blob blob-2" />
        <div className="aura-blob blob-3" />
      </div>

      <div className="dashboard-grid relative z-10">
        
        {/* --- LEFT PANEL: Navigation --- */}
        <aside className="glass-panel p-6">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
              <BookOpen className="text-white" size={20} />
            </div>
            <h1 className="text-xl font-bold tracking-tight">Edulevel</h1>
          </div>

          <div className="flex-1 overflow-y-auto scrollbar-thin flex flex-col gap-6">
            {!topicId ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-slate-500">
                <Layout size={40} className="mb-4 opacity-20" />
                <p className="text-sm">Upload a chapter to see the navigation</p>
              </div>
            ) : (
              <>
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-widest text-indigo-400 mb-4 flex items-center gap-2">
                    <ScrollText size={14} /> Chapter Map
                  </h3>
                  <div className="space-y-2">
                    {toc.map((t, i) => (
                      <button key={i} className="w-full flex items-center justify-between p-3 rounded-xl bg-white/5 border border-transparent hover:border-indigo-500/30 hover:bg-white/10 transition-all text-sm group">
                        <span className="text-slate-300 font-medium">{t.title}</span>
                        <span className="text-[10px] bg-slate-800 text-slate-500 px-1.5 rounded font-bold group-hover:text-indigo-400 transition-colors">P.{t.page}</span>
                      </button>
                    ))}
                  </div>
                </div>
                
                <div className="mt-auto">
                    <button onClick={() => setTopicId(null)} className="flex items-center gap-2 text-slate-500 hover:text-white text-xs font-bold transition-colors">
                      <RotateCcw size={14} /> Reset Session
                    </button>
                </div>
              </>
            )}
          </div>
        </aside>

        {/* --- CENTER PANEL: Chat --- */}
        <main className="glass-panel flex flex-col relative">
          {!topicId ? (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <div className="w-24 h-24 bg-indigo-600/10 rounded-3xl flex items-center justify-center mb-8 mx-auto border border-indigo-600/20">
                  <Upload className="text-indigo-500" size={40} />
                </div>
                <h2 className="text-3xl font-bold mb-4">Start your Study Session</h2>
                <p className="text-slate-400 mb-10 max-w-sm mx-auto">Upload any PDF textbook chapter to begin an interactive tutoring session grounded in your material.</p>
                <label className="btn-primary py-4 px-8 text-lg cursor-pointer mx-auto">
                  {isUploading ? <Loader2 className="animate-spin" /> : <Upload />}
                  <span>{isUploading ? uploadProgress : 'Select Textbook PDF'}</span>
                  <input type="file" className="hidden" accept=".pdf" onChange={handleUpload} disabled={isUploading} />
                </label>
              </motion.div>
            </div>
          ) : (
            <>
              <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-8 scroll-smooth scrollbar-thin">
                <AnimatePresence>
                  {messages.map((msg, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, x: msg.role === 'user' ? 20 : -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`max-w-[90%] md:max-w-[85%] flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 shadow-lg ${
                          msg.role === 'user' ? 'bg-indigo-600' : 'bg-slate-800'
                        }`}>
                          {msg.role === 'user' ? <User size={20} /> : <Bot size={20} className="text-indigo-400" />}
                        </div>
                        <div className={`space-y-4 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                          <div className={`p-5 rounded-2xl ${
                            msg.role === 'user' 
                              ? 'bg-indigo-600 text-white rounded-tr-none' 
                              : 'bg-white/5 text-slate-200 border border-white/10 backdrop-blur-md rounded-tl-none'
                          }`}>
                            <p className="leading-relaxed font-medium">{msg.content}</p>
                          </div>
                          
                          {msg.image && (
                            <div className="rounded-2xl border border-white/10 bg-black/40 p-2 overflow-hidden shadow-2xl max-w-sm">
                              <img src={msg.image.url} alt={msg.image.title} className="w-full rounded-xl" />
                              <div className="p-3 text-xs flex gap-2">
                                <ImageIcon size={14} className="text-indigo-400" />
                                <div>
                                  <p className="font-bold text-white">{msg.image.title}</p>
                                  <p className="text-slate-400 mt-1">{msg.image.description}</p>
                                </div>
                              </div>
                            </div>
                          )}
                          
                          {msg.sources && (
                            <div className="flex gap-2">
                              {msg.sources.map((s, i) => (
                                <span key={i} className="text-[10px] font-bold bg-indigo-500/10 text-indigo-400 px-2 py-1 rounded-lg border border-indigo-500/20">
                                  Page {s.page}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
                {isChatting && (
                  <div className="flex gap-4">
                    <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center"><Bot size={20} className="text-indigo-400" /></div>
                    <div className="bg-white/5 p-5 rounded-2xl rounded-tl-none border border-white/10"><Loader2 className="animate-spin text-indigo-400" /></div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Chat Input */}
              <div className="p-6 border-t border-white/5">
                <div className="flex gap-3 relative">
                  <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                    placeholder="Ask a question about this chapter..."
                    className="flex-1 bg-white/5 border border-white/10 rounded-2xl px-5 py-4 focus:ring-2 focus:ring-indigo-500/50 outline-none transition-all placeholder:text-slate-600"
                  />
                  <button onClick={handleSend} disabled={isChatting} className="btn-primary !p-4">
                    <Send size={24} />
                  </button>
                </div>
              </div>
            </>
          )}
        </main>

        {/* --- RIGHT PANEL: Grounding/Context --- */}
        <aside className="glass-panel p-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-indigo-400 mb-6 flex items-center gap-2">
            <Binary size={14} /> Source Grounding
          </h3>
          
          <div className="flex-1 overflow-y-auto scrollbar-thin space-y-4">
            {activeSources.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-slate-500">
                <Layers size={40} className="mb-4 opacity-20" />
                <p className="text-sm">Source chunks will appear here to verify AI answers</p>
              </div>
            ) : (
              activeSources.map((s, i) => (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} key={i} className="p-4 bg-white/5 border border-white/10 rounded-2xl text-[12px] leading-relaxed relative group">
                  <div className="absolute top-2 right-2 text-[10px] font-bold text-indigo-400 bg-indigo-500/20 px-2 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                    {(s.similarity * 100).toFixed(0)}% MATCH
                  </div>
                  <p className="text-slate-400 italic">Page {s.page}</p>
                  <p className="mt-2 text-slate-300">
                    {s.text || "No text snippet available for this source."}
                  </p>
                </motion.div>
              ))
            )}
          </div>
        </aside>

      </div>
    </div>
  );
};

export default App;
