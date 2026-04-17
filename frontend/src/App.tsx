import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, BookOpen, Bot, User, CheckCircle2, Loader2, Image as ImageIcon, RotateCcw, ChevronRight, Hash } from 'lucide-react';
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
  }[];
}

const App: React.FC = () => {
  // --- Session Persistence (Phase 6) ---
  const [topicId, setTopicId] = useState<string | null>(() => localStorage.getItem('edulevel_topic_id'));
  const [messages, setMessages] = useState<Message[]>(() => {
    const saved = localStorage.getItem('edulevel_messages');
    return saved ? JSON.parse(saved) : [];
  });
  
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Persistence side effects
  useEffect(() => {
    if (topicId) localStorage.setItem('edulevel_topic_id', topicId);
    else localStorage.removeItem('edulevel_topic_id');
  }, [topicId]);

  useEffect(() => {
    localStorage.setItem('edulevel_messages', JSON.stringify(messages));
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const resetSession = () => {
    if (window.confirm("Start a new study session? This will clear current progress.")) {
      setTopicId(null);
      setMessages([]);
      localStorage.removeItem('edulevel_topic_id');
      localStorage.removeItem('edulevel_messages');
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadProgress("Analyzing textbook structure...");
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/upload', formData);
      setTopicId(response.data.id);
      setUploadProgress(null);
      setMessages([{
        role: 'assistant',
        content: `Hi! I've analyzed "${file.name}". I found ${response.data.chunk_count} logical segments and categorized the diagrams. \n\nWhat would you like to explore today?`
      }]);
    } catch (error) {
      console.error('Upload failed:', error);
      setUploadProgress("Upload failed. Chapter too large or invalid PDF.");
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
      setMessages(prev => [...prev, { role: 'assistant', content: "I'm having trouble connecting to my brain. Check your Groq API key!" }]);
    } finally {
      setIsChatting(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col max-w-5xl mx-auto w-full p-4 md:p-8 h-screen max-h-screen overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between mb-6 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-500 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <BookOpen className="text-white" size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">Edulevel</h1>
            <p className="text-[10px] uppercase tracking-widest text-indigo-400 font-bold">Textbook Intelligence</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {topicId && (
            <button 
              onClick={resetSession}
              className="p-2 text-slate-400 hover:text-white hover:bg-white/5 rounded-lg transition-colors border border-transparent hover:border-slate-800"
              title="Reset Session"
            >
              <RotateCcw size={18} />
            </button>
          )}
          {topicId && (
            <div className="hidden md:flex items-center gap-2 text-emerald-400 text-xs font-medium bg-emerald-400/10 px-3 py-1.5 rounded-full border border-emerald-400/20">
              <CheckCircle2 size={14} />
              Chapter Active
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col glass-card overflow-hidden shadow-2xl relative">
        <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/5 to-transparent pointer-events-none" />
        
        {!topicId ? (
          /* Upload View with Premium Entrance */
          <div className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-slate-900/40">
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
              className="max-w-md"
            >
              <div className="relative inline-block mb-8">
                <div className="w-24 h-24 bg-indigo-600 rounded-3xl rotate-12 flex items-center justify-center shadow-2xl shadow-indigo-600/40">
                  <Upload className="text-white -rotate-12" size={36} />
                </div>
                <div className="absolute -bottom-2 -right-2 w-10 h-10 bg-emerald-500 rounded-full border-4 border-slate-900 flex items-center justify-center">
                  <ChevronRight className="text-white" size={20} />
                </div>
              </div>
              
              <h2 className="text-3xl font-bold mb-4 text-white">Upload Chapter</h2>
              <p className="text-slate-400 mb-10 leading-relaxed text-lg">
                Upload your textbook chapter. Our AI will analyze data, extract diagrams, and answer your complex questions.
              </p>
              
              <label className="btn-primary w-full py-4 text-lg cursor-pointer flex justify-center items-center gap-3 active:scale-95 transition-transform">
                {isUploading ? <Loader2 className="animate-spin" size={24} /> : <Upload size={24} />}
                <span>{isUploading ? uploadProgress : 'Begin Learning'}</span>
                <input type="file" className="hidden" accept=".pdf" onChange={handleUpload} disabled={isUploading} />
              </label>
            </motion.div>
          </div>
        ) : (
          /* Chat View with Staggered Messages */
          <>
            <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8 scroll-smooth scrollbar-thin">
              <AnimatePresence initial={false}>
                {messages.map((msg, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 20, x: msg.role === 'user' ? 20 : -20 }}
                    animate={{ opacity: 1, y: 0, x: 0 }}
                    transition={{ type: "spring", stiffness: 260, damping: 20 }}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-[90%] md:max-w-[70%] flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 shadow-lg ${
                        msg.role === 'user' ? 'bg-indigo-600' : 'bg-slate-800 border border-slate-700'
                      }`}>
                        {msg.role === 'user' ? <User size={20} className="text-white" /> : <Bot size={20} className="text-indigo-400" />}
                      </div>
                      
                      <div className={`space-y-4 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                        {/* Bubble */}
                        <div className={`p-5 rounded-3xl shadow-lg border ${
                          msg.role === 'user' 
                            ? 'bg-indigo-600 text-white rounded-tr-none border-indigo-500' 
                            : 'bg-slate-800/80 text-slate-200 rounded-tl-none border-slate-700/50 backdrop-blur-md'
                        }`}>
                          <p className="leading-relaxed text-[15px] whitespace-pre-wrap font-medium">{msg.content}</p>
                        </div>
                        
                        {/* Interactive Page Chips (Phase 6) */}
                        {!isChatting && msg.sources && msg.sources.length > 0 && (
                          <div className="flex flex-wrap gap-2">
                            {msg.sources.map((src, sIdx) => (
                              <div 
                                key={sIdx}
                                className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-900/50 border border-slate-700/50 rounded-lg text-xs font-bold text-slate-400 hover:text-indigo-400 hover:border-indigo-500/30 transition-all cursor-help"
                                title={`Match Score: ${(src.similarity * 100).toFixed(1)}%`}
                              >
                                <Hash size={12} />
                                <span>Page {src.page}</span>
                              </div>
                            ))}
                          </div>
                        )}
                        
                        {/* Premium Image Card (Phase 6) */}
                        {msg.image && (
                          <motion.div 
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="overflow-hidden rounded-3xl border border-slate-700 bg-slate-900 group shadow-2xl max-w-sm"
                          >
                            <div className="relative overflow-hidden">
                              <img src={msg.image.url} alt={msg.image.title} className="w-full object-contain bg-black/40 group-hover:scale-105 transition-transform duration-500" />
                              <div className="absolute top-3 right-3 bg-indigo-600/90 text-[10px] font-bold px-2 py-1 rounded-md text-white backdrop-blur-md">
                                TEXTBOOK FIGURE
                              </div>
                            </div>
                            <div className="p-4 bg-slate-800/80 backdrop-blur-md flex items-start gap-3">
                              <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center shrink-0">
                                <ImageIcon className="text-indigo-400" size={16} />
                              </div>
                              <div>
                                <p className="text-sm font-bold text-white mb-0.5">{msg.image.title}</p>
                                <p className="text-[11px] text-slate-400 leading-tight italic">{msg.image.description}</p>
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              
              {isChatting && (
                <div className="flex justify-start">
                  <div className="flex gap-4">
                    <div className="w-10 h-10 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center">
                      <Bot size={20} className="text-indigo-400" />
                    </div>
                    <div className="bg-slate-800/80 p-5 rounded-3xl rounded-tl-none border border-slate-700/50 backdrop-blur-md">
                      <div className="flex gap-1.5 item-center">
                        <motion.div animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1 }} className="w-1.5 h-1.5 bg-indigo-400 rounded-full" />
                        <motion.div animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1, delay: 0.2 }} className="w-1.5 h-1.5 bg-indigo-400 rounded-full" />
                        <motion.div animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1, delay: 0.4 }} className="w-1.5 h-1.5 bg-indigo-400 rounded-full" />
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} className="h-4" />
            </div>

            {/* Premium Input Tray */}
            <div className="p-6 border-t border-slate-800 bg-slate-900">
              <div className="relative flex items-center gap-3">
                <div className="flex-1 relative group">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                    placeholder="Ask a concept, e.g., 'Explain the nitrogen cycle...'"
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-2xl px-5 py-4 text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all text-[15px]"
                  />
                </div>
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || isChatting}
                  className="btn-primary !p-4 rounded-2xl shadow-lg shadow-indigo-600/20 active:scale-90 transition-transform disabled:opacity-50 disabled:grayscale"
                >
                  <Send size={24} />
                </button>
              </div>
            </div>
          </>
        )}
      </main>
      
      <footer className="mt-6 flex justify-center items-center gap-4 text-slate-500 text-xs font-bold uppercase tracking-widest bg-white/5 py-3 px-6 rounded-2xl border border-white/5">
        <Bot size={14} className="text-indigo-500" />
        Edulevel AI Textbook Engine V2.0
      </footer>
    </div>
  );
};

export default App;
