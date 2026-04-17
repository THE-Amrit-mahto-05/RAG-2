import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, BookOpen, Bot, User, CheckCircle2, Loader2, Image as ImageIcon } from 'lucide-react';
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
}

const App: React.FC = () => {
  const [topicId, setTopicId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  
  const chatEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
      setUploadProgress(null);
      setMessages([{
        role: 'assistant',
        content: `Ready! I've processed logical chunks from "${file.name}". What would you like to learn about this chapter?`
      }]);
    } catch (error) {
      console.error('Upload failed:', error);
      setUploadProgress("Upload failed. Please try again.");
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
        image: response.data.image
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat failed:', error);
    } finally {
      setIsChatting(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col max-w-5xl mx-auto w-full p-4 md:p-8">
      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-500 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <BookOpen className="text-white" size={24} />
          </div>
          <h1 className="text-2xl font-bold tracking-tight">Edulevel</h1>
        </div>
        
        {topicId && (
          <div className="flex items-center gap-2 text-emerald-400 text-sm font-medium bg-emerald-400/10 px-3 py-1.5 rounded-full border border-emerald-400/20">
            <CheckCircle2 size={16} />
            Study Session Active
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col glass-card overflow-hidden">
        {!topicId ? (
          /* Upload View */
          <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="max-w-md"
            >
              <div className="w-20 h-20 bg-indigo-500/10 rounded-full flex items-center justify-center mb-6 mx-auto">
                <Upload className="text-indigo-400" size={32} />
              </div>
              <h2 className="text-2xl font-semibold mb-3">Upload your textbook</h2>
              <p className="text-slate-400 mb-8">
                Upload any chapter PDF. Our AI will analyze the text and images to become your personal tutor for this topic.
              </p>
              
              <label className="btn-primary inline-flex">
                {isUploading ? (
                  <Loader2 className="animate-spin" size={20} />
                ) : (
                  <Upload size={20} />
                )}
                <span>{isUploading ? uploadProgress : 'Choose PDF File'}</span>
                <input type="file" className="hidden" accept=".pdf" onChange={handleUpload} disabled={isUploading} />
              </label>
            </motion.div>
          </div>
        ) : (
          /* Chat View */
          <>
            <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
              <AnimatePresence>
                {messages.map((msg, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-[85%] flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                        msg.role === 'user' ? 'bg-indigo-500' : 'bg-slate-800'
                      }`}>
                        {msg.role === 'user' ? <User size={16} /> : <Bot size={16} className="text-indigo-400" />}
                      </div>
                      <div className={`space-y-3 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                        <div className={`p-4 rounded-2xl ${
                          msg.role === 'user' 
                            ? 'bg-indigo-600 text-white rounded-tr-none' 
                            : 'bg-slate-800/50 text-slate-200 rounded-tl-none border border-slate-700/50'
                        }`}>
                          <p className="leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                        </div>
                        
                        {msg.image && (
                          <motion.div 
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="overflow-hidden rounded-2xl border border-slate-700 bg-slate-900"
                          >
                            <img src={msg.image.url} alt={msg.image.title} className="w-full max-h-72 object-contain bg-black/20" />
                            <div className="p-3 border-t border-slate-700 flex items-start gap-3">
                              <ImageIcon className="text-indigo-400 shrink-0" size={18} />
                              <div>
                                <p className="text-sm font-semibold text-slate-200">{msg.image.title}</p>
                                <p className="text-xs text-slate-400">{msg.image.description}</p>
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
                  <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center">
                      <Bot size={16} className="text-indigo-400" />
                    </div>
                    <div className="bg-slate-800/50 p-4 rounded-2xl rounded-tl-none border border-slate-700/50">
                      <Loader2 className="animate-spin text-indigo-400" size={20} />
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-slate-800 bg-slate-900/50">
              <div className="relative flex items-center gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="Ask a question about this chapter..."
                  className="flex-1 bg-slate-800/50 border border-slate-700 rounded-xl px-4 py-3 text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || isChatting}
                  className="btn-primary !p-3 rounded-xl"
                >
                  <Send size={20} />
                </button>
              </div>
            </div>
          </>
        )}
      </main>
      
      <footer className="mt-6 text-center text-slate-500 text-sm">
        Built with RAG for intelligent textbook exploration
      </footer>
    </div>
  );
};

export default App;
