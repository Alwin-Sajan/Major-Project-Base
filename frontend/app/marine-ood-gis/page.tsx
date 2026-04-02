"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Upload, Waves, Fish, Droplets, Loader2, Sparkles, CheckCircle2, AlertCircle, MapPin, Navigation, Send, Info } from 'lucide-react';
import dynamic from 'next/dynamic';

// --- Leaflet Fixes & Imports ---
// We import L dynamically inside the component to avoid SSR errors
import type { Icon } from 'leaflet';

// Dynamically import Leaflet components with SSR disabled
const MapContainer = dynamic(() => import('react-leaflet').then((mod) => mod.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then((mod) => mod.TileLayer), { ssr: false });
const Marker = dynamic(() => import('react-leaflet').then((mod) => mod.Marker), { ssr: false });
const Popup = dynamic(() => import('react-leaflet').then((mod) => mod.Popup), { ssr: false });

interface AnalysisResult {
  class_name: string;
  confidence: string;
  ood: boolean;
  location?: { lat: number; lng: number };
}

export default function MarineIdentifierPro() {
  const [isMounted, setIsMounted] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<Array<{role: string, content: string}>>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Set default coordinates (e.g., Kochi, India coast)
  const defaultPos: [number, number] = [9.9312, 76.2673];

  // 1. Handle Hydration & Leaflet Icon Fix
  useEffect(() => {
    setIsMounted(true);
    
    // This fixes the missing marker icon issue in Next.js
    import('leaflet').then((L) => {
      const DefaultIcon = L.icon({
        iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
        shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
      });
      L.Marker.prototype.options.icon = DefaultIcon;
    });
  }, []);

  // Scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
      setChatMessages([]);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;
    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await fetch('http://127.0.0.1:8000/predictOODDetection', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Backend unreachable');
      const data = await response.json();

      const analysis = {
        class_name: data.class_name,
        confidence: data.confidence.toFixed(1),
        ood: data.ood,
        // Simulation: in production, your backend might provide coordinates
        location: { lat: 9.9312 + (Math.random() * 0.1), lng: 76.2673 + (Math.random() * 0.1) }
      };

      setResult(analysis);
      initGeminiChat(analysis.class_name);
    } catch (err) {
      setError('Analysis failed. Ensure Python backend is running at :8000');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const initGeminiChat = async (species: string) => {
    setIsChatLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/assistant/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: `I just discovered a ${species}. Tell me about its habitat and diet.`, context: species })
      });
      const data = await res.json();
      setChatMessages([{ role: 'assistant', content: data.response }]);
    } catch {
      setChatMessages([{ role: 'assistant', content: `Identified: ${species}. I couldn't reach Gemini, but I'm ready for your questions!` }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const sendChatMessage = async () => {
    if (!chatInput.trim()) return;
    const userMsg = chatInput;
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsChatLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/assistant/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg, context: result?.class_name }),
      });
      const data = await response.json();
      setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: "Connection to AI Assistant lost." }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-100 selection:bg-blue-500/30">
      <header className="border-b border-white/5 bg-slate-900/40 backdrop-blur-md sticky top-0 z-[1000]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-600 rounded-lg shadow-lg shadow-blue-500/20">
              <Waves className="text-white w-6 h-6" />
            </div>
            <h1 className="text-xl font-bold tracking-tight">Marine<span className="text-blue-500">DB</span> <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded-full ml-2">OOD v2.0</span></h1>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-6">
        <div className="grid lg:grid-cols-12 gap-6">
          
          {/* LEFT: Upload & GIS */}
          <div className="lg:col-span-7 space-y-6">
            <section className="bg-slate-900/50 border border-white/5 rounded-3xl p-6 relative">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Upload size={18} className="text-blue-400" /> Imagery Analysis
              </h2>

              <div className={`relative group border-2 border-dashed rounded-2xl transition-all duration-300 ${imagePreview ? 'border-blue-500/50' : 'border-slate-700 hover:border-blue-500/50'} p-4 bg-slate-950/50`}>
                <input type="file" onChange={handleImageUpload} className="absolute inset-0 opacity-0 cursor-pointer z-10" />
                {imagePreview ? (
                  <img src={imagePreview} alt="Preview" className="w-full h-64 object-cover rounded-xl" />
                ) : (
                  <div className="py-12 flex flex-col items-center">
                    <div className="w-16 h-16 bg-blue-500/10 rounded-full flex items-center justify-center mb-4">
                        <Navigation className="text-blue-500 animate-pulse" />
                    </div>
                    <p className="text-slate-300 font-medium">Upload Marine Specimen</p>
                  </div>
                )}
              </div>

              {error && <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-lg flex items-center gap-2"><AlertCircle size={16}/>{error}</div>}

              <button onClick={analyzeImage} disabled={!selectedImage || isAnalyzing} className="w-full mt-6 py-4 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 rounded-xl font-bold transition-all flex items-center justify-center gap-3">
                {isAnalyzing ? <Loader2 className="animate-spin" /> : <Sparkles size={20} />}
                {isAnalyzing ? "Processing..." : "Detect Species"}
              </button>
            </section>

            {/* GIS MAP SECTION */}
            <section className="bg-slate-900/50 border border-white/5 rounded-3xl p-6 h-[400px] relative overflow-hidden">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <MapPin size={18} className="text-red-400" /> Sighting Map
                </h2>
                <div className="h-full rounded-2xl overflow-hidden bg-slate-950 z-0">
                  {isMounted ? (
                    <MapContainer center={defaultPos} zoom={12} style={{ height: '100%', width: '100%' }}>
                      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                      {result?.location && (
                        <Marker position={[result.location.lat, result.location.lng]}>
                          <Popup><div className="text-slate-900 font-bold">{result.class_name}</div></Popup>
                        </Marker>
                      )}
                    </MapContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-slate-500 text-sm italic">Loading Satellite Data...</div>
                  )}
                </div>
            </section>
          </div>

          {/* RIGHT: Stats & AI */}
          <div className="lg:col-span-5 space-y-6">
            {result && (
                <div className="bg-gradient-to-br from-blue-600 to-blue-700 rounded-3xl p-6 text-white shadow-2xl animate-in fade-in slide-in-from-bottom-4">
                    <div className="flex justify-between items-start mb-4">
                      <span className="px-2 py-1 bg-white/20 rounded text-[10px] font-bold tracking-widest uppercase">Detection Result</span>
                      {result.ood && <span className="px-2 py-1 bg-red-500 rounded text-[10px] font-bold">OOD DETECTED</span>}
                    </div>
                    <h2 className="text-3xl font-black">{result.class_name}</h2>
                    <div className="mt-4 h-1.5 w-full bg-white/20 rounded-full overflow-hidden">
                      <div className="h-full bg-white transition-all duration-1000" style={{ width: `${result.confidence}%` }}></div>
                    </div>
                    <p className="mt-2 text-blue-100 text-sm font-medium">Confidence Score: {result.confidence}%</p>
                </div>
            )}

            <section className="bg-slate-900/50 border border-white/5 rounded-3xl flex flex-col h-[650px]">
              <div className="p-6 border-b border-white/5 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gradient-to-tr from-purple-600 to-blue-600 rounded-full flex items-center justify-center"><Sparkles size={18} /></div>
                  <h3 className="font-bold">Gemini Assistant</h3>
                </div>
                {isChatLoading && <Loader2 size={16} className="animate-spin text-blue-400"/>}
              </div>

              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {chatMessages.length === 0 && (
                  <div className="h-full flex flex-col items-center justify-center text-slate-600 text-center">
                    <Info size={32} className="mb-2 opacity-20" />
                    <p className="text-sm">Upload a photo to unlock <br/>AI-powered species details</p>
                  </div>
                )}
                {chatMessages.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[90%] p-4 rounded-2xl text-sm ${msg.role === 'user' ? 'bg-blue-600 text-white rounded-tr-none' : 'bg-slate-800 text-slate-200 rounded-tl-none'}`}>
                      {msg.content}
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>

              <div className="p-4">
                <div className="relative">
                  <input 
                    value={chatInput} 
                    onChange={(e) => setChatInput(e.target.value)} 
                    onKeyDown={(e) => e.key === 'Enter' && sendChatMessage()}
                    placeholder="Ask Gemini about this species..." 
                    className="w-full bg-slate-950 border border-white/5 rounded-xl py-4 px-4 pr-12 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  />
                  <button onClick={sendChatMessage} className="absolute right-2 top-2 p-2 bg-blue-600 rounded-lg hover:bg-blue-500 transition-colors"><Send size={18} /></button>
                </div>
              </div>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}