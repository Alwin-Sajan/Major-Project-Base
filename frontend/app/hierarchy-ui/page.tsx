"use client";

import React, { useState } from 'react';
import { Upload, Waves, Fish, Droplets, Loader2, Sparkles, CheckCircle2, AlertCircle, Sun } from 'lucide-react';

interface AnalysisResult {
  class_name: string;
  confidence: string;
  ood: boolean;
  family?: string;
  genus?: string;
  species?: string;
}

export default function OODwithMarineSpeciesIdentifier() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisStage, setAnalysisStage] = useState<string>('');
  const [showChat, setShowChat] = useState<boolean>(false);
  const [chatMessages, setChatMessages] = useState<Array<{role: string, content: string}>>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await fetch('http://127.0.0.1:8000/predictHierarchy', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to fetch prediction');
      }

      const data = await response.json();

      if (data && data.confidence !== undefined) {
        let displayName = "";

        if (data.ood) {
          displayName = "Unknown Species";
        } else {
          displayName = data.species || "Unknown";
        }

        setResult({
          class_name: displayName,
          confidence: data.confidence.toFixed(1),
          ood: data.ood,
          family: data.family,
          genus: data.genus,
          species: data.species
        });

        setChatMessages([
          {
            role: 'assistant',
            content: data.ood
              ? "This appears to be an unknown or out-of-distribution species. I can still help with general marine biology questions!"
              : `I've identified this as ${data.species}! Ask me anything about it.`,
          },
        ]);
      } else {
        throw new Error('Invalid response from backend');
      }
    } catch (err) {
      setError('Analysis failed. Please ensure the backend server is running.');
      console.error(err);
    } finally {
      setIsAnalyzing(false);
      setAnalysisStage('');
    }
  };

  const sendChatMessage = async () => {
    if (!chatInput.trim() || !result) return;

    const userMessage = chatInput.trim();
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsChatLoading(true);

    try {
      await new Promise(resolve => setTimeout(resolve, 1000));

      const mockResponses = [
        `Great question! ${result.class_name} typically inhabits tropical and subtropical waters.`,
        `${result.class_name} feeds on small fish, plankton, and algae.`,
        `The conservation status of ${result.class_name} is being monitored.`,
        `Interesting fact: ${result.class_name} has fascinating marine adaptations!`
      ];
      
      const botResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];
      setChatMessages(prev => [...prev, { role: 'assistant', content: botResponse }]);
    } catch (error) {
      setChatMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.' 
      }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900 text-white">
      <header className="relative border-b border-blue-900/30 backdrop-blur-sm bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl">
              <Sun className="w-8 h-8" />
            </div>
            <h1 className="text-2xl font-bold">
              Marine Species Identification <span className='text-red-500'>With OOD</span>
            </h1>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-12 grid md:grid-cols-2 gap-8">

        {/* Upload */}
        <div>
          <input type="file" accept="image/*" onChange={handleImageUpload} />
          {imagePreview && <img src={imagePreview} className="mt-4 max-h-64" />}
          <button onClick={analyzeImage} className="mt-4 px-4 py-2 bg-blue-600 rounded">
            Identify Species
          </button>
        </div>

        {/* Results */}
        <div>
          {result && (
            <div>
              <h2 className="text-2xl font-bold mb-2">{result.class_name}</h2>

              {/* ✅ NEW hierarchy display */}
              {!result.ood && result.family && result.genus && result.species && (
                <p className="text-sm text-slate-400 mb-2">
                  {result.family} → {result.genus} → {result.species}
                </p>
              )}

              <p>Confidence: {result.confidence}%</p>
            </div>
          )}

          {error && <p className="text-red-400">{error}</p>}
        </div>
      </main>
    </div>
  );
}