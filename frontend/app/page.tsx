"use client";

import React, { useState } from 'react';
import { Upload, Waves, Fish, Droplets, Loader2, Sparkles, CheckCircle2, AlertCircle, CloudCog } from 'lucide-react';

interface AnalysisResult {
  class_name: string;
  confidence: string;
}

export default function MarineSpeciesIdentifier() {
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

  const simulateAnalysis = async () => {
    const stages = [
      'Initializing neural network...',
      'Processing image data...',
      'Extracting visual features...',
      'Analyzing marine characteristics...',
      'Comparing with species database...',
      'Calculating confidence scores...',
      'Finalizing identification...'
    ];

    for (let i = 0; i < stages.length; i++) {
      setAnalysisStage(stages[i]);
      await new Promise(resolve => setTimeout(resolve, 600));
    }
  };

const analyzeImage = async () => {
  if (!selectedImage) return;
  //setIsAnalyzing(true);
  setError(null);
  setResult(null);

  try {
    // Simulate analysis stages
    //await simulateAnalysis();

    // Prepare form data for the request
    const formData = new FormData();
    formData.append('file', selectedImage);

    // Make API call to backend for prediction
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Failed to fetch prediction');
    }

    // Parse the response
    const data = await response.json();

    // Assuming the backend sends back a response with `class_name` and `confidence`
    if (data && data.class_name && data.confidence) {
      setResult({
        class_name: data.class_name,
        confidence: data.confidence.toFixed(1),
      });

      // Initialize chatbot with species info
      setChatMessages([
        {
          role: 'assistant',
          content: `I've identified this as a ${data.class_name}! I'm here to answer any questions you have about this species. Feel free to ask about habitat, diet, conservation status, or any other details!`,
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
      // TODO: Replace with actual Gemini API call
      // const response = await fetch('https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent', {
      //   method: 'POST',
      //   headers: {
      //     'Content-Type': 'application/json',
      //     'x-goog-api-key': YOUR_GEMINI_API_KEY
      //   },
      //   body: JSON.stringify({
      //     contents: [{
      //       parts: [{
      //         text: `About ${result.class_name}: ${userMessage}`
      //       }]
      //     }]
      //   })
      // });

      // Simulated response
      await new Promise(resolve => setTimeout(resolve, 1000));
      const mockResponses = [
        `Great question! ${result.class_name} typically inhabits tropical and subtropical waters. They're known for their vibrant colors and unique behavior patterns.`,
        `${result.class_name} primarily feeds on small fish, plankton, and algae. Their diet varies based on their age and habitat location.`,
        `The conservation status of ${result.class_name} is currently being monitored. Climate change and ocean pollution pose significant threats to their population.`,
        `Interesting fact: ${result.class_name} has fascinating adaptations that help them survive in their marine environment!`
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
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
      </div>

      {/* Header */}
      <header className="relative border-b border-blue-900/30 backdrop-blur-sm bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl">
              <Waves className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Marine Species Identification <span className='text-red-500 font-medium'>Prototype</span>
              </h1>
              <p className="text-sm text-slate-400 mt-1">
                Adaptive Detection with Open-Set & Incremental Learning
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative max-w-6xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-full mb-6">
            <Sparkles className="w-4 h-4 text-blue-400" />
            <span className="text-sm text-blue-300">AI-Powered Marine Life Recognition</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">
            Discover Marine Species Instantly
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Upload an image of any marine creature and let our advanced neural network identify the species with remarkable accuracy.
          </p>
        </div>

        {/* Upload Section */}
        <div className="grid md:grid-cols-2 gap-8 mb-8">
          {/* Upload Area */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <Upload className="w-5 h-5 text-blue-400" />
              Upload Image
            </h3>
            
            <div
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              className="border-2 border-dashed border-slate-700 hover:border-blue-500/50 rounded-xl p-8 text-center cursor-pointer transition-all duration-300 hover:bg-slate-800/30 group"
            >
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                {imagePreview ? (
                  <div className="space-y-4">
                    <img
                      src={imagePreview}
                      alt="Preview"
                      className="max-h-64 mx-auto rounded-lg border border-slate-700"
                    />
                    <p className="text-sm text-slate-400">Click or drag to change image</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="w-20 h-20 mx-auto bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                      <Fish className="w-10 h-10 text-blue-400" />
                    </div>
                    <div>
                      <p className="text-lg font-medium text-slate-300 mb-2">
                        Drop your image here
                      </p>
                      <p className="text-sm text-slate-500">
                        or click to browse files
                      </p>
                    </div>
                    <p className="text-xs text-slate-600">
                      Supports: JPG, PNG, WebP
                    </p>
                  </div>
                )}
              </label>
            </div>

            <button
              onClick={analyzeImage}
              disabled={!selectedImage || isAnalyzing}
              className="w-full mt-6 px-6 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed rounded-xl font-semibold transition-all duration-300 transform hover:scale-[1.02] disabled:scale-100 disabled:opacity-50 flex items-center justify-center gap-2 shadow-lg shadow-blue-500/20"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Identify Species
                </>
              )}
            </button>
          </div>

          {/* Results Area */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8">
            <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <Droplets className="w-5 h-5 text-cyan-400" />
              Analysis Results
            </h3>

            {!isAnalyzing && !result && !error && (
              <div className="h-64 flex flex-col items-center justify-center text-slate-500">
                <Fish className="w-16 h-16 mb-4 opacity-50" />
                <p className="text-center">Upload an image and click "Identify Species" to begin analysis</p>
              </div>
            )}

            {isAnalyzing && (
              <div className="space-y-6">
                <div className="flex flex-col items-center justify-center py-8">
                  <div className="relative">
                    <div className="w-24 h-24 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
                    <Fish className="w-12 h-12 text-blue-400 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div className="flex items-center gap-3 text-sm">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <span className="text-slate-300">{analysisStage}</span>
                  </div>
                  <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full animate-pulse"></div>
                  </div>
                </div>

                <div className="bg-slate-800/50 rounded-lg p-4 space-y-2 text-sm text-slate-400">
                  <p>• Deep learning model processing</p>
                  <p>• Feature extraction in progress</p>
                  <p>• Cross-referencing species database</p>
                </div>
              </div>
            )}

            {result && (
              <div className="space-y-6 animate-in fade-in duration-500">
                <div className="flex items-start gap-3 p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                  <CheckCircle2 className="w-6 h-6 text-green-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-green-300 font-medium mb-1">Identification Successful</p>
                    <p className="text-sm text-green-400/70">Species recognized with high confidence</p>
                  </div>
                </div>

                <div className="bg-gradient-to-br from-slate-800/80 to-slate-800/40 rounded-xl p-6 border border-slate-700">
                  <p className="text-sm text-slate-400 mb-2">Identified Species</p>
                  <h4 className="text-2xl font-bold text-white mb-4">{result.class_name}</h4>
                  
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-slate-400">Confidence Level</span>
                        <span className="text-blue-400 font-semibold">{result.confidence}%</span>
                      </div>
                      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full transition-all duration-1000"
                          style={{width: `${result.confidence}%`}}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-800/50 rounded-lg p-4 text-center">
                    <p className="text-slate-400 text-sm mb-1">Model</p>
                    <p className="text-white font-semibold">ResNet-18</p>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-4 text-center">
                    <p className="text-slate-400 text-sm mb-1">Status</p>
                    <p className="text-green-400 font-semibold">Active</p>
                  </div>
                </div>

                {/* Chat with AI Button */}
                <button
                  onClick={() => setShowChat(!showChat)}
                  className="w-full mt-4 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 rounded-xl font-semibold transition-all duration-300 flex items-center justify-center gap-2"
                >
                  <Sparkles className="w-5 h-5" />
                  {showChat ? 'Hide AI Assistant' : 'Ask AI About This Species'}
                </button>
              </div>
            )}

            {error && (
              <div className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-red-300 font-medium mb-1">Analysis Error</p>
                  <p className="text-sm text-red-400/70">{error}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* AI Chatbot Section */}
        {showChat && result && (
          <div className="mt-8 bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-6 animate-in fade-in duration-300">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">AI Marine Assistant</h3>
                <p className="text-sm text-slate-400">Powered by Gemini</p>
              </div>
            </div>

            {/* Chat Messages */}
            <div className="bg-slate-800/50 rounded-xl p-4 mb-4 max-h-96 overflow-y-auto space-y-4">
              {chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {msg.role === 'assistant' && (
                    <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center flex-shrink-0">
                      <Fish className="w-4 h-4" />
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] px-4 py-3 rounded-2xl ${
                      msg.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-700 text-slate-100'
                    }`}
                  >
                    <p className="text-sm">{msg.content}</p>
                  </div>
                  {msg.role === 'user' && (
                    <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                      <span className="text-xs font-bold">You</span>
                    </div>
                  )}
                </div>
              ))}
              {isChatLoading && (
                <div className="flex gap-3 justify-start">
                  <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                    <Loader2 className="w-4 h-4 animate-spin" />
                  </div>
                  <div className="bg-slate-700 px-4 py-3 rounded-2xl">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Chat Input */}
            <div className="flex gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
                placeholder="Ask about habitat, diet, behavior..."
                className="flex-1 px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-purple-500 transition-colors"
              />
              <button
                onClick={sendChatMessage}
                disabled={!chatInput.trim() || isChatLoading}
                className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed rounded-xl font-semibold transition-all duration-300"
              >
                Send
              </button>
            </div>

            {/* Quick Questions */}
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                onClick={() => {
                  setChatInput('What is the habitat of this species?');
                  setTimeout(sendChatMessage, 100);
                }}
                className="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-full transition-colors"
              >
                🏝️ Habitat
              </button>
              <button
                onClick={() => {
                  setChatInput('What does this species eat?');
                  setTimeout(sendChatMessage, 100);
                }}
                className="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-full transition-colors"
              >
                🍽️ Diet
              </button>
              <button
                onClick={() => {
                  setChatInput('What is the conservation status?');
                  setTimeout(sendChatMessage, 100);
                }}
                className="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-full transition-colors"
              >
                🌍 Conservation
              </button>
              <button
                onClick={() => {
                  setChatInput('Tell me interesting facts about this species');
                  setTimeout(sendChatMessage, 100);
                }}
                className="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-full transition-colors"
              >
                ✨ Fun Facts
              </button>
            </div>
          </div>
        )}

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-6 mt-12">
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-xl p-6 hover:border-blue-500/30 transition-colors">
            <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
              <Sparkles className="w-6 h-6 text-blue-400" />
            </div>
            <h4 className="font-semibold mb-2">Open-Set Detection</h4>
            <p className="text-sm text-slate-400">Identifies unknown species and adapts to new marine life classifications.</p>
          </div>
          
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-xl p-6 hover:border-cyan-500/30 transition-colors">
            <div className="w-12 h-12 bg-cyan-500/20 rounded-lg flex items-center justify-center mb-4">
              <Fish className="w-6 h-6 text-cyan-400" />
            </div>
            <h4 className="font-semibold mb-2">Incremental Learning</h4>
            <p className="text-sm text-slate-400">Continuously learns from new data without forgetting previous knowledge.</p>
          </div>
          
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-xl p-6 hover:border-purple-500/30 transition-colors">
            <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center mb-4">
              <Waves className="w-6 h-6 text-purple-400" />
            </div>
            <h4 className="font-semibold mb-2">High Accuracy</h4>
            <p className="text-sm text-slate-400">Advanced neural networks trained on extensive marine biodiversity datasets.</p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative border-t border-slate-800 mt-16">
        <div className="max-w-7xl mx-auto px-6 py-8 text-center text-slate-500 text-sm">
          <p>Marine Species Identification System • Advanced AI Research Project</p>
        </div>
      </footer>
    </div>
  );
}