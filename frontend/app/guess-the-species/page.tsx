"use client";

import React, { useState, useEffect } from 'react';
import { 
  Trophy, 
  User, 
  Lightbulb, 
  Send, 
  Loader2, 
  CheckCircle2, 
  XCircle,
  HelpCircle,
  Sparkles,
  Waves
} from 'lucide-react';

interface QuestionData {
  question: string;
  hint: string;
}

interface CheckResponse {
  result: boolean;
  points: number;
}

export default function TriviaGame() {
  // Game State
  const [score, setScore] = useState<number>(0);
  const [currentQuestion, setCurrentQuestion] = useState<QuestionData | null>(null);
  const [userAnswer, setUserAnswer] = useState<string>('');
  
  // UI State
  const [isLoading, setIsLoading] = useState<boolean>(true); 
  const [isChecking, setIsChecking] = useState<boolean>(false);
  const [showHint, setShowHint] = useState<boolean>(false); // This tracks if hint was used
  const [feedback, setFeedback] = useState<'correct' | 'wrong' | null>(null);
  const [lastPointsEarned, setLastPointsEarned] = useState<number>(0);
  const [streak, setStreak] = useState<number>(0);

  useEffect(() => {
    fetchQuestion();
  }, []);

  const fetchQuestion = async () => {
    setIsLoading(true);
    setFeedback(null);
    setShowHint(false); // Reset hint for the new question
    setUserAnswer('');
    
    try {
      const response = await fetch('http://127.0.0.1:8000/getQuestion');
      if (!response.ok) throw new Error('Failed to fetch question');
      const data = await response.json();
      setCurrentQuestion(data);
    } catch (error) {
      console.error("Error fetching question:", error);
      setCurrentQuestion({
        question: "Couldnt fetch qn",
        hint: "shit"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!userAnswer.trim() || isChecking) return;

    setIsChecking(true);

    try {
      // Sending both the answer AND the hint_used boolean to FastAPI
      const response = await fetch('http://127.0.0.1:8000/checkAnswer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            answer: userAnswer,
            hint_used: showHint // Sending the state of hint usage
        }),
      });

      if (!response.ok) throw new Error('Failed to verify');
      
      const data: CheckResponse = await response.json();

      if (data.result) {
        setFeedback('correct');
        setLastPointsEarned(data.points); // Store points from backend
        setScore(prev => prev + data.points);
        setStreak(prev => prev + 1);
        
        setTimeout(() => {
          fetchQuestion();
        }, 1800);
      } else {
        setFeedback('wrong');
        setStreak(0);
      }

    } catch (error) {
      console.error("Error checking answer", error);
    } finally {
      setIsChecking(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black-950 to-slate-900 text-white font-sans">
      
      {/* Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
      </div>

      {/* Header */}
      <header className="relative border-b border-blue-900/30 backdrop-blur-sm bg-slate-900/50 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl">
                <Waves className="w-6 h-6 text-white" />
              </div>
              <h1 className="hidden sm:block text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                MarineQuiz <span className='text-red-500 font-medium'>Alpha</span>
              </h1>
            </div>

            <div className="flex items-center gap-6">
              <div className="flex items-center gap-3 px-4 py-2 bg-slate-800/50 rounded-full border border-slate-700/50">
                <Trophy className={`w-5 h-5 ${streak > 2 ? 'text-yellow-400 animate-bounce' : 'text-blue-400'}`} />
                <div className="flex flex-col leading-none">
                  <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">Score</span>
                  <span className="text-lg font-bold text-white tabular-nums">{score}</span>
                </div>
              </div>
              <div className="flex items-center gap-3 pl-1 pr-4 py-1 bg-slate-800/50 rounded-full border border-slate-700/50">
                <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-blue-500 to-cyan-600 p-[2px]">
                  <div className="w-full h-full rounded-full bg-slate-900 flex items-center justify-center overflow-hidden">
                    <User className="w-5 h-5 text-slate-300" />
                  </div>
                </div>
                <span className="text-sm font-medium text-slate-200">Student Name</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="relative max-w-2xl mx-auto px-6 py-12 flex flex-col justify-center min-h-[80vh]">
        <div className="relative group">
          <div className={`absolute -inset-1 rounded-2xl blur opacity-25 transition duration-1000 
            ${feedback === 'correct' ? 'bg-green-500' : feedback === 'wrong' ? 'bg-red-500' : 'bg-gradient-to-r from-blue-600 to-cyan-600'}`}>
          </div>

          <div className="relative bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 sm:p-12">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-12 space-y-4">
                <Loader2 className="w-10 h-10 text-blue-500 animate-spin" />
                <p className="text-slate-400 animate-pulse">Conneting to Server</p>
              </div>
            ) : (
              <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-sm font-medium text-blue-400 uppercase tracking-widest">
                    <HelpCircle className="w-4 h-4" />
                    <span>Challenge</span>
                  </div>
                  <h2 className="text-2xl sm:text-3xl font-bold text-slate-100 leading-tight">
                    {currentQuestion?.question}
                  </h2>
                </div>

                {/* Hint Logic */}
                <div className="relative">
                  {!showHint ? (
                    <button 
                      onClick={() => setShowHint(true)}
                      className="flex items-center gap-2 text-sm text-slate-400 hover:text-blue-400 transition-colors group/hint"
                    >
                      <Lightbulb className="w-4 h-4 group-hover/hint:scale-110 transition-transform" />
                      <span>Use Hint? <span className="underline decoration-slate-600 underline-offset-4 group-hover/hint:decoration-blue-500/50">(Reduces points)</span></span>
                    </button>
                  ) : (
                    <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4 animate-in fade-in slide-in-from-top-2">
                      <div className="flex gap-3">
                        <Lightbulb className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                        <p className="text-slate-300 text-sm italic">"{currentQuestion?.hint}"</p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  <div className="relative">
                    <input
                      type="text"
                      value={userAnswer}
                      onChange={(e) => {
                        setUserAnswer(e.target.value);
                        if (feedback === 'wrong') setFeedback(null);
                      }}
                      onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
                      placeholder="Enter classification..."
                      disabled={feedback === 'correct'}
                      className={`w-full bg-slate-800/30 text-white placeholder:text-slate-600 px-6 py-5 rounded-xl border-2 outline-none transition-all duration-300 text-lg
                        ${feedback === 'correct' ? 'border-green-500/50 bg-green-500/10' : feedback === 'wrong' ? 'border-red-500/50 bg-red-500/10' : 'border-slate-700 focus:border-blue-500'}
                      `}
                    />
                    <div className="absolute right-4 top-1/2 -translate-y-1/2">
                      {isChecking && <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />}
                      {!isChecking && feedback === 'correct' && <CheckCircle2 className="w-6 h-6 text-green-400" />}
                      {!isChecking && feedback === 'wrong' && <XCircle className="w-6 h-6 text-red-400" />}
                    </div>
                  </div>

                  <button
                    onClick={handleSubmit}
                    disabled={!userAnswer || isChecking || feedback === 'correct'}
                    className={`w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-2 transition-all duration-300 
                      ${feedback === 'correct' ? 'bg-green-600/50' : 'bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white shadow-lg shadow-blue-500/10'}
                      disabled:opacity-50
                    `}
                  >
                    {feedback === 'correct' ? <><Sparkles className="w-5 h-5" /> Verified</> : <><Send className="w-5 h-5" /> Submit Answer</>}
                  </button>
                </div>

                {feedback === 'wrong' && (
                  <div className="flex items-center justify-center gap-2 text-red-400 font-medium animate-in fade-in">
                    <XCircle className="w-4 h-4" /><span>Try a different classification.</span>
                  </div>
                )}
                 {feedback === 'correct' && (
                  <div className="flex items-center justify-center gap-2 text-green-400 font-medium animate-in fade-in">
                    <CheckCircle2 className="w-4 h-4" /><span>Success! +{lastPointsEarned} Points.</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}