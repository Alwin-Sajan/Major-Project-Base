"use client";

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { 
  Trophy, User, Lightbulb, Send, Loader2, CheckCircle2, 
  XCircle, HelpCircle, Sparkles, Waves, ShieldCheckIcon, AlertCircle
} from 'lucide-react';
import { useAuth } from '@/app/context/AuthContext'; 

interface QuestionData {
  question: string;
  hint: string;
  ans: string; 
}

interface CheckResponse {
  result: boolean;
  points: number;
  correct_answer?: string;
}

export default function TriviaGame() {
  const { user: authUser } = useAuth();

  // Game State
  const [score, setScore] = useState<number>(0);
  const [currentQuestion, setCurrentQuestion] = useState<QuestionData | null>(null);
  const [userAnswer, setUserAnswer] = useState<string>('');
  
  // UI State
  const [isLoading, setIsLoading] = useState<boolean>(true); 
  const [isChecking, setIsChecking] = useState<boolean>(false);
  const [showHint, setShowHint] = useState<boolean>(false); 
  const [feedback, setFeedback] = useState<'correct' | 'wrong' | null>(null);
  const [lastPointsEarned, setLastPointsEarned] = useState<number>(0);
  const [streak, setStreak] = useState<number>(0);
  const [revealAnswer, setRevealAnswer] = useState<string | null>(null);

  // 1. Fetch points from database endpoint
  const fetchPoints = useCallback(async () => {
    const userString = localStorage.getItem('marine_user');
    if (!userString) return;
    let sidInt = 0;
    try {
      const userData = JSON.parse(userString);
      sidInt = userData.uid
      const response = await fetch(`http://127.0.0.1:8000/getpoints?uid=${sidInt}`);
      if (response.ok) {
        const data = await response.json();
        setScore(data.points);
      }
    } catch (error) {
      console.error("Failed to fetch points:", error);
    }
  }, []);

  const fetchQuestion = async () => {
    setIsLoading(true);
    setFeedback(null);
    setShowHint(false); 
    setUserAnswer('');
    setRevealAnswer(null);
    
    try {
      const response = await fetch('http://127.0.0.1:8000/getQuestion');
      if (!response.ok) throw new Error('Failed to fetch');
      const data = await response.json();
      setCurrentQuestion(data);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPoints();
    fetchQuestion();
  }, [fetchPoints]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!userAnswer.trim() || isChecking || feedback === 'correct') return;
    
    setIsChecking(true);

    const userString = localStorage.getItem('marine_user');
    let sidInt = 0;
    if (userString) {
      try {
        const userData = JSON.parse(userString);
        sidInt = userData.uid;
      } catch (error) {
        console.error("Error parsing user data", error);
      }
    }

    // Substring Match Logic for Scientific Names
    const input = userAnswer.toLowerCase().trim();
    const actualAns = currentQuestion?.ans?.toLowerCase().trim() || "";
    const isActuallyCorrect = input.length > 2 && actualAns.includes(input);

    try {
      const response = await fetch('http://127.0.0.1:8000/checkAnswer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            sid: sidInt,
            answer: isActuallyCorrect ? "correct" : "incorrect",
            hint_used: showHint 
        }),
      });

      if (!response.ok) throw new Error('Failed to verify');
      
      const data: CheckResponse = await response.json();

      if (data.result) {
        setFeedback('correct');
        setLastPointsEarned(data.points); 
        setScore(prev => prev + data.points); 
        setStreak(prev => prev + 1);
        setTimeout(() => { fetchQuestion(); }, 2000);
      } else {
        setFeedback('wrong');
        setStreak(0);
        setRevealAnswer(data.correct_answer || null);
      }
    } catch (error) {
      console.error("Submission error:", error);
    } finally {
      setIsChecking(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black-950 to-slate-900 text-white font-sans">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
      </div>

      <header className="relative border-b border-blue-900/30 backdrop-blur-sm bg-slate-900/50 sticky top-0 z-40">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl">
                <Waves className="w-6 h-6 text-white" />
              </div>
              <h1 className="hidden sm:block text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                MarineQuiz <span className='text-red-500 font-medium italic'>Alpha</span>
              </h1>
            </div>

            <div className="flex items-center gap-4">
              <Link href="/leaderboard">
                <button className="flex items-center gap-3 px-4 py-2 bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500 rounded-full shadow-lg hover:shadow-pink-500/40 hover:scale-105 transition-all duration-300 active:scale-95">
                  <Trophy className="w-5 h-5 text-white animate-pulse" />
                  <div className="flex flex-col leading-none text-left">
                    <span className="text-[10px] text-white/80 font-medium uppercase tracking-wider">
                      Leaderboard
                    </span>
                    <span className="text-lg font-bold text-white">
                      View Rankings
                    </span>
                  </div>
                </button> 
              </Link>
          </div>


            <div className="flex items-center gap-4">
              <div className="flex items-center gap-3 px-4 py-2 bg-slate-800/50 rounded-full border border-slate-700/50 shadow-inner">
                <Trophy className={`w-5 h-5 ${streak > 2 ? 'text-yellow-400 animate-bounce' : 'text-blue-400'}`} />
                <div className="flex flex-col leading-none">
                  <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">Total Score</span>
                  <span className="text-lg font-bold text-white tabular-nums">{score}</span>
                </div>
              </div>

              <div className="flex items-center gap-3 pl-1 pr-4 py-1 bg-slate-800/50 rounded-full border border-slate-700/50">
                <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-blue-500 to-cyan-600 p-[2px]">
                  <div className="w-full h-full rounded-full bg-slate-900 flex items-center justify-center overflow-hidden">
                    {authUser?.type === 'student' ? <User className="w-5 h-5 text-slate-300" /> : <ShieldCheckIcon className="w-5 h-5 text-purple-400" />}
                  </div>
                </div>
                <div className="flex flex-col leading-tight">
                  <span className="text-sm font-bold text-slate-100">{authUser?.user || "Guest"}</span>
                  <span className="text-[10px] text-cyan-400 uppercase tracking-tighter font-semibold">{authUser?.type || "user"}</span>
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

          <div className="relative bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 sm:p-12 shadow-2xl">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-12 space-y-4">
                <Loader2 className="w-10 h-10 text-blue-500 animate-spin" />
                <p className="text-slate-400 animate-pulse font-medium tracking-wide">Fetching Specimen Details...</p>
              </div>
            ) : (
              <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-sm font-medium text-blue-400 uppercase tracking-widest">
                    <HelpCircle className="w-4 h-4" />
                    <span>Identification Challenge</span>
                  </div>
                  <h2 className="text-2xl sm:text-3xl font-bold text-slate-100 leading-tight">
                    {currentQuestion?.question}
                  </h2>
                </div>

                <div className="relative">
                  {!showHint ? (
                    <button onClick={() => setShowHint(true)} className="flex items-center gap-2 text-sm text-slate-400 hover:text-blue-400 transition-colors group/hint">
                      <Lightbulb className="w-4 h-4 group-hover/hint:scale-110 transition-transform" />
                      <span>Use Hint? <span className="underline decoration-slate-600 underline-offset-4 group-hover/hint:decoration-blue-500/50">(Reduces points)</span></span>
                    </button>
                  ) : (
                    <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4 animate-in fade-in slide-in-from-top-2">
                        <p className="text-slate-300 text-sm italic">"{currentQuestion?.hint}"</p>
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  <input
                    type="text"
                    value={userAnswer}
                    onChange={(e) => {
                      setUserAnswer(e.target.value);
                      if (feedback === 'wrong') { setFeedback(null); setRevealAnswer(null); }
                    }}
                    onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
                    placeholder="Enter scientific name..."
                    disabled={feedback === 'correct'}
                    className={`w-full bg-slate-800/30 text-white placeholder:text-slate-600 px-6 py-5 rounded-xl border-2 outline-none transition-all duration-300 text-lg
                      ${feedback === 'correct' ? 'border-green-500/50 bg-green-500/10' : feedback === 'wrong' ? 'border-red-500/50 bg-red-500/10' : 'border-slate-700 focus:border-blue-500'}
                    `}
                  />

                  {feedback === 'wrong' && revealAnswer && (
                    <div className="p-4 bg-slate-800/80 border-l-4 border-red-500 rounded-r-xl animate-in zoom-in-95">
                      <div className="flex items-center gap-2 mb-1">
                        <AlertCircle className="w-4 h-4 text-red-400" />
                        <span className="text-xs font-bold text-red-400 uppercase tracking-widest">Learning Discovery</span>
                      </div>
                      <p className="text-slate-300 text-sm">
                        Correct ID: <span className="text-white font-bold italic">{revealAnswer}</span>
                      </p>
                      <button onClick={fetchQuestion} className="mt-3 text-xs text-blue-400 hover:text-cyan-400 font-medium flex items-center gap-1 transition-all">
                        Skip to Next Specimen <Send className="w-3 h-3" />
                      </button>
                    </div>
                  )}

                  <button onClick={handleSubmit} disabled={!userAnswer || isChecking || feedback === 'correct'} className={`w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-2 transition-all duration-300 ${feedback === 'correct' ? 'bg-green-600/50 cursor-default' : 'bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white shadow-lg shadow-blue-500/10'} disabled:opacity-50`}>
                    {feedback === 'correct' ? <><Sparkles className="w-5 h-5" /> Verified</> : isChecking ? <Loader2 className="w-5 h-5 animate-spin" /> : <><Send className="w-5 h-5" /> Submit Answer</>}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}