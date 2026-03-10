'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Trophy, User, Lightbulb, Send, Loader2, CheckCircle2, 
  XCircle, HelpCircle, Sparkles, Waves, ShieldCheckIcon, AlertCircle,
  Zap, Target, Award, Fish, Github
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
  const [questionsAnswered, setQuestionsAnswered] = useState<number>(0);

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
        setQuestionsAnswered(prev => prev + 1);
        setTimeout(() => { fetchQuestion(); }, 2500);
      } else {
        setFeedback('wrong');
        setStreak(0);
        setQuestionsAnswered(prev => prev + 1);
        setRevealAnswer(data.correct_answer || null);
      }
    } catch (error) {
      console.error("Submission error:", error);
    } finally {
      setIsChecking(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-blue-950 text-white font-sans relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <motion.div 
          className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/15 rounded-full blur-3xl"
          animate={{
            y: [0, 50, 0],
            x: [0, 30, 0],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
        <motion.div 
          className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/15 rounded-full blur-3xl"
          animate={{
            y: [0, -50, 0],
            x: [0, -30, 0],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
        <motion.div 
          className="absolute top-1/2 right-1/3 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 6,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      </div>

      {/* Header */}
      <motion.header 
        className="relative border-b border-blue-900/40 backdrop-blur-md bg-gradient-to-r from-slate-900/80 to-blue-900/40 sticky top-0 z-40"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
          {/* Logo */}
          <motion.div className="flex items-center gap-3" whileHover={{ scale: 1.05 }}>
            <motion.div 
              className="p-2.5 bg-gradient-to-br from-cyan-500 via-blue-500 to-purple-500 rounded-xl"
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.6 }}
            >
              <Fish className="w-6 h-6 text-white" />
            </motion.div>
            <div>
              <h1 className="hidden sm:block text-xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                Marine.AI Quiz
              </h1>
              <p className="hidden md:block text-xs text-slate-400">Identify. Learn. Discover.</p>
            </div>
          </motion.div>

          {/* Center Stats */}
          <div className="flex items-center gap-3 md:gap-6">
            {/* Streak */}
            <motion.div 
              className="flex items-center gap-2 px-3 py-2 bg-slate-800/60 backdrop-blur-sm rounded-full border border-orange-500/30"
              animate={streak > 2 ? { scale: [1, 1.05, 1] } : {}}
              transition={{ duration: 0.6, repeat: Infinity }}
            >
              <Zap className={`w-4 h-4 ${streak > 0 ? 'text-orange-400 animate-pulse' : 'text-slate-400'}`} />
              <div className="flex flex-col leading-none">
                <span className="text-[10px] text-slate-400 font-medium uppercase">Streak</span>
                <span className="text-sm font-bold text-white">{streak}</span>
              </div>
            </motion.div>

            {/* Score */}
            <motion.div 
              className="flex items-center gap-2 px-3 py-2 bg-slate-800/60 backdrop-blur-sm rounded-full border border-cyan-500/30"
              initial={{ scale: 1 }}
              whileHover={{ scale: 1.05 }}
            >
              <Trophy className="w-4 h-4 text-yellow-400" />
              <div className="flex flex-col leading-none">
                <span className="text-[10px] text-slate-400 font-medium uppercase">Score</span>
                <span className="text-sm font-bold text-white tabular-nums">{score}</span>
              </div>
            </motion.div>

            {/* Questions Answered */}
            <motion.div 
              className="flex items-center gap-2 px-3 py-2 bg-slate-800/60 backdrop-blur-sm rounded-full border border-blue-500/30"
              initial={{ scale: 1 }}
              whileHover={{ scale: 1.05 }}
            >
              <Target className="w-4 h-4 text-blue-400" />
              <div className="flex flex-col leading-none">
                <span className="text-[10px] text-slate-400 font-medium uppercase">Answered</span>
                <span className="text-sm font-bold text-white">{questionsAnswered}</span>
              </div>
            </motion.div>
          </div>

          {/* Right Actions */}
          <div className="flex items-center gap-3">
            <Link href="/leaderboard">
              <motion.button 
                className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-pink-500 to-purple-600 rounded-full shadow-lg hover:shadow-pink-500/40 transition-all duration-300"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Trophy className="w-4 h-4 text-white animate-pulse" />
                <span className="hidden sm:inline text-sm font-bold">Leaderboard</span>
              </motion.button> 
            </Link>

            {/* User Profile */}
            <motion.div 
              className="flex items-center gap-2 px-3 py-2 bg-slate-800/60 backdrop-blur-sm rounded-full border border-blue-500/30"
              whileHover={{ scale: 1.05 }}
            >
              <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-cyan-500 to-blue-600 p-[2px]">
                <div className="w-full h-full rounded-full bg-slate-900 flex items-center justify-center">
                  {authUser?.type === 'student' ? <User className="w-4 h-4 text-slate-300" /> : <ShieldCheckIcon className="w-4 h-4 text-purple-400" />}
                </div>
              </div>
              <div className="hidden sm:flex flex-col leading-tight">
                <span className="text-xs font-bold text-white">{authUser?.user || "Guest"}</span>
                <span className="text-[9px] text-cyan-400 uppercase tracking-tighter">{authUser?.type}</span>
              </div>
            </motion.div>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <main className="relative max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-16 flex flex-col justify-center min-h-[calc(100vh-100px)]">
        <AnimatePresence mode="wait">
          <motion.div 
            key={currentQuestion?.question}
            className="relative group"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
          >
            {/* Glow Background */}
            <motion.div
              className={`absolute -inset-1 rounded-2xl blur opacity-30 transition duration-1000 ${
                feedback === 'correct' ? 'bg-green-500' : feedback === 'wrong' ? 'bg-red-500' : 'bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500'
              }`}
              animate={{
                opacity: [0.3, 0.5, 0.3],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
              }}
            />

            {/* Main Card */}
            <div className="relative bg-gradient-to-br from-slate-900/80 via-slate-800/60 to-blue-900/40 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-8 sm:p-12 shadow-2xl">
              {isLoading ? (
                <motion.div 
                  className="flex flex-col items-center justify-center py-16 space-y-6"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                  >
                    <div className="w-16 h-16 rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 p-1">
                      <div className="w-full h-full rounded-full bg-slate-900 flex items-center justify-center">
                        <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
                      </div>
                    </div>
                  </motion.div>
                  <motion.p 
                    className="text-slate-300 font-medium tracking-wide"
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    Discovering Next Specimen...
                  </motion.p>
                </motion.div>
              ) : (
                <motion.div 
                  className="space-y-8"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  {/* Question Header */}
                  <div className="space-y-4">
                    <motion.div 
                      className="flex items-center gap-2 text-sm font-bold text-cyan-400 uppercase tracking-widest"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.1 }}
                    >
                      <Fish className="w-4 h-4" />
                      <span>Species Identification Challenge</span>
                      <motion.span 
                        className="ml-auto text-xs text-slate-400 font-normal normal-case"
                        animate={{ opacity: [0.5, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      >
                        Level {Math.min(5, Math.floor(questionsAnswered / 10) + 1)}
                      </motion.span>
                    </motion.div>

                    <motion.h2 
                      className="text-3xl sm:text-4xl font-black text-transparent bg-gradient-to-r from-cyan-300 via-blue-300 to-purple-300 bg-clip-text leading-tight"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.2 }}
                    >
                      {currentQuestion?.question}
                    </motion.h2>
                  </div>

                  {/* Hint Section */}
                  <motion.div 
                    className="relative"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                  >
                    {!showHint ? (
                      <motion.button 
                        onClick={() => setShowHint(true)} 
                        className="flex items-center gap-2 text-sm text-slate-400 hover:text-cyan-400 transition-all duration-300 group/hint px-3 py-2 rounded-lg hover:bg-slate-800/30"
                        whileHover={{ x: 5 }}
                      >
                        <Lightbulb className="w-4 h-4 group-hover/hint:scale-125 transition-transform" />
                        <span>Need Help? <span className="text-xs text-slate-500">(-5 pts)</span></span>
                      </motion.button>
                    ) : (
                      <motion.div
                        className="bg-gradient-to-r from-cyan-500/20 via-blue-500/20 to-purple-500/20 border border-cyan-500/30 rounded-xl p-4 backdrop-blur-sm"
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                      >
                        <div className="flex items-start gap-3">
                          <Lightbulb className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                          <p className="text-slate-200 text-sm italic leading-relaxed">"{currentQuestion?.hint}"</p>
                        </div>
                      </motion.div>
                    )}
                  </motion.div>

                  {/* Input & Submit */}
                  <motion.div 
                    className="space-y-4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.4 }}
                  >
                    <div className="relative">
                      <motion.input
                        type="text"
                        value={userAnswer}
                        onChange={(e) => {
                          setUserAnswer(e.target.value);
                          if (feedback === 'wrong') { setFeedback(null); setRevealAnswer(null); }
                        }}
                        onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
                        placeholder="Type the scientific name..."
                        disabled={feedback === 'correct'}
                        className={`w-full bg-slate-800/50 text-white placeholder:text-slate-500 px-6 py-4 rounded-xl border-2 outline-none transition-all duration-300 text-lg font-medium backdrop-blur-sm
                          ${feedback === 'correct' ? 'border-green-500/60 bg-green-500/10 text-green-100' : feedback === 'wrong' ? 'border-red-500/60 bg-red-500/10' : 'border-slate-700/50 focus:border-cyan-500 focus:bg-slate-800/70'}`}
                      />
                      <motion.div
                        className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 text-sm"
                        animate={{ opacity: [0.5, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      >
                        {userAnswer.length}/50
                      </motion.div>
                    </div>

                    {/* Feedback Messages */}
                    <AnimatePresence>
                      {feedback === 'correct' && (
                        <motion.div
                          className="p-4 bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50 rounded-xl"
                          initial={{ opacity: 0, scale: 0.95 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0 }}
                        >
                          <div className="flex items-center gap-3">
                            <motion.div animate={{ rotate: 360, scale: [1, 1.2, 1] }} transition={{ duration: 0.6, repeat: Infinity }}>
                              <CheckCircle2 className="w-5 h-5 text-green-400" />
                            </motion.div>
                            <div>
                              <p className="text-green-200 font-bold text-sm">Correct Identification!</p>
                              <p className="text-green-300 text-xs">+{lastPointsEarned} points</p>
                            </div>
                          </div>
                        </motion.div>
                      )}

                      {feedback === 'wrong' && revealAnswer && (
                        <motion.div
                          className="p-4 bg-gradient-to-r from-red-500/20 to-orange-500/20 border-l-4 border-red-500 rounded-r-xl backdrop-blur-sm"
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0 }}
                        >
                          <div className="space-y-3">
                            <div className="flex items-center gap-2">
                              <AlertCircle className="w-4 h-4 text-red-400" />
                              <span className="text-xs font-bold text-red-400 uppercase tracking-widest">Learning Opportunity</span>
                            </div>
                            <p className="text-slate-200 text-sm">
                              The correct identification is: <span className="text-white font-bold italic text-base">{revealAnswer}</span>
                            </p>
                            <motion.button 
                              onClick={fetchQuestion} 
                              className="text-xs text-cyan-400 hover:text-cyan-300 font-semibold flex items-center gap-1 transition-all mt-2 px-2 py-1 rounded hover:bg-slate-800/50"
                              whileHover={{ x: 5 }}
                            >
                              Next Challenge <Send className="w-3 h-3" />
                            </motion.button>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    {/* Submit Button */}
                    <motion.button 
                      onClick={handleSubmit} 
                      disabled={!userAnswer.trim() || isChecking || feedback === 'correct'}
                      className={`w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-2 transition-all duration-300 ${
                        feedback === 'correct' 
                          ? 'bg-green-500/40 cursor-default' 
                          : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/40'
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                      whileHover={feedback !== 'correct' ? { scale: 1.02 } : {}}
                      whileTap={feedback !== 'correct' ? { scale: 0.98 } : {}}
                    >
                      {feedback === 'correct' ? (
                        <>
                          <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity }}>
                            <Sparkles className="w-5 h-5" />
                          </motion.div>
                          Verified!
                        </>
                      ) : isChecking ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          Verifying...
                        </>
                      ) : (
                        <>
                          <Send className="w-5 h-5" />
                          Submit Answer
                        </>
                      )}
                    </motion.button>
                  </motion.div>
                </motion.div>
              )}
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Progress Bar */}
        <motion.div 
          className="mt-12 max-w-4xl mx-auto w-full"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-slate-400 font-medium uppercase">Progress</span>
            <span className="text-xs text-slate-400">{questionsAnswered} / 50 Questions</span>
          </div>
          <div className="w-full h-2 bg-slate-800/50 rounded-full overflow-hidden border border-slate-700/30">
            <motion.div
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-600"
              initial={{ width: 0 }}
              animate={{ width: `${(questionsAnswered / 50) * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </motion.div>
      </main>


    </div>
  );
}