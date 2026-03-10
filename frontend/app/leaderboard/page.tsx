"use client";

import React, { useEffect, useState } from 'react';
import { 
  Trophy, 
  Medal, 
  Crown, 
  Waves, 
  ChevronLeft, 
  Loader2, 
  TrendingUp,
  User,
  Star,
  Award,
  Sparkles,
  Fish
} from 'lucide-react';
import Link from 'next/link';

interface LeaderboardEntry {
  sid: number;
  username: string;
  score: number;
  institution: string;
}

export default function LeaderboardPage() {
  const [leaders, setLeaders] = useState<LeaderboardEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // setLeaders([
    //   {'sid': 2, 'username': 'Archana', 'score': 27, 'institution': 'TIST'}, 
    //   {'sid': 3, 'username': 'Saj', 'score': 24, 'institution': 'CUSAT'}, 
    //   {'sid': 4, 'username': 'Ashik', 'score': 15, 'institution': 'TKM'}, 
    //   {'sid': 1, 'username': 'Aravind', 'score': 7, 'institution': 'TIST'}
    // ]);
    // setIsLoading(false);
  
    const fetchLeaderboard = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/leaderboard');
        if (response.ok) {
          const data = await response.json();
          setLeaders(data);
        }
      } catch (error) {
        console.error("Failed to fetch leaderboard", error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchLeaderboard();
  }, []);

  const getTopThreeStyle = (index: number) => {
    switch (index) {
      case 0: 
        return {
          bg: "bg-gradient-to-br from-yellow-500/20 via-amber-500/10 to-transparent",
          border: "border-yellow-500/60",
          glow: "shadow-lg shadow-yellow-500/20",
          badge: "bg-gradient-to-br from-yellow-400 to-amber-600",
          textGlow: "text-yellow-400 drop-shadow-[0_0_8px_rgba(251,191,36,0.5)]"
        };
      case 1:
        return {
          bg: "bg-gradient-to-br from-slate-400/15 via-slate-500/10 to-transparent",
          border: "border-slate-400/50",
          glow: "shadow-lg shadow-slate-400/20",
          badge: "bg-gradient-to-br from-slate-300 to-slate-500",
          textGlow: "text-slate-300 drop-shadow-[0_0_8px_rgba(203,213,225,0.4)]"
        };
      case 2:
        return {
          bg: "bg-gradient-to-br from-amber-700/15 via-orange-700/10 to-transparent",
          border: "border-amber-600/50",
          glow: "shadow-lg shadow-amber-600/20",
          badge: "bg-gradient-to-br from-amber-600 to-orange-700",
          textGlow: "text-amber-500 drop-shadow-[0_0_8px_rgba(245,158,11,0.4)]"
        };
      default:
        return {
          bg: "bg-slate-900/40",
          border: "border-slate-800",
          glow: "",
          badge: "bg-slate-800",
          textGlow: "text-white"
        };
    }
  };

  const getRankIcon = (index: number) => {
    switch (index) {
      case 0: return <Trophy className="w-8 h-8 text-yellow-400 drop-shadow-[0_0_8px_rgba(251,191,36,0.6)]" />;
      case 1: return <Medal className="w-7 h-7 text-slate-300 drop-shadow-[0_0_6px_rgba(203,213,225,0.5)]" />;
      case 2: return <Medal className="w-7 h-7 text-amber-600 drop-shadow-[0_0_6px_rgba(217,119,6,0.5)]" />;
      default: return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900 text-white p-6 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
        
        {/* Floating fish icons */}
        <Fish className="absolute top-20 right-1/4 w-8 h-8 text-cyan-400/20 animate-bounce" style={{animationDuration: '3s'}} />
        <Fish className="absolute bottom-32 left-1/3 w-6 h-6 text-blue-400/20 animate-bounce" style={{animationDuration: '4s', animationDelay: '1s'}} />
        <Waves className="absolute top-1/3 left-1/4 w-10 h-10 text-cyan-400/20 animate-pulse" />
      </div>

      <div className="max-w-4xl mx-auto relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Link href="/" className="flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors group">
            <ChevronLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
            <span>Back</span>
          </Link>
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl shadow-lg shadow-cyan-500/30">
              <Trophy className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                Leaderboard
              </h1>
              <p className="text-sm text-slate-400">Marine Species Champions</p>
            </div>
          </div>
        </div>

        {/* Top 3 Podium */}
        {!isLoading && leaders.length >= 3 && (
          <div className="mb-12">
            <div className="flex items-end justify-center gap-4 mb-8">
              {/* 2nd Place */}
              <div className="flex flex-col items-center flex-1 max-w-[200px]">
                <div className="relative mb-4">
                  <div className="absolute inset-0 bg-slate-400/20 rounded-full blur-xl"></div>
                  <div className="relative w-20 h-20 bg-gradient-to-br from-slate-300 to-slate-500 rounded-full flex items-center justify-center border-4 border-slate-900 shadow-lg">
                    <User className="w-10 h-10 text-white" />
                  </div>
                  <div className="absolute -bottom-2 -right-2 w-10 h-10 bg-gradient-to-br from-slate-300 to-slate-500 rounded-full flex items-center justify-center border-2 border-slate-900 shadow-lg">
                    <span className="text-sm font-bold text-slate-900">2</span>
                  </div>
                </div>
                <h3 className="font-bold text-lg text-white mb-1">{leaders[1].username}</h3>
                <p className="text-xs text-blue-400 font-semibold mb-2">{leaders[1].institution}</p>
                <div className="bg-slate-900/60 backdrop-blur-sm border border-slate-700 rounded-xl px-4 py-2 shadow-lg">
                  <p className="text-2xl font-black text-slate-300">{leaders[1].score}</p>
                  <p className="text-xs text-slate-500">points</p>
                </div>
                {/* Podium */}
                <div className="w-full h-32 bg-gradient-to-b from-slate-400/20 to-slate-600/20 rounded-t-xl mt-4 border-t-4 border-slate-400 flex items-center justify-center">
                  <Medal className="w-12 h-12 text-slate-300 drop-shadow-lg" />
                </div>
              </div>

              {/* 1st Place - Champion */}
              <div className="flex flex-col items-center flex-1 max-w-[220px] -mt-8">
                <div className="relative mb-4">
                  {/* Glow effect */}
                  <div className="absolute inset-0 bg-yellow-400/30 rounded-full blur-2xl animate-pulse"></div>
                  <div className="relative w-28 h-28 bg-gradient-to-br from-yellow-400 via-amber-500 to-orange-600 rounded-full flex items-center justify-center border-4 border-yellow-300 shadow-2xl shadow-yellow-500/50">
                    <User className="w-14 h-14 text-white" />
                  </div>
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <Crown className="w-8 h-8 text-yellow-400 fill-yellow-400 drop-shadow-lg animate-bounce" />
                  </div>
                  <div className="absolute -bottom-2 -right-2 w-12 h-12 bg-gradient-to-br from-yellow-400 to-amber-600 rounded-full flex items-center justify-center border-2 border-yellow-300 shadow-lg">
                    <span className="text-lg font-black text-white">1</span>
                  </div>
                  {/* Sparkles */}
                  <Sparkles className="absolute -top-1 -left-2 w-5 h-5 text-yellow-400 animate-pulse" />
                  <Sparkles className="absolute -top-1 -right-2 w-4 h-4 text-yellow-400 animate-pulse" style={{animationDelay: '0.5s'}} />
                </div>
                <h3 className="font-bold text-xl text-yellow-400 mb-1 drop-shadow-lg">{leaders[0].username}</h3>
                <p className="text-xs text-cyan-400 font-semibold mb-2">{leaders[0].institution}</p>
                <div className="bg-gradient-to-br from-yellow-500/20 to-amber-600/20 backdrop-blur-sm border-2 border-yellow-500/60 rounded-xl px-6 py-3 shadow-2xl shadow-yellow-500/30">
                  <p className="text-3xl font-black text-yellow-400 drop-shadow-lg">{leaders[0].score}</p>
                  <p className="text-xs text-yellow-300/80">points</p>
                </div>
                {/* Podium */}
                <div className="w-full h-40 bg-gradient-to-b from-yellow-400/20 to-amber-600/20 rounded-t-xl mt-4 border-t-4 border-yellow-400 flex items-center justify-center relative overflow-hidden">
                  <Trophy className="w-16 h-16 text-yellow-400 drop-shadow-2xl z-10" />
                  {/* Shine effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-[shimmer_2s_infinite]"></div>
                </div>
              </div>

              {/* 3rd Place */}
              <div className="flex flex-col items-center flex-1 max-w-[200px]">
                <div className="relative mb-4">
                  <div className="absolute inset-0 bg-amber-600/20 rounded-full blur-xl"></div>
                  <div className="relative w-20 h-20 bg-gradient-to-br from-amber-600 to-orange-700 rounded-full flex items-center justify-center border-4 border-slate-900 shadow-lg">
                    <User className="w-10 h-10 text-white" />
                  </div>
                  <div className="absolute -bottom-2 -right-2 w-10 h-10 bg-gradient-to-br from-amber-600 to-orange-700 rounded-full flex items-center justify-center border-2 border-slate-900 shadow-lg">
                    <span className="text-sm font-bold text-white">3</span>
                  </div>
                </div>
                <h3 className="font-bold text-lg text-white mb-1">{leaders[2].username}</h3>
                <p className="text-xs text-blue-400 font-semibold mb-2">{leaders[2].institution}</p>
                <div className="bg-slate-900/60 backdrop-blur-sm border border-slate-700 rounded-xl px-4 py-2 shadow-lg">
                  <p className="text-2xl font-black text-amber-500">{leaders[2].score}</p>
                  <p className="text-xs text-slate-500">points</p>
                </div>
                {/* Podium */}
                <div className="w-full h-24 bg-gradient-to-b from-amber-600/20 to-orange-700/20 rounded-t-xl mt-4 border-t-4 border-amber-600 flex items-center justify-center">
                  <Medal className="w-12 h-12 text-amber-600 drop-shadow-lg" />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-slate-900/50 backdrop-blur-md border border-slate-800 p-5 rounded-2xl hover:border-cyan-500/50 transition-colors">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-blue-500/20 rounded-lg">
                <User className="w-5 h-5 text-blue-400" />
              </div>
              <p className="text-slate-400 text-sm font-medium">Total Players</p>
            </div>
            <p className="text-3xl font-black text-white">{leaders.length}</p>
          </div>
          
          <div className="bg-slate-900/50 backdrop-blur-md border border-slate-800 p-5 rounded-2xl hover:border-yellow-500/50 transition-colors">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-yellow-500/20 rounded-lg">
                <Trophy className="w-5 h-5 text-yellow-400" />
              </div>
              <p className="text-slate-400 text-sm font-medium">Top Score</p>
            </div>
            <p className="text-3xl font-black text-yellow-400">{leaders[0]?.score || 0}</p>
          </div>
          
          <div className="bg-slate-900/50 backdrop-blur-md border border-slate-800 p-5 rounded-2xl hover:border-green-500/50 transition-colors">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-green-500/20 rounded-lg">
                <TrendingUp className="w-5 h-5 text-green-400" />
              </div>
              <p className="text-slate-400 text-sm font-medium">Status</p>
            </div>
            <p className="text-3xl font-black text-green-400">Active</p>
          </div>
        </div>

        {/* Rest of Rankings */}
        {!isLoading && leaders.length > 3 && (
          <div className="space-y-3">
            <h2 className="text-lg font-bold text-slate-300 mb-4 flex items-center gap-2">
              <Award className="w-5 h-5 text-blue-400" />
              Other Rankings
            </h2>
            {leaders.slice(3).map((player, index) => {
              const actualIndex = index + 3;
              const style = getTopThreeStyle(actualIndex);
              
              return (
                <div 
                  key={player.sid}
                  className={`relative flex items-center justify-between p-4 rounded-xl border ${style.border} ${style.bg} backdrop-blur-sm transition-all hover:scale-[1.02] ${style.glow}`}
                >
                  <div className="flex items-center gap-4">
                    {/* Rank Number */}
                    <div className={`w-12 h-12 ${style.badge} rounded-xl flex items-center justify-center font-black text-lg shadow-lg`}>
                      {actualIndex + 1}
                    </div>

                    {/* Avatar */}
                    <div className="w-12 h-12 rounded-full bg-slate-800 border-2 border-slate-700 flex items-center justify-center shadow-lg">
                      <User className="w-6 h-6 text-slate-400" />
                    </div>

                    {/* User Info */}
                    <div>
                      <p className="font-bold text-white text-lg">{player.username}</p>
                      <p className="text-xs uppercase tracking-wide text-blue-400 font-semibold">
                        {player.institution}
                      </p>
                    </div>
                  </div>

                  {/* Score */}
                  <div className="text-right">
                    <p className="text-xs text-slate-500 uppercase tracking-widest font-medium mb-1">Points</p>
                    <p className="text-2xl font-black text-white">
                      {player.score.toLocaleString()}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="w-12 h-12 animate-spin text-cyan-500 mb-4" />
            <p className="text-slate-400">Loading leaderboard...</p>
          </div>
        )}
      </div>

      {/* Custom shimmer animation */}
      <style jsx>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
}