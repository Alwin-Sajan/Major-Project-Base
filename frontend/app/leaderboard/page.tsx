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
  User
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

  const getRankStyle = (index: number) => {
    switch (index) {
      case 0: return { icon: <Trophy className="w-6 h-6 text-yellow-400" />, color: "from-yellow-500/20 to-transparent", border: "border-yellow-500/50" };
      case 1: return { icon: <Medal className="w-6 h-6 text-slate-300" />, color: "from-slate-400/10 to-transparent", border: "border-slate-400/40" };
      case 2: return { icon: <Medal className="w-6 h-6 text-amber-600" />, color: "from-amber-700/10 to-transparent", border: "border-amber-700/40" };
      default: return { icon: null, color: "bg-slate-900/40", border: "border-slate-800" };
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black to-slate-900 text-white font-sans p-6">
      {/* Background Glows */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] right-[-10%] w-[500px] h-[500px] bg-blue-600/10 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[-10%] left-[-10%] w-[500px] h-[500px] bg-cyan-600/10 rounded-full blur-[120px]"></div>
      </div>

      <div className="max-w-3xl mx-auto relative">
        {/* Header */}
        <div className="flex items-center justify-between mb-12">
          <Link href="/" className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors group">
            <ChevronLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
            <span>Back to Game</span>
          </Link>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500 rounded-lg">
              <Waves className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
              Global Leaderboard
            </h1>
          </div>
        </div>

        {/* Stats Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-2xl backdrop-blur-md">
                <p className="text-slate-500 text-xs uppercase tracking-widest mb-1">Total Players</p>
                <p className="text-xl font-bold">{leaders.length}</p>
            </div>
            <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-2xl backdrop-blur-md">
                <p className="text-slate-500 text-xs uppercase tracking-widest mb-1">Top Score</p>
                <p className="text-xl font-bold text-yellow-400">{leaders[0]?.score || 0}</p>
            </div>
            <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-2xl backdrop-blur-md">
                <p className="text-slate-500 text-xs uppercase tracking-widest mb-1">Community Rank</p>
                <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-green-400" />
                    <p className="text-xl font-bold text-white">Active</p>
                </div>
            </div>
        </div>

        {/* List */}
        <div className="space-y-3">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-20 opacity-50">
              <Loader2 className="w-10 h-10 animate-spin text-blue-500 mb-4" />
              <p>Syncing Ranks...</p>
            </div>
          ) : (
            leaders.map((player, index) => {
              const style = getRankStyle(index);
              return (
                <div 
                  key={player.sid}
                  className={`relative flex items-center justify-between p-4 rounded-2xl border ${style.border} bg-gradient-to-r ${style.color} backdrop-blur-sm transition-all hover:scale-[1.01]`}
                >
                  <div className="flex items-center gap-4">
                    {/* Rank Number or Trophy */}
                    <div className="w-10 flex justify-center">
                      {style.icon ? style.icon : <span className="text-slate-500 font-mono font-bold">{index + 1}</span>}
                    </div>

                    {/* Avatar Placeholder */}
                    <div className="w-10 h-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
                        <User className="w-5 h-5 text-slate-500" />
                    </div>

                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-slate-100">{player.username}</span>
                        {index === 0 && <Crown className="w-3 h-3 text-yellow-500 fill-yellow-500" />}
                      </div>
                      <span className="text-[10px] uppercase tracking-tighter text-blue-400 font-semibold">
                        {player.institution}
                      </span>
                    </div>
                  </div>

                  <div className="text-right">
                    <p className="text-xs text-slate-500 uppercase tracking-widest font-medium">Points</p>
                    <p className={`text-xl font-black ${index === 0 ? 'text-yellow-400' : 'text-white'}`}>
                      {player.score.toLocaleString()}
                    </p>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}