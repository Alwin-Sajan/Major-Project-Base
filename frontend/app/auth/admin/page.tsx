'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { FishIcon, EyeIcon, EyeOffIcon, LockIcon, UserIcon, ShieldCheckIcon } from 'lucide-react';
import { useRouter } from 'next/navigation';
// 1. Import the hook
import { useAuth } from '@/app/context/AuthContext'; 

export default function AdminAuthPage() {
  const router = useRouter();
  const { login } = useAuth(); // 2. Get the login function
  
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    username: '',
    password: '',
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    // Points to your admin login endpoint
    const url = `http://localhost:8000/admin/login`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: formData.username,
          password: formData.password
        }),
      });

      const result = await response.json();

      if (response.ok) {
        // 3. Save the Admin session (uid, user, type: "admin")
        login(result); 
        
        // 4. Redirect to home
        router.push('/'); 
      } else {
        alert(result.detail || "Admin Authentication Failed");
      }
    } catch (error) {
      alert("Backend server connection failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-900 via-slate-800 to-purple-900 flex items-center justify-center p-4 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-0 right-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20" />
      
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-md z-10">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <FishIcon className="w-8 h-8 text-purple-400" />
            <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">Marine.AI</h1>
          </div>
          <div className="flex items-center justify-center gap-2 mb-4">
            <button onClick={() => router.push('/auth')} className="text-purple-400 hover:text-purple-300 transition text-sm">← Back</button>
            <p className="text-slate-300 uppercase tracking-widest text-sm font-bold flex items-center gap-2">
              <ShieldCheckIcon size={16} className="text-purple-400" /> Admin Portal
            </p>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-purple-400/30 rounded-2xl p-8 shadow-2xl">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-slate-300 text-sm font-semibold mb-2">Admin Username</label>
              <div className="relative">
                <UserIcon className="absolute left-3 top-3.5 w-5 h-5 text-purple-400" />
                <input 
                  name="username" 
                  value={formData.username}
                  onChange={handleInputChange} 
                  required 
                  type="text" 
                  placeholder="Admin" 
                  className="w-full pl-10 pr-4 py-2.5 bg-slate-700/50 border border-purple-400/30 rounded-lg focus:border-purple-400 focus:outline-none text-white placeholder:text-slate-500" 
                />
              </div>
            </div>

            <div>
              <label className="block text-slate-300 text-sm font-semibold mb-2">Security Password</label>
              <div className="relative">
                <LockIcon className="absolute left-3 top-3.5 w-5 h-5 text-purple-400" />
                <input 
                  name="password" 
                  value={formData.password}
                  onChange={handleInputChange} 
                  required 
                  type={showPassword ? 'text' : 'password'} 
                  placeholder="••••••••" 
                  className="w-full pl-10 pr-12 py-2.5 bg-slate-700/50 border border-purple-400/30 rounded-lg focus:border-purple-400 focus:outline-none text-white placeholder:text-slate-500" 
                />
                <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-3.5 text-purple-400 hover:text-purple-300 transition">
                  {showPassword ? <EyeOffIcon size={20} /> : <EyeIcon size={20} />}
                </button>
              </div>
            </div>

            <button 
              type="submit" 
              disabled={loading} 
              className="w-full mt-2 py-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white font-bold rounded-lg flex items-center justify-center gap-2 hover:from-purple-400 hover:to-pink-500 transition-all shadow-lg shadow-purple-900/20 disabled:opacity-50"
            >
              {loading ? (
                <motion.div 
                  animate={{ rotate: 360 }} 
                  transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                  className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full"
                />
              ) : "Authenticate Admin"}
            </button>
          </form>

          <div className="mt-8 text-center border-t border-purple-400/10 pt-4">
            <p className="text-slate-500 text-xs italic">
              Authorized access only. All actions are logged.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}