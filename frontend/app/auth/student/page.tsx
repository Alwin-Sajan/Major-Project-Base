'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FishIcon, EyeIcon, EyeOffIcon, MailIcon, LockIcon, UserIcon } from 'lucide-react';
import { useRouter } from 'next/navigation';
// 1. Import the useAuth hook
import { useAuth } from '@/app/context/AuthContext'; 

export default function StudentAuthPage() {
  const router = useRouter();
  const { login } = useAuth(); // 2. Destructure the login function
  
  const [isLogin, setIsLogin] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    username: '',
    institution: '',
    password: '',
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    const path = isLogin ? '/login' : '/register';
    const url = `http://localhost:8000/student${path}`;

    const payload = isLogin 
      ? { email: formData.email, password: formData.password }
      : { 
          username: formData.username, 
          email: formData.email, 
          institution: formData.institution, 
          password: formData.password 
        };

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (response.ok) {
        // 3. Save to LocalStorage and Context
        // result contains { uid, user, type, message } from FastAPI
        login(result); 
        
        // 4. Redirect home
        router.push('/'); 
      } else {
        alert(result.detail || "Authentication Failed");
      }
    } catch (error) {
      alert("Backend server is not running or CORS is blocked");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-900 via-slate-800 to-blue-900 flex items-center justify-center p-4 relative overflow-hidden">
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20" />
      
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-md z-10">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <FishIcon className="w-8 h-8 text-cyan-400" />
            <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">Marine.AI</h1>
          </div>
          <div className="flex items-center justify-center gap-2 mb-4">
            <button onClick={() => router.push('/auth')} className="text-cyan-400 hover:text-cyan-300 transition">← Back</button>
            <p className="text-slate-300 uppercase tracking-widest text-sm font-bold">👨‍🎓 Student Portal</p>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-md border border-cyan-400/30 rounded-2xl p-8 shadow-2xl">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
                    <label className="block text-slate-300 text-sm font-semibold mb-2">Email</label>
                    <div className="relative">
                      <MailIcon className="absolute left-3 top-3.5 w-5 h-5 text-cyan-400" />
                      <input name="email" value={formData.email} onChange={handleInputChange} required type="email" placeholder="you@example.com" className="w-full pl-10 pr-4 py-2.5 bg-slate-700/50 border border-cyan-400/30 rounded-lg focus:border-cyan-400 focus:outline-none text-white" />
                    </div>
            </div>

            <AnimatePresence mode='wait'>
              {!isLogin && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="space-y-4 overflow-hidden">
                  <div>
              <label className="block text-slate-300 text-sm font-semibold mb-2">Username</label>
              <div className="relative">
                <UserIcon className="absolute left-3 top-3.5 w-5 h-5 text-cyan-400" />
                <input name="username" value={formData.username} onChange={handleInputChange} required type="text" placeholder="Dexter Morgan" className="w-full pl-10 pr-4 py-2.5 bg-slate-700/50 border border-cyan-400/30 rounded-lg focus:border-cyan-400 focus:outline-none text-white" />
              </div>

                  </div>
                  <div>
                    <label className="block text-slate-300 text-sm font-semibold mb-2">Institution</label>
                    <div className="relative">
                      <UserIcon className="absolute left-3 top-3.5 w-5 h-5 text-cyan-400" />
                      <input name="institution" value={formData.institution} onChange={handleInputChange} required type="text" placeholder="University Name" className="w-full pl-10 pr-4 py-2.5 bg-slate-700/50 border border-cyan-400/30 rounded-lg focus:border-cyan-400 focus:outline-none text-white" />
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div>
              <label className="block text-slate-300 text-sm font-semibold mb-2">Password</label>
              <div className="relative">
                <LockIcon className="absolute left-3 top-3.5 w-5 h-5 text-cyan-400" />
                <input name="password" value={formData.password} onChange={handleInputChange} required type={showPassword ? 'text' : 'password'} placeholder="••••••••" className="w-full pl-10 pr-12 py-2.5 bg-slate-700/50 border border-cyan-400/30 rounded-lg focus:border-cyan-400 focus:outline-none text-white" />
                <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-3.5 text-cyan-400">
                  {showPassword ? <EyeOffIcon size={20} /> : <EyeIcon size={20} />}
                </button>
              </div>
            </div>

            <button type="submit" disabled={loading} className="w-full mt-6 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-bold rounded-lg flex items-center justify-center gap-2 hover:from-cyan-400 hover:to-blue-500 transition-all disabled:opacity-50">
              {loading ? "Connecting..." : (isLogin ? "Sign In" : "Create Account")}
            </button>
          </form>

          <div className="mt-6 text-center border-t border-cyan-400/20 pt-6">
            <button onClick={() => setIsLogin(!isLogin)} className="text-cyan-400 font-semibold hover:underline">
              {isLogin ? "Don't have an account? Sign Up" : "Already have an account? Sign In"}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}