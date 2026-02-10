// app/login/page.tsx
'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FishIcon, LogInIcon, UserPlusIcon, EyeIcon, EyeOffIcon, MailIcon, LockIcon, UserIcon } from 'lucide-react';
import Link from 'next/link';

export default function LoginPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [userType, setUserType] = useState<'student' | 'admin' | null>(null);
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    fullName: '',
    institution: '',
  });

  const containerVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: { duration: 0.5, ease: 'easeOut' },
    },
    exit: {
      opacity: 0,
      scale: 0.95,
      transition: { duration: 0.3 },
    },
  };

  const inputVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: (index: number) => ({
      opacity: 1,
      x: 0,
      transition: { delay: index * 0.1, duration: 0.4 },
    }),
  };

  const buttonVariants = {
    hover: { scale: 1.02, boxShadow: '0 10px 30px rgba(34, 197, 94, 0.3)' },
    tap: { scale: 0.98 },
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      // Handle login/signup logic here
      console.log('Form submitted:', { ...formData, userType });
    }, 1500);
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-900 via-slate-800 to-blue-900 flex items-center justify-center p-4 relative overflow-hidden">
      {/* Animated background elements */}
      <motion.div
        className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20"
        animate={{
          y: [0, 100, 0],
          x: [0, 50, 0],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
      <motion.div
        className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20"
        animate={{
          y: [0, -100, 0],
          x: [0, -50, 0],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />

      <AnimatePresence mode="wait">
        {/* User Type Selection Screen */}
        {!userType ? (
          <motion.div
            key="userType"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="w-full max-w-2xl"
          >
            <motion.div
              className="text-center mb-12"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="flex items-center justify-center gap-3 mb-4">
                <FishIcon className="w-10 h-10 text-cyan-400" />
                <Link href="/" className="text-5xl z-40 md:text-6xl font-black bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                  Marine.AI
                </Link>
              </div>
              <p className="text-cyan-200 text-lg">Welcome Back</p>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Student Card */}
              <motion.button
                onClick={() => {
                  setUserType('student');
                  setIsLogin(true);
                }}
                className="relative group"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-2xl opacity-75 group-hover:opacity-100 blur-xl transition duration-300"
                  animate={{ opacity: [0.5, 0.8, 0.5] }}
                  transition={{ duration: 3, repeat: Infinity }}
                />
                <div className="relative bg-slate-800 border-2 border-cyan-400/50 rounded-2xl p-8 hover:border-cyan-400 transition duration-300">
                  <div className="flex flex-col items-center gap-4">
                    <motion.div
                      animate={{ y: [0, -10, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      <UserIcon className="w-16 h-16 text-cyan-400" />
                    </motion.div>
                    <div>
                      <h3 className="text-2xl font-bold text-white mb-2">Student</h3>
                      <p className="text-slate-300">Learn & Explore marine species</p>
                    </div>
                    <motion.div
                      className="text-cyan-400 text-sm font-semibold"
                      animate={{ x: [0, 5, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      Click to continue →
                    </motion.div>
                  </div>
                </div>
              </motion.button>

              {/* Admin Card */}
              <motion.button
                onClick={() => {
                  setUserType('admin');
                  setIsLogin(true);
                }}
                className="relative group"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl opacity-75 group-hover:opacity-100 blur-xl transition duration-300"
                  animate={{ opacity: [0.5, 0.8, 0.5] }}
                  transition={{ duration: 3, repeat: Infinity, delay: 0.5 }}
                />
                <div className="relative bg-slate-800 border-2 border-purple-400/50 rounded-2xl p-8 hover:border-purple-400 transition duration-300">
                  <div className="flex flex-col items-center gap-4">
                    <motion.div
                      animate={{ y: [0, -10, 0] }}
                      transition={{ duration: 2, repeat: Infinity, delay: 0.2 }}
                    >
                      <LockIcon className="w-16 h-16 text-purple-400" />
                    </motion.div>
                    <div>
                      <h3 className="text-2xl font-bold text-white mb-2">Admin</h3>
                      <p className="text-slate-300">Manage & Monitor system</p>
                    </div>
                    <motion.div
                      className="text-purple-400 text-sm font-semibold"
                      animate={{ x: [0, 5, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      Click to continue →
                    </motion.div>
                  </div>
                </div>
              </motion.button>
            </div>
          </motion.div>
        ) : (
          /* Login/Signup Form */
          <motion.div
            key="form"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="w-full max-w-md"
          >
            {/* Header */}
            <motion.div
              className="text-center mb-8"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="flex items-center justify-center gap-3 mb-4">
                <FishIcon className="w-8 h-8 text-cyan-400" />
                <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                  Marine.AI
                </h1>
              </div>
              <div className="flex items-center justify-center gap-2 mb-4">
                <button
                  onClick={() => setUserType(null)}
                  className="text-cyan-400 hover:text-cyan-300 transition"
                >
                  ← Back
                </button>
                <p className="text-slate-300">
                  {userType === 'admin' ? '👨‍💼 Admin' : '👨‍🎓 Student'}
                </p>
              </div>
              <p className="text-slate-400">
                {isLogin ? 'Sign in to your account' : 'Create a new account'}
              </p>
            </motion.div>

            {/* Form Card */}
            <motion.div
              className="bg-slate-800/50 backdrop-blur-md border border-cyan-400/30 rounded-2xl p-8 shadow-2xl"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <form onSubmit={handleSubmit} className="space-y-4">
                {/* Full Name (Sign Up Only) */}
                <AnimatePresence>
                  {!isLogin && (
                    <motion.div
                      custom={0}
                      variants={inputVariants}
                      initial="hidden"
                      animate="visible"
                    >
                      <label className="block text-slate-300 text-sm font-semibold mb-2">
                        Full Name
                      </label>
                      <div className="relative">
                        <UserIcon className="absolute left-3 top-3.5 w-5 h-5 text-cyan-400" />
                        <input
                          type="text"
                          name="fullName"
                          value={formData.fullName}
                          onChange={handleInputChange}
                          placeholder="John Doe"
                          className="w-full pl-10 pr-4 py-2.5 bg-slate-700/50 border border-cyan-400/30 rounded-lg focus:border-cyan-400 focus:outline-none transition text-white placeholder-slate-500"
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Institution (Sign Up Only) */}
                <AnimatePresence>
                  {!isLogin && (
                    <motion.div
                      custom={1}
                      variants={inputVariants}
                      initial="hidden"
                      animate="visible"
                    >
                      <label className="block text-slate-300 text-sm font-semibold mb-2">
                        Institution
                      </label>
                      <div className="relative">
                        <UserIcon className="absolute left-3 top-3.5 w-5 h-5 text-cyan-400" />
                        <input
                          type="text"
                          name="institution"
                          value={formData.institution}
                          onChange={handleInputChange}
                          placeholder="Your Institution"
                          className="w-full pl-10 pr-4 py-2.5 bg-slate-700/50 border border-cyan-400/30 rounded-lg focus:border-cyan-400 focus:outline-none transition text-white placeholder-slate-500"
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Email */}
                <motion.div
                  custom={isLogin ? 0 : 2}
                  variants={inputVariants}
                  initial="hidden"
                  animate="visible"
                >
                  <label className="block text-slate-300 text-sm font-semibold mb-2">
                    Email
                  </label>
                  <div className="relative">
                    <MailIcon className="absolute left-3 top-3.5 w-5 h-5 text-cyan-400" />
                    <input
                      type="email"
                      name="email"
                      value={formData.email}
                      onChange={handleInputChange}
                      placeholder="you@example.com"
                      className="w-full pl-10 pr-4 py-2.5 bg-slate-700/50 border border-cyan-400/30 rounded-lg focus:border-cyan-400 focus:outline-none transition text-white placeholder-slate-500"
                    />
                  </div>
                </motion.div>

                {/* Password */}
                <motion.div
                  custom={isLogin ? 1 : 3}
                  variants={inputVariants}
                  initial="hidden"
                  animate="visible"
                >
                  <label className="block text-slate-300 text-sm font-semibold mb-2">
                    Password
                  </label>
                  <div className="relative">
                    <LockIcon className="absolute left-3 top-3.5 w-5 h-5 text-cyan-400" />
                    <input
                      type={showPassword ? 'text' : 'password'}
                      name="password"
                      value={formData.password}
                      onChange={handleInputChange}
                      placeholder="••••••••"
                      className="w-full pl-10 pr-12 py-2.5 bg-slate-700/50 border border-cyan-400/30 rounded-lg focus:border-cyan-400 focus:outline-none transition text-white placeholder-slate-500"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-3.5 text-cyan-400 hover:text-cyan-300 transition"
                    >
                      {showPassword ? (
                        <EyeOffIcon className="w-5 h-5" />
                      ) : (
                        <EyeIcon className="w-5 h-5" />
                      )}
                    </button>
                  </div>
                </motion.div>

                {/* Confirm Password (Sign Up Only) */}
                <AnimatePresence>
                  {!isLogin && (
                    <motion.div
                      custom={4}
                      variants={inputVariants}
                      initial="hidden"
                      animate="visible"
                    >
                      <label className="block text-slate-300 text-sm font-semibold mb-2">
                        Confirm Password
                      </label>
                      <div className="relative">
                        <LockIcon className="absolute left-3 top-3.5 w-5 h-5 text-cyan-400" />
                        <input
                          type={showPassword ? 'text' : 'password'}
                          name="confirmPassword"
                          value={formData.confirmPassword}
                          onChange={handleInputChange}
                          placeholder="••••••••"
                          className="w-full pl-10 pr-4 py-2.5 bg-slate-700/50 border border-cyan-400/30 rounded-lg focus:border-cyan-400 focus:outline-none transition text-white placeholder-slate-500"
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Remember Me (Login Only) */}
                <AnimatePresence>
                  {isLogin && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="flex items-center gap-2"
                    >
                      <input
                        type="checkbox"
                        id="remember"
                        className="w-4 h-4 bg-slate-700 border border-cyan-400/30 rounded cursor-pointer"
                      />
                      <label htmlFor="remember" className="text-slate-400 text-sm cursor-pointer">
                        Remember me
                      </label>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Submit Button */}
                <motion.button
                  type="submit"
                  disabled={loading}
                  variants={buttonVariants}
                  whileHover="hover"
                  whileTap="tap"
                  className="w-full mt-6 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-bold rounded-lg transition duration-300 disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <motion.div
                        className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                      />
                      Processing...
                    </>
                  ) : (
                    <>
                      {isLogin ? <LogInIcon className="w-5 h-5" /> : <UserPlusIcon className="w-5 h-5" />}
                      {isLogin ? 'Sign In' : 'Create Account'}
                    </>
                  )}
                </motion.button>
              </form>

              {/* Toggle Login/Signup */}
              <motion.div
                className="mt-6 text-center border-t border-cyan-400/20 pt-6"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
              >
                <p className="text-slate-400 text-sm">
                  {isLogin ? "Don't have an account?" : 'Already have an account?'}{' '}
                  <button
                    onClick={() => setIsLogin(!isLogin)}
                    className="text-cyan-400 hover:text-cyan-300 font-semibold transition"
                  >
                    {isLogin ? 'Sign Up' : 'Sign In'}
                  </button>
                </p>
              </motion.div>

              {/* Forgot Password (Login Only) */}
              <AnimatePresence>
                {isLogin && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="mt-4 text-center"
                  >
                    <Link href="#" className="text-cyan-400 hover:text-cyan-300 text-sm transition">
                      Forgot password?
                    </Link>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}