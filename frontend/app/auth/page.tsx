'use client';

import { motion } from 'framer-motion';
import { FishIcon, UserIcon, LockIcon } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

export default function LoginPage() {
  const router = useRouter();

  const containerVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: { opacity: 1, scale: 1, transition: { duration: 0.5, ease: 'easeOut' } },
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-900 via-slate-800 to-blue-900 flex items-center justify-center p-4 relative overflow-hidden">
      <motion.div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20" animate={{ y: [0, 100, 0], x: [0, 50, 0] }} transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }} />
      <motion.div className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20" animate={{ y: [0, -100, 0], x: [0, -50, 0] }} transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }} />

      <motion.div variants={containerVariants} initial="hidden" animate="visible" className="w-full max-w-2xl z-10">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <FishIcon className="w-10 h-10 text-cyan-400" />
            <Link href="/" className="text-5xl md:text-6xl font-black bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
              Marine.AI
            </Link>
          </div>
          <p className="text-cyan-200 text-lg">Welcome Back</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <motion.button onClick={() => router.push('/auth/student')} className="relative group" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-2xl opacity-75 group-hover:opacity-100 blur-xl transition duration-300" />
            <div className="relative bg-slate-800 border-2 border-cyan-400/50 rounded-2xl p-8 hover:border-cyan-400 transition duration-300">
              <div className="flex flex-col items-center gap-4">
                <UserIcon className="w-16 h-16 text-cyan-400" />
                <h3 className="text-2xl font-bold text-white">Student</h3>
                <p className="text-slate-300 text-center text-sm">Learn & Explore marine species</p>
                <div className="text-cyan-400 text-sm font-semibold">Click to continue →</div>
              </div>
            </div>
          </motion.button>

          <motion.button onClick={() => router.push('/auth/admin')} className="relative group" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl opacity-75 group-hover:opacity-100 blur-xl transition duration-300" />
            <div className="relative bg-slate-800 border-2 border-purple-400/50 rounded-2xl p-8 hover:border-purple-400 transition duration-300">
              <div className="flex flex-col items-center gap-4">
                <LockIcon className="w-16 h-16 text-purple-400" />
                <h3 className="text-2xl font-bold text-white">Admin</h3>
                <p className="text-slate-300 text-center text-sm">Manage & Monitor system</p>
                <div className="text-purple-400 text-sm font-semibold">Click to continue →</div>
              </div>
            </div>
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
}