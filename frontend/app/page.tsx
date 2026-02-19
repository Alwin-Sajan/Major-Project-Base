'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { useState } from 'react';
import { 
  FishIcon, BotIcon, BarChart3Icon, Gamepad2Icon, 
  SearchIcon, LogOutIcon, UserIcon, ShieldCheckIcon 
} from 'lucide-react';
// 1. Import the useAuth hook
import { useAuth } from './context/AuthContext'; 

export default function EntryPage() {
  const { user, logout } = useAuth(); // 2. Get user and logout from context
  const [hoveredButton, setHoveredButton] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const buttons = [
    {
      id: 'ood',
      href: '/ood-detection-ui',
      label: 'OOD Detection',
      description: 'Identify unknown marine species',
      icon: SearchIcon,
      color: 'from-blue-500 to-cyan-500',
      hoverColor: 'hover:shadow-blue-500/50'
    },
    {
      id: 'taxonomy',
      href: '/taxonomy-bot',
      label: 'Taxonomy Bot',
      description: 'AI-powered species information',
      icon: BotIcon,
      color: 'from-purple-500 to-pink-500',
      hoverColor: 'hover:shadow-purple-500/50'
    },
    {
      id: 'cluster',
      href: '/cluster-view-ui',
      label: 'Clustering',
      description: 'Visualize species relationships',
      icon: BarChart3Icon,
      color: 'from-green-500 to-emerald-500',
      hoverColor: 'hover:shadow-green-500/50'
    },
    {
      id: 'game',

      href: '/guess-the-species',
      label: 'Guess Game',
      description: 'Learn while playing',
      icon: Gamepad2Icon,
      color: 'from-orange-500 to-red-500',
      hoverColor: 'hover:shadow-orange-500/50'
    }
  ];

  const handleButtonClick = () => {
    setLoading(true);
    setTimeout(() => setLoading(false), 500);
  };

  // Variants for animations (keeping your existing logic)
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1, delayChildren: 0.3 } },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.8 } },
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-blue-900">
      
      {/* --- TOP RIGHT AUTH SECTION --- */}
      <motion.div
        className="absolute top-6 right-6 z-20 flex items-center gap-4"
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
      >
        {!user ? (
          // Show Login if not logged in
          <Link href="/auth">
            <motion.button
              className="px-6 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold rounded-lg shadow-lg flex items-center gap-2"
              whileHover={{ scale: 1.05, boxShadow: '0 0 20px rgba(34, 211, 238, 0.6)' }}
              whileTap={{ scale: 0.95 }}
            >
              <UserIcon className="w-4 h-4" />
              Login
            </motion.button>
          </Link>
        ) : (
          // Show Profile Info and Logout if logged in
          <div className="flex items-center gap-4 bg-slate-800/40 backdrop-blur-md p-1.5 pr-4 rounded-full border border-white/10 shadow-2xl">
            {/* Conditional Avatar Icon */}
            <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-cyan-400 to-blue-500 flex items-center justify-center shadow-inner">
              {user.type === 'student' ? (
                <UserIcon className="w-6 h-6 text-white" /> // Student Icon
              ) : (
                <ShieldCheckIcon className="w-6 h-6 text-white" /> // Scientist/Admin Icon
              )}
            </div>
            
            <div className="flex flex-col">
              <span className="text-white text-sm font-bold leading-none">{user.user}</span>
              <span className="text-cyan-400 text-[10px] uppercase tracking-wider font-medium">
                {user.type}
              </span>
            </div>

            <button 
              onClick={logout}
              className="ml-2 p-2 hover:bg-red-500/20 rounded-full text-slate-400 hover:text-red-400 transition-colors"
              title="Logout"
            >
              <LogOutIcon className="w-5 h-5" />
            </button>
          </div>
        )}
      </motion.div>

      {/* --- REST OF THE BACKGROUND AND CONTENT --- */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Your existing animated background circles... */}
        <motion.div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20" animate={{ y: [0, 100, 0], x: [0, 50, 0] }} transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }} />
        {/* ... (Repeat for other circles) ... */}
      </div>

      <motion.div
        className="relative z-10 flex flex-col items-center justify-center h-full px-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Title Section */}
        <div className="text-center mb-12">
           <motion.div className="flex items-center justify-center gap-3 mb-4">
            <FishIcon className="w-12 h-12 text-cyan-400" />
            <h1 className="text-6xl md:text-7xl font-black bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent drop-shadow-lg">
              Marine.AI
            </h1>
            <FishIcon className="w-12 h-12 text-blue-400" />
          </motion.div>
          <p className="text-xl md:text-2xl text-cyan-200 font-light">Adaptive Marine Species Identification</p>
        </div>

        {/* Buttons Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 w-full max-w-6xl">
          {buttons.map((button) => {
            const IconComponent = button.icon;
            return (
              <motion.div key={button.id} variants={itemVariants} onHoverStart={() => setHoveredButton(button.id)} onHoverEnd={() => setHoveredButton(null)}>
                <Link href={button.href} onClick={handleButtonClick}>
                   <motion.button className={`w-full h-32 rounded-2xl border-2 border-transparent bg-gradient-to-br ${button.color} p-1 relative group overflow-hidden shadow-xl ${button.hoverColor} transition-all duration-300`}>
                      <div className="absolute inset-0 bg-slate-900/30 backdrop-blur-sm rounded-xl" />
                      <div className="relative h-full w-full flex flex-col items-center justify-center gap-2 text-white">
                        <IconComponent className="w-8 h-8" />
                        <h3 className="text-base font-bold">{button.label}</h3>
                      </div>
                   </motion.button>
                </Link>
              </motion.div>
            );
          })}
        </div>
      </motion.div>

      {/* Loading Overlay (Same as before) */}
      {loading && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="text-white">Loading Marine.AI...</div>
        </div>
      )}
    </div>
  );
}