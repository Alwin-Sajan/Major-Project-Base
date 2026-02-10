// app/page.tsx
'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { useState } from 'react';
import { FishIcon, BotIcon, BarChart3Icon, Gamepad2Icon, SearchIcon, LogOutIcon } from 'lucide-react';

export default function EntryPage() {
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

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
      },
    },
  };

  const titleVariants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: {
        duration: 1,
        ease: 'easeOut',
      },
    },
  };

  const waveVariants = {
    animate: {
      y: [0, -20, 0],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: 'easeInOut',
      },
    },
  };

  const handleButtonClick = () => {
    setLoading(true);
    setTimeout(() => setLoading(false), 500);
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-blue-900">
      {/* Login Button - Top Right */}
      <motion.div
        className="absolute top-6 right-6 z-20"
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.5 }}
      >
        <Link href="/login">
          <motion.button
            className="px-6 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold rounded-lg shadow-lg hover:shadow-cyan-500/50 transition-all duration-300 flex items-center gap-2"
            whileHover={{ scale: 1.05, boxShadow: '0 0 20px rgba(34, 211, 238, 0.6)' }}
            whileTap={{ scale: 0.95 }}
          >
            <LogOutIcon className="w-4 h-4" />
            Login
          </motion.button>
        </Link>
      </motion.div>

      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
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
          className="absolute top-1/2 right-1/4 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20"
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
        <motion.div
          className="absolute bottom-0 left-1/2 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20"
          animate={{
            y: [0, 50, 0],
          }}
          transition={{
            duration: 12,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      </div>

      {/* Content */}
      <motion.div
        className="relative z-10 flex flex-col items-center justify-center h-full px-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Title Section */}
        <motion.div
          className="text-center mb-12"
          variants={titleVariants}
        >
          <motion.div
            className="flex items-center justify-center gap-3 mb-4"
            variants={waveVariants}
            animate="animate"
          >
            <FishIcon className="w-12 h-12 text-cyan-400" />
            <h1 className="text-6xl md:text-7xl font-black bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent drop-shadow-lg">
              Marine.AI
            </h1>
            <FishIcon className="w-12 h-12 text-blue-400" />
          </motion.div>
          <motion.p
            className="text-xl md:text-2xl text-cyan-200 font-light"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 1 }}
          >
            Adaptive Marine Species Identification
          </motion.p>
          <motion.p
            className="text-sm md:text-base text-blue-300 mt-2 font-light"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.7, duration: 1 }}
          >
            Discover • Identify • Learn • Explore
          </motion.p>
        </motion.div>

        {/* Divider */}
        <motion.div
          className="w-24 h-1 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full mb-12"
          initial={{ width: 0 }}
          animate={{ width: 96 }}
          transition={{ delay: 0.8, duration: 0.8 }}
        />

        {/* Buttons Grid */}
        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 w-full max-w-6xl"
          variants={containerVariants}
        >
          {buttons.map((button) => {
            const IconComponent = button.icon;
            return (
              <motion.div
                key={button.id}
                variants={itemVariants}
                onHoverStart={() => setHoveredButton(button.id)}
                onHoverEnd={() => setHoveredButton(null)}
              >
                <Link href={button.href} onClick={handleButtonClick}>
                  <motion.button
                    className={`w-full h-32 rounded-2xl border-2 border-transparent bg-gradient-to-br ${button.color} p-1 relative group overflow-hidden shadow-xl ${button.hoverColor} transition-all duration-300`}
                    whileHover={{
                      scale: 1.05,
                      boxShadow: '0 20px 40px rgba(0,0,0,0.3)',
                    }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {/* Glass morphism background */}
                    <div className="absolute inset-0 bg-slate-900/30 backdrop-blur-sm rounded-xl" />

                    {/* Content */}
                    <motion.div
                      className="relative h-full w-full flex flex-col items-center justify-center gap-2 text-white"
                      animate={{
                        y: hoveredButton === button.id ? -5 : 0,
                      }}
                      transition={{ duration: 0.3 }}
                    >
                      <motion.div
                        animate={{
                          y: hoveredButton === button.id ? -5 : 0,
                          scale: hoveredButton === button.id ? 1.2 : 1,
                        }}
                        transition={{ duration: 0.3 }}
                      >
                        <IconComponent className="w-8 h-8" />
                      </motion.div>
                      <motion.h3
                        className="text-base font-bold"
                        animate={{
                          y: hoveredButton === button.id ? -2 : 0,
                        }}
                        transition={{ duration: 0.3 }}
                      >
                        {button.label}
                      </motion.h3>
                      <motion.p
                        className="text-xs font-light text-gray-200 text-center"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{
                          opacity: hoveredButton === button.id ? 1 : 0,
                          height: hoveredButton === button.id ? 'auto' : 0,
                        }}
                        transition={{ duration: 0.3 }}
                      >
                        {button.description}
                      </motion.p>
                    </motion.div>

                    {/* Shine effect on hover */}
                    <motion.div
                      className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-0"
                      animate={{
                        opacity: hoveredButton === button.id ? 0.1 : 0,
                        x: hoveredButton === button.id ? 100 : -100,
                      }}
                      transition={{ duration: 0.6 }}
                    />
                  </motion.button>
                </Link>
              </motion.div>
            );
          })}
        </motion.div>

        {/* Loading Animation */}
        {loading && (
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="flex flex-col items-center gap-6"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
            >
              {/* Loading spinner */}
              <motion.div
                className="relative w-20 h-20"
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
              >
                <div className="absolute inset-0 rounded-full bg-gradient-to-r from-cyan-400 to-blue-600" />
                <div className="absolute inset-2 rounded-full bg-slate-900" />
              </motion.div>

              {/* Loading text */}
              <motion.p
                className="text-white text-lg font-semibold"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                Loading Marine.AI
              </motion.p>

              {/* Animated dots */}
              <div className="flex gap-2">
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    className="w-3 h-3 rounded-full bg-cyan-400"
                    animate={{ y: [0, -10, 0] }}
                    transition={{
                      duration: 1,
                      delay: i * 0.2,
                      repeat: Infinity,
                    }}
                  />
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </motion.div>

      {/* Footer */}
      <motion.footer
        className="relative z-10 absolute bottom-0 w-full text-center pb-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
      >
        <p className="text-slate-400 text-sm">
          Adaptive Marine Species Identification with Open-Set Detection & Incremental Learning
        </p>
      </motion.footer>
    </div>
  );
}