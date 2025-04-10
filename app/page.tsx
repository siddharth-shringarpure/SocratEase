"use client";

/**
 * @fileoverview Landing page component for SocratEase
 */

import { motion } from "framer-motion";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import Link from "next/link";
import { useState } from "react";
import { BackendStatus } from "@/components/custom/BackendStatus";

const FADE_IN = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.8, ease: [0.22, 1, 0.36, 1] },
  },
};

const CONTAINER = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.3,
      delayChildren: 0.2,
    },
  },
};

const ARROW = {
  rest: {
    x: -4,
    opacity: 0.7,
    scale: 0.9,
  },
  hover: {
    x: 4,
    opacity: 1,
    scale: 1,
    transition: { duration: 0.3, ease: [0.22, 1, 0.36, 1] },
  },
};

interface NavigationCard {
  title: string;
  description: string;
  href: string;
  icon: string;
  color: string;
  hoverColor: string;
}

/**
 * Main landing page component showcasing key features and navigation options
 * @returns {JSX.Element} The rendered landing page
 */
export default function LandingPage(): JSX.Element {
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);

  // Navigation options with their respective details
  const navCards: NavigationCard[] = [
    {
      title: "Practice Speaking",
      description: "Get real-time AI feedback on your communication skills",
      href: "/practice",
      icon: "🎯",
      color: "from-blue-500/10 to-cyan-500/10",
      hoverColor: "from-blue-500/20 to-cyan-500/20",
    },
    {
      title: "Track Progress",
      description:
        "Visualise your improvement over time with detailed analytics",
      href: "/analytics",
      icon: "📈",
      color: "from-violet-500/10 to-purple-500/10",
      hoverColor: "from-violet-500/20 to-purple-500/20",
    },
    {
      title: "Camera Practice",
      description:
        "Practice with real-time facial expression and eye contact feedback",
      href: "/camera",
      icon: "📹",
      color: "from-green-500/10 to-emerald-500/10",
      hoverColor: "from-green-500/20 to-emerald-500/20",
    },
  ];

  return (
    <main className="container mx-auto px-4 py-4 mt-16 sm:mt-12">
      <motion.main
        initial="hidden"
        animate="visible"
        className="min-h-[calc(100vh-5rem)] flex flex-col items-center justify-center bg-gradient-to-b from-background to-background/95"
      >
        <div className="w-full max-w-4xl mx-auto">
          {/* Hero section */}
          <motion.div
            variants={FADE_IN}
            className="text-center mb-8 sm:mb-12 space-y-6 sm:space-y-8 px-4"
          >
            <motion.div
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 1, ease: [0.34, 1.56, 0.64, 1] }}
              className="inline-block text-4xl sm:text-6xl mb-2"
            >
              🗣️
            </motion.div>
            <motion.h1
              className="text-4xl sm:text-6xl md:text-7xl font-bold tracking-tight"
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{
                delay: 0.3,
                duration: 0.8,
                ease: [0.22, 1, 0.36, 1],
              }}
            >
              Socrat<span className="text-primary">Ease</span>
            </motion.h1>
            <motion.p
              className="text-lg sm:text-xl text-muted-foreground/80 font-light max-w-2xl mx-auto px-4"
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{
                delay: 0.5,
                duration: 0.8,
                ease: [0.22, 1, 0.36, 1],
              }}
            >
              Master the art of communication with AI-powered feedback and
              insights
            </motion.p>
          </motion.div>

          {/* Navigation cards grid */}
          <motion.div
            variants={CONTAINER}
            className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 sm:gap-5 px-4"
          >
            {navCards.map((card, index) => (
              <motion.div
                key={card.title}
                variants={FADE_IN}
                onHoverStart={() => setHoveredCard(index)}
                onHoverEnd={() => setHoveredCard(null)}
                className="h-full"
              >
                <Link href={card.href} className="h-full block">
                  <Card className="group relative border border-border/50 overflow-hidden transition-all duration-700 hover:border-primary/20 h-full">
                    <motion.div
                      className={`absolute inset-0 bg-gradient-to-r ${
                        hoveredCard === index ? card.hoverColor : card.color
                      } transition-opacity duration-700`}
                      initial={false}
                      animate={{ opacity: hoveredCard === index ? 1 : 0.5 }}
                    />
                    <CardHeader className="py-6 sm:py-8 relative h-full">
                      <div className="flex flex-col items-center text-center h-full">
                        <div className="h-12 sm:h-16 flex items-center justify-center">
                          <motion.span
                            className="text-3xl sm:text-4xl"
                            animate={{
                              scale: hoveredCard === index ? 1.1 : 1,
                              y: hoveredCard === index ? -2 : 0,
                            }}
                            transition={{
                              type: "spring",
                              stiffness: 200,
                              damping: 15,
                            }}
                          >
                            {card.icon}
                          </motion.span>
                        </div>
                        <div className="flex-1 flex flex-col justify-center my-3 sm:my-4">
                          <CardTitle className="text-lg sm:text-xl font-semibold mb-2 sm:mb-3">
                            {card.title}
                          </CardTitle>
                          <CardDescription className="text-sm text-muted-foreground/70 px-2">
                            {card.description}
                          </CardDescription>
                        </div>
                        <div className="h-6 sm:h-8 flex items-center justify-center">
                          <motion.svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="20"
                            height="20"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="text-primary"
                            variants={ARROW}
                            initial="rest"
                            animate={hoveredCard === index ? "hover" : "rest"}
                          >
                            <path d="M5 12h14" />
                            <path d="m12 5 7 7-7 7" />
                          </motion.svg>
                        </div>
                      </div>
                    </CardHeader>
                  </Card>
                </Link>
              </motion.div>
            ))}
          </motion.div>

          <motion.footer
            variants={FADE_IN}
            className="text-center text-sm text-muted-foreground/60 mt-8 sm:mt-12 space-y-2 px-4"
          >
            <p>Empowering better communication through AI 🚀</p>
            <br />
            <BackendStatus />
          </motion.footer>
        </div>
      </motion.main>
    </main>
  );
}
