"use client";

/**
 * @fileoverview Displays an access denied page with animated elements and navigation options.
 * Provides clear feedback when users attempt to access unauthorised content.
 */

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";

const FADE_IN_ANIMATION = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 },
};

const EMOJI_ANIMATION = {
  initial: { scale: 0.5, rotate: -10 },
  animate: { scale: 1, rotate: 0 },
  transition: {
    type: "spring",
    stiffness: 260,
    damping: 20,
    delay: 0.2,
  },
};

/**
 * Renders an access denied page with animated elements
 * @returns {JSX.Element} The access denied component
 */
export function AccessDenied(): JSX.Element {
  return (
    <main className="container mx-auto flex flex-col items-center justify-center min-h-screen p-4 py-8">
      <motion.div
        className="text-center space-y-8 max-w-3xl"
        {...FADE_IN_ANIMATION}
      >
        <motion.div className="text-6xl mb-6" {...EMOJI_ANIMATION}>
          🚫
        </motion.div>

        <div className="space-y-6">
          <motion.h1
            className="text-5xl md:text-6xl font-bold"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.4 }}
          >
            Access Denied
          </motion.h1>

          <motion.h2
            className="text-xl md:text-2xl font-semibold"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.4 }}
          >
            Permission Required
          </motion.h2>

          <motion.p
            className="text-muted-foreground max-w-md mx-auto text-lg"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.4 }}
          >
            This recording doesn&apos;t seem to belong to you.
          </motion.p>
        </div>

        <motion.div
          className="flex flex-col sm:flex-row gap-4 justify-center mt-12"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.4 }}
        >
          <Button asChild variant="outline" size="lg" className="px-8">
            <Link href="/">Return Home</Link>
          </Button>
          <Button asChild size="lg" className="px-8">
            <Link href="/recordings">View your recordings</Link>
          </Button>
        </motion.div>
      </motion.div>
    </main>
  );
}
