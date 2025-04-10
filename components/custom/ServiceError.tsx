"use client";

/**
 * @fileoverview A component that displays a service error message with animated elements,
 * providing users with feedback when the speech service encounters issues.
 */

import { motion } from "framer-motion";
import Link from "next/link";
import { Button } from "@/components/ui/button";

interface ServiceErrorProps {
  error: string;
}

/**
 * Renders an animated error page when speech services are unavailable
 * @param {ServiceErrorProps} props - Component properties
 * @returns {JSX.Element} The rendered error page
 */
export function ServiceError({ error }: ServiceErrorProps) {
  const FADE_IN_ANIMATION = {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    transition: { duration: 0.4 },
  };

  return (
    <main className="container mx-auto flex flex-col items-center justify-center min-h-screen p-4 py-8">
      {/* Main content wrapper with initial fade-in animation */}
      <motion.div
        className="text-center space-y-8 max-w-3xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Sad emoji with bounce animation  */}
        <motion.div
          className="text-6xl mb-6"
          initial={{ scale: 0.5, rotate: -10 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{
            type: "spring",
            stiffness: 260,
            damping: 20,
            delay: 0.2,
          }}
        >
          😞
        </motion.div>

        <div className="space-y-6">
          <motion.h1
            className="text-5xl md:text-6xl font-bold"
            {...FADE_IN_ANIMATION}
            transition={{ ...FADE_IN_ANIMATION.transition, delay: 0.3 }}
          >
            Service Temporarily Unavailable
          </motion.h1>

          <motion.h2
            className="text-xl md:text-2xl font-semibold"
            {...FADE_IN_ANIMATION}
            transition={{ ...FADE_IN_ANIMATION.transition, delay: 0.4 }}
          >
            Technical Difficulties
          </motion.h2>

          <motion.p
            className="text-muted-foreground max-w-md mx-auto text-lg"
            {...FADE_IN_ANIMATION}
            transition={{ ...FADE_IN_ANIMATION.transition, delay: 0.5 }}
          >
            Our customised speech feedback service is currently facing errors.
            If this issue persists, please contact the developers.
          </motion.p>
        </div>

        {/* Navigation buttons */}
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
