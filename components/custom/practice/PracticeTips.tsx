"use client";

/**
 * @fileoverview Displays practice tips with animated transitions for conversation modes.
 * Provides visual feedback and guidance to users before starting practice sessions.
 */

import { motion } from "framer-motion";
import { ConversationMode } from "@/types/practice";

const FADE_IN_ANIMATION = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

const TIP_ANIMATION = {
  initial: { opacity: 0, x: -20 },
  animate: { opacity: 1, x: 0 },
};

interface PracticeTipsProps {
  selectedModeData: ConversationMode;
}

/**
 * Renders practice tips with staggered animations
 * @param {PracticeTipsProps} props - Component configuration
 * @returns {JSX.Element} Animated tips display
 */
export function PracticeTips({
  selectedModeData,
}: PracticeTipsProps): JSX.Element {
  return (
    <motion.div
      {...FADE_IN_ANIMATION}
      transition={{ delay: 0.3 }}
      className="space-y-2 bg-muted/30 p-4 rounded-lg"
    >
      <h3 className="font-semibold mb-2">Practice Tips:</h3>
      {selectedModeData.tips.map((tip: string, index: number) => (
        <motion.p
          key={index}
          variants={TIP_ANIMATION}
          initial="initial"
          animate="animate"
          transition={{ delay: 0.1 * index }}
          className="flex items-center gap-2 text-sm text-muted-foreground"
        >
          <span className="text-primary">•</span> {tip}
        </motion.p>
      ))}
    </motion.div>
  );
}
