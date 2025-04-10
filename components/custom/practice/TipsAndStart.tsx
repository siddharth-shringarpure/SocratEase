"use client";

/**
 * @fileoverview Displays practice tips and a start button for conversation modes.
 * Provides animated transitions and clear instructions before starting practice sessions.
 */

import { motion } from "framer-motion";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { ConversationMode } from "@/types/practice";

const FADE_ANIMATION = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

const STAGGER_ANIMATION = {
  animate: {
    transition: {
      staggerChildren: 0.1,
    },
  },
};

interface TipsAndStartProps {
  selectedModeData: ConversationMode;
  onStartPractice: () => void;
}

/**
 * Displays practice tips and start button with animations
 * @param {TipsAndStartProps} props - Component configuration
 * @returns {JSX.Element} Tips and start interface
 */
export function TipsAndStart({
  selectedModeData,
  onStartPractice,
}: TipsAndStartProps): JSX.Element {
  return (
    <motion.div
      key="tips-page"
      {...FADE_ANIMATION}
      transition={{ duration: 0.5 }}
    >
      <Card className="border-2">
        <CardHeader className="text-center">
          <motion.div
            className="text-4xl mb-4"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1 }}
          >
            {selectedModeData.emoji}
          </motion.div>
          <CardTitle className="text-2xl">
            {selectedModeData.name} Practice
          </CardTitle>
          <CardDescription>Review these tips before you begin</CardDescription>
        </CardHeader>

        <CardContent>
          <motion.div className="space-y-6">
            {/* Tips container with staggered animation */}
            <motion.div
              variants={STAGGER_ANIMATION}
              initial="initial"
              animate="animate"
              className="space-y-4 bg-muted/30 p-6 rounded-lg"
            >
              {selectedModeData.tips.map((tip: string, index: number) => (
                <motion.div
                  key={index}
                  variants={FADE_ANIMATION}
                  custom={index}
                  className="flex items-start gap-3 hover:bg-muted/50 p-2 rounded transition-colors"
                >
                  <span className="text-primary text-xl">•</span>
                  <p className="text-base">{tip}</p>
                </motion.div>
              ))}
            </motion.div>

            {/* TODO: Add loading state for button during transition??? */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full bg-primary text-primary-foreground hover:bg-primary/90 text-lg py-6 px-4 rounded-md font-medium"
              onClick={onStartPractice}
            >
              Start Practice
            </motion.button>
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
