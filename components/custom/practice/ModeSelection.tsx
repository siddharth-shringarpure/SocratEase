"use client";

/**
 * @fileoverview Component for selecting conversation practice modes with animated cards
 * Provides an interactive interface for users to choose their preferred practice mode
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
      staggerChildren: 0.1, // Creates a cascading effect for child elements
    },
  },
};

// TODO: Make hover animations more subtle for better UX
const CARD_ANIMATION = {
  initial: { opacity: 0, scale: 0.9 },
  animate: { opacity: 1, scale: 1 },
  hover: {
    scale: 1.02,
    borderColor: "hsl(var(--primary) / 0.5)",
    transition: { duration: 0.2 },
  },
  tap: { scale: 0.98 },
};

interface ModeSelectionProps {
  conversationModes: ConversationMode[];
  selectedMode: string | null;
  onModeSelect: (modeId: string) => void;
}

/**
 * Renders a grid of selectable conversation mode cards
 * @param {ModeSelectionProps} props - Component properties
 * @returns {JSX.Element} Mode selection interface
 */
export function ModeSelection({
  conversationModes,
  selectedMode,
  onModeSelect,
}: ModeSelectionProps): JSX.Element {
  return (
    <motion.div
      key="mode-selection"
      {...FADE_ANIMATION}
      transition={{ duration: 0.5 }}
    >
      <Card className="border-2">
        <CardHeader>
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <CardTitle className="text-center text-2xl">
              Choose Your Practice Mode
            </CardTitle>
            <CardDescription className="text-center">
              Select what type of conversation you&apos;d like to practice
            </CardDescription>
          </motion.div>
        </CardHeader>

        <CardContent>
          <motion.div
            variants={STAGGER_ANIMATION}
            initial="initial"
            animate="animate"
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            {conversationModes.map((mode: ConversationMode, index: number) => (
              <motion.button
                key={mode.id}
                variants={CARD_ANIMATION}
                initial="initial"
                animate="animate"
                whileHover="hover"
                whileTap="tap"
                custom={index}
                onClick={() => onModeSelect(mode.id)}
                className={`
                  w-full group relative h-auto p-6 
                  text-left flex flex-col items-center gap-4 
                  rounded-md border-2 bg-background transition-colors
                  ${
                    selectedMode === mode.id ? "border-primary" : "border-input"
                  }
                `}
              >
                <motion.div
                  className="text-4xl mb-2"
                  whileHover={{ rotate: [0, -10, 10, 0] }}
                  transition={{ duration: 0.5 }}
                >
                  {mode.emoji}
                </motion.div>
                <div className="text-center">
                  <div className="font-semibold text-lg mb-2">{mode.name}</div>
                  <div className="text-sm text-muted-foreground">
                    {mode.description}
                  </div>
                </div>
              </motion.button>
            ))}
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
