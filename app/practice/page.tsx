"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";

// Update the SpeechRecognitionEvent interface
interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList;
  timeStamp: number;
}

interface SpeechRecognitionResultList {
  length: number;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  [index: number]: {
    transcript: string;
  };
}

interface SpeechRecognition {
  continuous: boolean;
  interimResults: boolean;
  onresult: (event: SpeechRecognitionEvent) => void;
  start: () => void;
  stop: () => void;
}

declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognition;
    webkitSpeechRecognition: new () => SpeechRecognition;
  }
}

interface ConversationMode {
  id: string;
  name: string;
  description: string;
  tips: string[];
  emoji: string;
}

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

const modeCardVariants = {
  initial: { opacity: 0, scale: 0.9 },
  animate: { opacity: 1, scale: 1 },
  hover: { 
    scale: 1.02,
    borderColor: "hsl(var(--primary) / 0.5)",
    transition: { duration: 0.2 }
  },
  tap: { scale: 0.98 }
};

const pulseVariants = {
  initial: { scale: 1 },
  animate: {
    scale: [1, 1.1, 1],
    transition: {
      duration: 2,
      repeat: Infinity,
      ease: "easeInOut"
    }
  }
};

export default function PracticePage() {
  const [isRecording, setIsRecording] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(1);
  const [speechRate, setSpeechRate] = useState<number | null>(null);
  const [speed, setSpeed] = useState([0]); // Default to normal speed
  const [selectedMode, setSelectedMode] = useState<string | null>(null);
  const [showPractice, setShowPractice] = useState(false);
  const totalQuestions = 6;

  const conversationModes: ConversationMode[] = [
    {
      id: "persuasive",
      name: "Persuasive",
      description: "Learn to convince and influence others effectively",
      emoji: "🎯",
      tips: [
        "Use concrete evidence and examples",
        "Address potential counterarguments",
        "Maintain a confident and assertive tone"
      ]
    },
    {
      id: "emotive",
      name: "Emotive",
      description: "Express feelings and emotions clearly",
      emoji: "💝",
      tips: [
        "Use appropriate emotional language",
        "Match your tone to the emotion",
        "Practice empathetic responses"
      ]
    },
    {
      id: "public-speaking",
      name: "Public Speaking",
      description: "Master speaking in front of audiences",
      emoji: "🎤",
      tips: [
        "Project your voice clearly",
        "Use engaging body language",
        "Structure your speech with clear points"
      ]
    },
    {
      id: "rizzing",
      name: "Rizzing",
      description: "Practice charismatic and engaging conversation",
      emoji: "✨",
      tips: [
        "Stay confident and authentic",
        "Use appropriate humor",
        "Read and respond to social cues"
      ]
    },
    {
      id: "basic-conversations",
      name: "Basic Conversations",
      description: "Everyday casual interactions",
      emoji: "💬",
      tips: [
        "Keep the conversation flowing naturally",
        "Ask open-ended questions",
        "Show genuine interest"
      ]
    },
    {
      id: "formal-conversations",
      name: "Formal Conversations",
      description: "Professional and business communication",
      emoji: "👔",
      tips: [
        "Maintain professional language",
        "Be concise and clear",
        "Use appropriate formal expressions"
      ]
    },
    {
      id: "debating",
      name: "Debating",
      description: "Structured argument and discussion",
      emoji: "⚖️",
      tips: [
        "Present logical arguments",
        "Listen actively to counterpoints",
        "Support claims with evidence"
      ]
    },
    {
      id: "storytelling",
      name: "Storytelling",
      description: "Engaging narrative communication",
      emoji: "📚",
      tips: [
        "Set up a clear narrative structure",
        "Use descriptive language",
        "Maintain audience engagement"
      ]
    }
  ];

  let recognition: SpeechRecognition | null = null;

  useEffect(() => {
    if (typeof window !== "undefined") {
      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;

        recognition.onresult = (event) => {
          const last = event.results.length - 1;
          const words = event.results[last][0].transcript.split(" ").length;
          const duration = event.timeStamp / 1000; // convert to seconds
          const wordsPerMinute = (words / duration) * 60;
          setSpeechRate(Math.round(wordsPerMinute));
        };
      }
    }
  }, []);

  const handleStartRecording = () => {
    setIsRecording(true);
    recognition?.start();
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    recognition?.stop();
  };

  const handleModeSelect = (modeId: string) => {
    setSelectedMode(modeId);
  };

  const handleStartPractice = () => {
    setShowPractice(true);
  };

  const selectedModeData = selectedMode 
    ? conversationModes.find(mode => mode.id === selectedMode)
    : null;

  return (
    <motion.main
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="container max-w-4xl mx-auto p-4 py-8"
    >
      <AnimatePresence mode="wait">
        <div className="space-y-8">
          {/* Mode Selection */}
          {!selectedMode && (
            <motion.div
              key="mode-selection"
              {...fadeIn}
              transition={{ duration: 0.5 }}
            >
              <Card className="border-2">
                <CardHeader>
                  <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                  >
                    <CardTitle className="text-center text-2xl">Choose Your Practice Mode</CardTitle>
                    <CardDescription className="text-center">
                      Select what type of conversation you'd like to practice
                    </CardDescription>
                  </motion.div>
                </CardHeader>
                <CardContent>
                  <motion.div
                    variants={staggerContainer}
                    initial="initial"
                    animate="animate"
                    className="grid grid-cols-1 md:grid-cols-2 gap-6"
                  >
                    {conversationModes.map((mode: ConversationMode, index: number) => (
                      <motion.button
                        key={mode.id}
                        variants={modeCardVariants}
                        initial="initial"
                        animate="animate"
                        whileHover="hover"
                        whileTap="tap"
                        custom={index}
                        onClick={() => handleModeSelect(mode.id)}
                        className={`w-full group relative h-auto p-6 text-left flex flex-col items-center gap-4 rounded-md border-2 bg-background transition-colors
                          ${selectedMode === mode.id ? 'border-primary' : 'border-input'}
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
          )}

          {/* Tips and Start Page */}
          {selectedMode && !showPractice && selectedModeData && (
            <motion.div
              key="tips-page"
              {...fadeIn}
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
                  <CardTitle className="text-2xl">{selectedModeData.name} Practice</CardTitle>
                  <CardDescription>
                    Review these tips before you begin
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <motion.div className="space-y-6">
                    <motion.div
                      variants={staggerContainer}
                      initial="initial"
                      animate="animate"
                      className="space-y-4 bg-muted/30 p-6 rounded-lg"
                    >
                      {selectedModeData.tips.map((tip, index) => (
                        <motion.div
                          key={index}
                          variants={fadeIn}
                          custom={index}
                          className="flex items-start gap-3 hover:bg-muted/50 p-2 rounded transition-colors"
                        >
                          <span className="text-primary text-xl">•</span>
                          <p className="text-base">{tip}</p>
                        </motion.div>
                      ))}
                    </motion.div>
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className="w-full bg-primary text-primary-foreground hover:bg-primary/90 text-lg py-6 px-4 rounded-md font-medium"
                      onClick={handleStartPractice}
                    >
                      Start Practice
                    </motion.button>
                  </motion.div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Practice Session */}
          {showPractice && (
            <motion.div
              key="practice-session"
              {...fadeIn}
              transition={{ duration: 0.5 }}
            >
              <Card className="border-2">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      <motion.span
                        animate={{ scale: isRecording ? [1, 1.2, 1] : 1 }}
                        transition={{ duration: 1, repeat: isRecording ? Infinity : 0 }}
                        className="text-primary"
                      >
                        ●
                      </motion.span>
                      {isRecording ? "Listening..." : "Ready"}
                    </CardTitle>
                    <div className="flex items-center gap-2">
                      <motion.span
                        className="text-2xl"
                        animate={{ rotate: [0, 10, -10, 0] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      >
                        {selectedModeData?.emoji}
                      </motion.span>
                      <span className="text-sm text-muted-foreground">
                        Question {currentQuestion} of {totalQuestions}
                      </span>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    {/* Recording Visualization */}
                    <div className="aspect-[3/2] bg-muted/30 rounded-lg flex items-center justify-center">
                      <motion.div
                        variants={pulseVariants}
                        initial="initial"
                        animate={isRecording ? "animate" : "initial"}
                        className={`w-24 h-24 rounded-full bg-primary/10 flex items-center justify-center`}
                      >
                        <motion.div
                          variants={pulseVariants}
                          initial="initial"
                          animate={isRecording ? "animate" : "initial"}
                          className={`w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center`}
                        >
                          <motion.div
                            variants={pulseVariants}
                            initial="initial"
                            animate={isRecording ? "animate" : "initial"}
                            className={`w-8 h-8 rounded-full bg-primary`}
                          />
                        </motion.div>
                      </motion.div>
                    </div>

                    {/* Controls */}
                    <div className="flex justify-center gap-4">
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={isRecording ? handleStopRecording : handleStartRecording}
                        className={`text-lg py-6 px-8 rounded-md font-medium ${
                          isRecording 
                            ? 'bg-destructive text-destructive-foreground hover:bg-destructive/90'
                            : 'bg-primary text-primary-foreground hover:bg-primary/90'
                        }`}
                      >
                        {isRecording ? "Stop" : "Start"}
                      </motion.button>
                    </div>

                    {/* Mode-specific tips */}
                    {selectedModeData && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="space-y-2 bg-muted/30 p-4 rounded-lg"
                      >
                        {selectedModeData.tips.map((tip, index) => (
                          <motion.p
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.1 * index }}
                            className="flex items-center gap-2 text-sm text-muted-foreground"
                          >
                            <span className="text-primary">•</span> {tip}
                          </motion.p>
                        ))}
                      </motion.div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {speechRate && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-sm text-muted-foreground mt-2"
            >
              Speaking rate: {speechRate} words per minute
            </motion.div>
          )}

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="space-y-2"
          >
            <label className="text-sm font-medium">Speech Speed</label>
            <Slider
              value={speed}
              onValueChange={setSpeed}
              min={-1}
              max={1}
              step={0.1}
            />
            <div className="text-sm text-muted-foreground text-center">
              {speed[0] > 0 ? "+" : ""}
              {speed[0].toFixed(1)}x
            </div>
          </motion.div>
        </div>
      </AnimatePresence>
    </motion.main>
  );
}
