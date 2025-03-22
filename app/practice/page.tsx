"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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

export default function PracticePage() {
  const [isRecording, setIsRecording] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(1);
  const [speechRate, setSpeechRate] = useState<number | null>(null);
  const [speed, setSpeed] = useState([0]); // Default to normal speed
  const totalQuestions = 6;

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

  return (
    <main className="container max-w-4xl mx-auto p-4 py-8">
      <div className="space-y-8">
        {/* Topic Selection */}
        <Card className="border-2">
          <CardHeader>
            <CardTitle>Communication Practice</CardTitle>
            <CardDescription>
              What would you like to practice today?
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <Label htmlFor="topic">Topic</Label>
                <Input
                  id="topic"
                  placeholder="e.g. Public Speaking, Team Presentation"
                  className="mt-1.5"
                />
              </div>
              <Button className="w-full">Start Practice</Button>
            </div>
          </CardContent>
        </Card>

        {/* Practice Session */}
        <Card className="border-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <span className="text-primary">●</span>
                {isRecording ? "Listening..." : "Ready"}
              </CardTitle>
              <span className="text-sm text-muted-foreground">
                Question {currentQuestion} of {totalQuestions}
              </span>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Recording Visualization */}
              <div className="aspect-[3/2] bg-muted/30 rounded-lg flex items-center justify-center">
                <div
                  className={`w-24 h-24 rounded-full bg-primary/10 flex items-center justify-center transition-all duration-700 ${
                    isRecording ? "scale-110 bg-primary/20" : ""
                  }`}
                >
                  <div
                    className={`w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center transition-all duration-700 ${
                      isRecording ? "scale-110 bg-primary/30" : ""
                    }`}
                  >
                    <div
                      className={`w-8 h-8 rounded-full bg-primary transition-all duration-700 ${
                        isRecording ? "scale-110" : ""
                      }`}
                    />
                  </div>
                </div>
              </div>

              {/* Controls */}
              <div className="flex justify-center gap-4">
                <Button
                  size="lg"
                  variant={isRecording ? "destructive" : "default"}
                  onClick={
                    isRecording ? handleStopRecording : handleStartRecording
                  }
                >
                  {isRecording ? "Stop" : "Start"}
                </Button>
              </div>

              {/* Tips */}
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>• Speak clearly and maintain good posture</p>
                <p>• Use appropriate gestures and body language</p>
                <p>• Keep a steady pace and natural tone</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {speechRate && (
          <div className="text-sm text-muted-foreground mt-2">
            Speaking rate: {speechRate} words per minute
          </div>
        )}

        <div className="space-y-2">
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
        </div>
      </div>
    </main>
  );
}
