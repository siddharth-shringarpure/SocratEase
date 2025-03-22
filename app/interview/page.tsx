"use client";

import { useState } from "react";
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

export default function InterviewPage() {
  const [isRecording, setIsRecording] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(1);
  const totalQuestions = 6;

  const handleStartRecording = () => {
    setIsRecording(true);
    // TODO: Implement recording logic
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    // TODO: Implement stop recording logic
  };

  return (
    <main className="container max-w-4xl mx-auto p-4 py-8">
      <div className="space-y-8">
        {/* Position Selection */}
        <Card className="border-2">
          <CardHeader>
            <CardTitle>Interview Practice</CardTitle>
            <CardDescription>
              What position would you like to interview for?
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <Label htmlFor="position">Position</Label>
                <Input
                  id="position"
                  placeholder="e.g. Software Engineer, Product Manager"
                  className="mt-1.5"
                />
              </div>
              <Button className="w-full">Start Interview</Button>
            </div>
          </CardContent>
        </Card>

        {/* Interview Session */}
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

              {/* Feedback Area */}
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>• Make sure your volume is up</p>
                <p>• Speak clearly and at a natural pace</p>
                <p>• Position yourself in a well-lit environment</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
