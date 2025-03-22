"use client";

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function TranscribePage() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = handleStop;
      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleStop = async () => {
    setIsLoading(true);
    try {
      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });

      const response = await fetch("/api/speech2text", {
        method: "POST",
        body: audioBlob,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      setTranscript(data.text);
    } catch (error) {
      console.error("Error transcribing audio:", error);
      setTranscript("Error transcribing audio. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Speech to Text</h1>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Record Audio</CardTitle>
          <CardDescription>
            Click the button below to start recording your voice
          </CardDescription>
        </CardHeader>
        <CardContent className="flex justify-center gap-4">
          {!isRecording ? (
            <Button onClick={startRecording}>Start Recording</Button>
          ) : (
            <Button onClick={stopRecording} variant="destructive">
              Stop Recording
            </Button>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Transcript</CardTitle>
          <CardDescription>
            Your speech will appear here after recording
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="text-center py-4">Transcribing...</div>
          ) : (
            <div className="p-4 border rounded-md min-h-[100px] whitespace-pre-wrap">
              {transcript ||
                "No transcript yet. Record some audio to see the result."}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
