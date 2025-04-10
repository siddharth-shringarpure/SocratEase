"use client";

/**
 * @fileoverview Developer testing page for audio transcription with Whisper.
 */

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

// Audio recording configuration
const AUDIO_CONFIG = {
  channelCount: 1, // Mono for better speech recognition
  sampleRate: 44100, // Standard sample rate
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

const RECORDER_OPTIONS = {
  mimeType: "audio/webm",
  audioBitsPerSecond: 320000, // High quality audio
};

/**
 * Audio transcription page component with recording and real-time display
 * @returns {JSX.Element} The rendered transcription page
 */
export default function TranscribePage(): JSX.Element {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [transcript, setTranscript] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [recordingTime, setRecordingTime] = useState<number>(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const startRecording = async (): Promise<void> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: AUDIO_CONFIG,
      });

      const mediaRecorder = new MediaRecorder(stream, RECORDER_OPTIONS);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event): void => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = handleStop;
      mediaRecorder.start(1000); // collect chunks every second
      setIsRecording(true);
      setRecordingTime(0);

      // Start recording timer
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      // TODO: Add proper error handling UI feedback
    }
  };

  const stopRecording = (): void => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream
        .getTracks()
        .forEach((track) => track.stop());
      setIsRecording(false);

      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const handleStop = async (): Promise<void> => {
    setIsLoading(true);

    try {
      const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" });
      console.log("Audio blob size:", audioBlob.size, "bytes"); // helpful for debugging

      const formData = new FormData();
      formData.append("file", audioBlob, "recording.webm");

      const response = await fetch("/api/speech2text", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.error || `Server responded with ${response.status}`
        );
      }

      const data = await response.json();
      if (!data.text) {
        throw new Error("No transcription received");
      }

      setTranscript(data.text);
    } catch (error) {
      console.error("transcription error:", error);
      setTranscript(
        error instanceof Error
          ? error.message
          : "Error transcribing audio. Please try again."
      );
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Formats seconds into MM:SS display format
   * @param {number} seconds - Time in seconds to format
   * @returns {string} Formatted time string
   */
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Audio to Text</h1>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Record Audio</CardTitle>
          <CardDescription>
            Click the button below to start recording audio for transcription
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-4">
          <div className="w-full text-center p-4 rounded-lg bg-muted">
            {isRecording ? (
              <div className="flex items-center justify-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
                <span>Recording... {formatTime(recordingTime)}</span>
              </div>
            ) : (
              <span>Ready to record</span>
            )}
          </div>

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
            The transcribed text will appear here after recording
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="text-center py-4">
              <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-2" />
              Transcribing...
            </div>
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
