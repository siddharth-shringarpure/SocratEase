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
  const videoRef = useRef<HTMLVideoElement>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: {
          channelCount: 1,         // Mono audio
          sampleRate: 44100,       // 44.1kHz sample rate
          echoCancellation: true,  // Reduce echo
          noiseSuppression: true,  // Reduce background noise
          autoGainControl: true    // Automatic volume adjustment
        }
      });

      // Set up video preview
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const options = {
        mimeType: 'video/mp4',
        videoBitsPerSecond: 2500000,  // 2.5 Mbps
        audioBitsPerSecond: 320000    // 320 kbps
      };

      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = handleStop;
      mediaRecorder.start(1000); // Collect data every second
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing camera/microphone:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setIsRecording(false);
    }
  };

  const handleStop = async () => {
    setIsLoading(true);
    try {
      const videoBlob = new Blob(chunksRef.current, { type: 'video/mp4' });
      console.log('Video blob size:', videoBlob.size, 'bytes');
      
      // Create FormData and append the blob as a file
      const formData = new FormData();
      formData.append('file', videoBlob, 'recording.mp4');

      console.log('Sending transcription request...');
      const response = await fetch("http://localhost:5328/api/speech2text", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('Transcription error response:', errorData);
        throw new Error(errorData.error || `Server responded with ${response.status}`);
      }

      const data = await response.json();
      if (!data.text) {
        throw new Error('No transcription received from server');
      }
      setTranscript(data.text);
    } catch (error) {
      console.error("Error transcribing audio:", error);
      setTranscript(error instanceof Error ? error.message : "Error transcribing audio. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Video to Text</h1>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Record Video</CardTitle>
          <CardDescription>
            Click the button below to start recording video with audio
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-4">
          <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
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
            The speech from your video will appear here after recording
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="text-center py-4">Transcribing...</div>
          ) : (
            <div className="p-4 border rounded-md min-h-[100px] whitespace-pre-wrap">
              {transcript ||
                "No transcript yet. Record some video to see the result."}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
