'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function CameraPage() {
  const [isStreaming, setIsStreaming] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      streamRef.current = stream;
      setIsStreaming(true);
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Failed to access camera. Please make sure you have granted camera permissions.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      streamRef.current = null;
      setIsStreaming(false);
    }
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <main className="container py-8 flex flex-col items-center">
      <h1 className="text-4xl font-bold mb-8 text-center">
        Camera Practice
      </h1>

      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle className="text-center">Camera Feed</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-4">
          <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full h-full object-cover"
            />
          </div>
          <Button
            onClick={isStreaming ? stopCamera : startCamera}
            variant={isStreaming ? "destructive" : "default"}
            className="w-32"
          >
            {isStreaming ? 'Stop Camera' : 'Start Camera'}
          </Button>
        </CardContent>
      </Card>
    </main>
  );
} 