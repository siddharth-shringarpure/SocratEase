'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import * as faceapi from 'face-api.js';

type Emotions = {
  neutral: number;
  happy: number;
  sad: number;
  angry: number;
  fearful: number;
  disgusted: number;
  surprised: number;
};

export default function CameraPage() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);
  const [emotions, setEmotions] = useState<Emotions>({
    neutral: 0,
    happy: 0,
    sad: 0,
    angry: 0,
    fearful: 0,
    disgusted: 0,
    surprised: 0
  });
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const loadModels = async () => {
      try {
        setModelLoading(true);
        setModelError(null);
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
          faceapi.nets.faceExpressionNet.loadFromUri('/models')
        ]);
        setModelLoading(false);
      } catch (error) {
        console.error('Error loading models:', error);
        setModelError('Failed to load face detection models. Please refresh the page.');
        setModelLoading(false);
      }
    };
    loadModels();

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const detectEmotions = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    try {
      const displaySize = {
        width: videoRef.current.width,
        height: videoRef.current.height
      };

      faceapi.matchDimensions(canvasRef.current, displaySize);

      const detections = await faceapi.detectAllFaces(
        videoRef.current,
        new faceapi.TinyFaceDetectorOptions()
      ).withFaceExpressions();

      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      canvasRef.current.getContext('2d')?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      faceapi.draw.drawDetections(canvasRef.current, resizedDetections);
      faceapi.draw.drawFaceExpressions(canvasRef.current, resizedDetections);

      if (resizedDetections.length > 0) {
        const expressions = resizedDetections[0].expressions as Emotions;
        setEmotions(expressions);
      }
    } catch (error) {
      console.error('Error detecting emotions:', error);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play();
          }
        };
      }
      streamRef.current = stream;
      setIsStreaming(true);
      
      // Start emotion detection loop
      intervalRef.current = setInterval(detectEmotions, 100);
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
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

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
          {modelLoading && (
            <div className="w-full p-4 bg-muted rounded-lg text-center">
              Loading face detection models...
            </div>
          )}
          {modelError && (
            <div className="w-full p-4 bg-destructive/10 text-destructive rounded-lg text-center">
              {modelError}
            </div>
          )}
          <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full h-full object-cover"
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full"
            />
          </div>
          <Button
            onClick={isStreaming ? stopCamera : startCamera}
            variant={isStreaming ? "destructive" : "default"}
            className="w-32"
            disabled={modelLoading || !!modelError}
          >
            {isStreaming ? 'Stop Camera' : 'Start Camera'}
          </Button>
          
          {isStreaming && Object.keys(emotions).length > 0 && (
            <div className="w-full p-4 bg-muted rounded-lg">
              <h3 className="font-semibold mb-2">Detected Emotions:</h3>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(emotions).map(([emotion, probability]) => (
                  <div key={emotion} className="flex justify-between">
                    <span className="capitalize">{emotion}:</span>
                    <span>{(probability * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </main>
  );
} 