"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useRouter } from "next/navigation";

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
  const [isProcessing, setIsProcessing] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);
  const [uploadedAudio, setUploadedAudio] = useState<string | null>(null);
  const [separateAudioRecording, setSeparateAudioRecording] = useState(false);
  const [emotions, setEmotions] = useState<Emotions>({
    neutral: 0,
    happy: 0,
    sad: 0,
    angry: 0,
    fearful: 0,
    disgusted: 0,
    surprised: 0,
  });
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const router = useRouter();
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const lowPassNodeRef = useRef<BiquadFilterNode | null>(null);
  const highPassNodeRef = useRef<BiquadFilterNode | null>(null);

  const detectEmotions = async () => {
    if (!videoRef.current || !canvasRef.current || isProcessing) return;

    try {
      setIsProcessing(true);
      setBackendError(null);

      // Test connection first
      const isConnected = await testApiConnection();
      if (!isConnected) {
        setIsProcessing(false);
        return;
      }

      // Capture the current frame from video
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(videoRef.current, 0, 0);
      const imageData = canvas.toDataURL("image/jpeg", 0.8);

      // Send to backend API with better error handling
      try {
        const response = await fetch(
          "http://localhost:5328/api/detect-emotion",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json",
            },
            body: JSON.stringify({ image: imageData }),
          }
        );

        if (!response.ok) {
          if (response.status === 0) {
            throw new Error("Cannot connect to server. Is it running?");
          }
          const errorText = await response.text();
          throw new Error(`Server error (${response.status}): ${errorText}`);
        }

        const result = await response.json();
        if (result.success && result.face_detected) {
          setEmotions(result.emotions);

          // Draw face detection results if needed
          if (canvasRef.current) {
            const ctx = canvasRef.current.getContext("2d");
            ctx?.clearRect(
              0,
              0,
              canvasRef.current.width,
              canvasRef.current.height
            );
            // You could draw the detected face box here if needed
          }
        } else if (!result.face_detected) {
          console.log("No face detected in frame");
        }
      } catch (error: unknown) {
        if (error instanceof Error) {
          throw new Error(`API Error: ${error.message}`);
        }
        throw new Error("API Error: An unknown error occurred");
      }
    } catch (error) {
      console.error("Error detecting emotions:", error);
      setBackendError(
        error instanceof Error
          ? error.message
          : "Failed to process image. Please try again."
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: true,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: 44100,
          channelCount: 2,
          sampleSize: 16
        }
      });

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

      intervalRef.current = setInterval(detectEmotions, 500);
    } catch (error) {
      console.error("Error accessing camera:", error);
      alert(
        "Failed to access camera or microphone. Please make sure you have granted necessary permissions."
      );
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
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

  // Update the testApiConnection function
  const testApiConnection = async () => {
    try {
      // Use the absolute URL with the correct port
      const response = await fetch("http://localhost:5328/api/test", {
        // Add these headers to help with CORS
        headers: {
          Accept: "application/json",
        },
      });
      const data = await response.json();
      console.log("API test response:", data);
      return true;
    } catch (error) {
      console.error("API connection test failed:", error);
      setBackendError(
        "Cannot connect to Python server. Make sure it's running on port 5328."
      );
      return false;
    }
  };

  // Call this in useEffect or before starting the camera
  useEffect(() => {
    testApiConnection();

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Add this function to get the dominant emotion
  const getDominantEmotion = (emotions: Record<string, number>): string => {
    if (Object.keys(emotions).length === 0) return "none";

    let maxEmotion = "";
    let maxValue = 0;

    Object.entries(emotions).forEach(([emotion, value]) => {
      if (value > maxValue) {
        maxValue = value;
        maxEmotion = emotion;
      }
    });

    return maxEmotion;
  };

  const startRecording = async () => {
    if (!streamRef.current) return;
    
    try {
      setRecordingError(null);
      
      // Check supported MIME types for MP4
      const mimeTypes = [
        'video/mp4;codecs=avc1.42E01E,mp4a.40.2',  // H.264 + AAC
        'video/mp4'
      ];
      
      let selectedMimeType = '';
      for (const mimeType of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          console.log('Using MIME type:', mimeType);
          selectedMimeType = mimeType;
          break;
        }
      }
      
      if (!selectedMimeType) {
        throw new Error('No supported MP4 video format found');
      }
      
      // High quality settings for MP4
      const options = {
        mimeType: selectedMimeType,
        videoBitsPerSecond: 8000000,  // 8 Mbps for high quality video
        audioBitsPerSecond: 320000,
        videoKeyFrameInterval: 1000,   // Key frame every second
        videoQuality: 1.0,            // Maximum quality setting
        audioSampleRate: 44100,       // High sample rate
        audioChannelCount: 2          // Stereo audio
      };
      
      const mediaRecorder = new MediaRecorder(streamRef.current, options);
      
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const videoBlob = new Blob(chunksRef.current, { type: 'video/mp4' });
        
        const formData = new FormData();
        formData.append('video', videoBlob, 'recording.mp4');
        
        try {
          setIsUploading(true);
          const response = await fetch('http://localhost:5328/api/upload-video', {
            method: 'POST',
            body: formData,
          });
          
          if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
          }
          
          const result = await response.json();
          console.log('Video uploaded successfully:', result.filename);
          setUploadedVideo(result.filename);
          if (result.has_audio) {
            setUploadedAudio(result.audio_filename);
          }
          router.push(`/recordings/${result.filename}${result.has_audio ? `?audio=${result.audio_filename}` : ''}`);
        } catch (error) {
          console.error('Error uploading video:', error);
          setRecordingError(error instanceof Error ? error.message : 'Failed to upload video');
        } finally {
          setIsUploading(false);
        }
      };
      
      mediaRecorder.start(100);  // Collect data every 100ms for smoother recording
      setIsRecording(true);
      
    } catch (error) {
      console.error('Error starting recording:', error);
      setRecordingError('Failed to start recording. Please check your camera and microphone permissions.');
    }
  };
  
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  return (
    <main className="container py-8 flex flex-col items-center">
      <h1 className="text-4xl font-bold mb-8 text-center">Camera Practice</h1>

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
              muted={isRecording}
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
          >
            {isStreaming ? "Stop Camera" : "Start Camera"}
          </Button>

          {isStreaming && (
            <Button
              onClick={isRecording ? stopRecording : startRecording}
              variant={isRecording ? "destructive" : "default"}
              className="w-32"
              disabled={!isStreaming || isUploading}
            >
              {isRecording ? "Stop Recording" : "Start Recording"}
            </Button>
          )}

          {isUploading && (
            <div className="w-full p-4 bg-primary/10 rounded-lg text-center mt-4 flex items-center justify-center gap-2">
              <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              <span className="text-primary">Uploading video...</span>
            </div>
          )}

          {recordingError && (
            <div className="w-full p-4 bg-destructive/10 text-destructive rounded-lg text-center mt-4">
              {recordingError}
            </div>
          )}

          {isStreaming && Object.keys(emotions).length > 0 && (
            <>
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

              <div className="w-full p-4 bg-primary/10 text-primary rounded-lg text-center mt-4">
                <h3 className="font-semibold mb-1">Dominant Emotion:</h3>
                <div className="text-2xl font-bold capitalize">
                  {getDominantEmotion(emotions)}
                </div>
              </div>
            </>
          )}

          {backendError && (
            <div className="w-full p-4 bg-destructive/10 text-destructive rounded-lg text-center mt-4">
              {backendError}
            </div>
          )}
        </CardContent>
      </Card>
    </main>
  );
}
