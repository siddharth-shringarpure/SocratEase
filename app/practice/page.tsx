"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";

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

interface Emotions {
  neutral: number;
  happy: number;
  sad: number;
  angry: number;
  fearful: number;
  disgusted: number;
  surprised: number;
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
  exit: { opacity: 0, y: -20 },
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const modeCardVariants = {
  initial: { opacity: 0, scale: 0.9 },
  animate: { opacity: 1, scale: 1 },
  hover: {
    scale: 1.02,
    borderColor: "hsl(var(--primary) / 0.5)",
    transition: { duration: 0.2 },
  },
  tap: { scale: 0.98 },
};

const pulseVariants = {
  initial: { scale: 1 },
  animate: {
    scale: [1, 1.1, 1],
    transition: {
      duration: 2,
      repeat: Infinity,
      ease: "easeInOut",
    },
  },
};

// Helper functions for analysis
const getTopEmotions = (
  emotionsData: Array<{ timestamp: number; emotions: Emotions }>
) => {
  // Calculate average emotion values
  const emotionSums: { [key: string]: number } = {};
  const emotionCounts: { [key: string]: number } = {};

  emotionsData.forEach(({ emotions }) => {
    Object.entries(emotions).forEach(([emotion, value]) => {
      emotionSums[emotion] = (emotionSums[emotion] || 0) + value;
      emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
    });
  });

  // Calculate averages and sort
  const averageEmotions = Object.entries(emotionSums).map(([emotion, sum]) => ({
    emotion,
    average: sum / emotionCounts[emotion],
  }));

  return averageEmotions
    .sort((a, b) => b.average - a.average)
    .slice(0, 3)
    .map(({ emotion, average }) => ({
      emotion,
      percentage: (average * 100).toFixed(1),
    }));
};

const getDominantGazeDirection = (
  gazeData: Array<{ timestamp: number; direction: string }>
) => {
  const directionCounts: { [key: string]: number } = {};

  gazeData.forEach(({ direction }) => {
    directionCounts[direction] = (directionCounts[direction] || 0) + 1;
  });

  let dominantDirection = "";
  let maxCount = 0;

  Object.entries(directionCounts).forEach(([direction, count]) => {
    if (count > maxCount) {
      maxCount = count;
      dominantDirection = direction;
    }
  });

  const percentage = ((maxCount / gazeData.length) * 100).toFixed(1);
  return { direction: dominantDirection, percentage };
};

export default function PracticePage() {
  const router = useRouter();
  // Camera-related state
  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);
  const [uploadedAudio, setUploadedAudio] = useState<string | null>(null);
  const [separateAudioRecording, setSeparateAudioRecording] = useState(false);
  const [gazeDirection, setGazeDirection] = useState<string>("center");
  const [emotions, setEmotions] = useState<Emotions>({
    neutral: 0,
    happy: 0,
    sad: 0,
    angry: 0,
    fearful: 0,
    disgusted: 0,
    surprised: 0,
  });
  const [recordingDuration, setRecordingDuration] = useState<number>(0);

  // Recording analysis state
  const [recordingEmotions, setRecordingEmotions] = useState<
    Array<{ timestamp: number; emotions: Emotions }>
  >([]);
  const [recordingGaze, setRecordingGaze] = useState<
    Array<{ timestamp: number; direction: string }>
  >([]);
  const [analysisReady, setAnalysisReady] = useState(false);

  // Add refs for storing recording data
  const emotionsDataRef = useRef<
    Array<{ timestamp: number; emotions: Emotions }>
  >([]);
  const gazeDataRef = useRef<Array<{ timestamp: number; direction: string }>>(
    []
  );

  // Camera-related refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const lowPassNodeRef = useRef<BiquadFilterNode | null>(null);
  const highPassNodeRef = useRef<BiquadFilterNode | null>(null);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);

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
        "Maintain a confident and assertive tone",
      ],
    },
    {
      id: "emotive",
      name: "Emotive",
      description: "Express feelings and emotions clearly",
      emoji: "💝",
      tips: [
        "Use appropriate emotional language",
        "Match your tone to the emotion",
        "Practice empathetic responses",
      ],
    },
    {
      id: "public-speaking",
      name: "Public Speaking",
      description: "Master speaking in front of audiences",
      emoji: "🎤",
      tips: [
        "Project your voice clearly",
        "Use engaging body language",
        "Structure your speech with clear points",
      ],
    },
    {
      id: "rizzing",
      name: "Rizzing",
      description: "Practice charismatic and engaging conversation",
      emoji: "✨",
      tips: [
        "Stay confident and authentic",
        "Use appropriate humor",
        "Read and respond to social cues",
      ],
    },
    {
      id: "basic-conversations",
      name: "Basic Conversations",
      description: "Everyday casual interactions",
      emoji: "💬",
      tips: [
        "Keep the conversation flowing naturally",
        "Ask open-ended questions",
        "Show genuine interest",
      ],
    },
    {
      id: "formal-conversations",
      name: "Formal Conversations",
      description: "Professional and business communication",
      emoji: "👔",
      tips: [
        "Maintain professional language",
        "Be concise and clear",
        "Use appropriate formal expressions",
      ],
    },
    {
      id: "debating",
      name: "Debating",
      description: "Structured argument and discussion",
      emoji: "⚖️",
      tips: [
        "Present logical arguments",
        "Listen actively to counterpoints",
        "Support claims with evidence",
      ],
    },
    {
      id: "storytelling",
      name: "Storytelling",
      description: "Engaging narrative communication",
      emoji: "📚",
      tips: [
        "Set up a clear narrative structure",
        "Use descriptive language",
        "Maintain audience engagement",
      ],
    },
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
    ? conversationModes.find((mode) => mode.id === selectedMode)
    : null;

  // Camera-related functions
  const detectCombined = async () => {
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

      // Send to combined backend API
      const response = await fetch(
        `${
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:5328"
        }/api/detect-combined`,
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
        throw new Error("Failed to process frame");
      }

      const result = await response.json();

      if (result.success && result.face_detected) {
        // Update emotions
        setEmotions(result.emotions);

        // Store emotions and gaze data if recording
        if (isRecording) {
          const timestamp = Date.now();
          const emotionsData = { timestamp, emotions: result.emotions };
          const gazeData = result.gaze
            ? { timestamp, direction: result.gaze.direction }
            : null;

          // Store in refs instead of state
          emotionsDataRef.current.push(emotionsData);
          if (gazeData) {
            gazeDataRef.current.push(gazeData);
          }
        }

        // Update gaze
        if (result.gaze) {
          setGazeDirection(result.gaze.direction);

          // Update canvas with gaze visualization
          const video = videoRef.current;
          const canvas = canvasRef.current;

          // Get the display size
          const displayRect = video.getBoundingClientRect();
          const displayWidth = displayRect.width;
          const displayHeight = displayRect.height;

          // Update canvas size if needed
          if (
            canvas.width !== displayWidth ||
            canvas.height !== displayHeight
          ) {
            canvas.width = displayWidth;
            canvas.height = displayHeight;
          }

          // If recording, clear the canvas and return
          if (isRecording) {
            const overlayCtx = canvas.getContext("2d");
            if (overlayCtx) {
              overlayCtx.clearRect(0, 0, displayWidth, displayHeight);
            }
            return;
          }

          // Calculate scaling factors
          const scaleX = displayWidth / video.videoWidth;
          const scaleY = displayHeight / video.videoHeight;
          const scale = Math.min(scaleX, scaleY);

          // Calculate centering offsets
          const offsetX = (displayWidth - video.videoWidth * scale) / 2;
          const offsetY = (displayHeight - video.videoHeight * scale) / 2;

          // Get overlay context
          const overlayCtx = canvas.getContext("2d", {
            willReadFrequently: true,
          });
          if (!overlayCtx) return;

          // Clear previous drawings
          overlayCtx.clearRect(0, 0, displayWidth, displayHeight);

          // Set up transform for proper scaling and centering
          overlayCtx.save();
          overlayCtx.translate(offsetX, offsetY);
          overlayCtx.scale(scale, scale);

          // Draw minimal face outline
          if (result.gaze.landmarks) {
            overlayCtx.strokeStyle = "rgba(255, 255, 255, 0.3)"; // Very subtle white outline
            overlayCtx.lineWidth = 1;

            // Draw face outline using selected landmarks
            const faceOutlinePoints = result.gaze.landmarks.filter(
              (_: [number, number], index: number) =>
                // Only use points that form the face outline
                [
                  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
                ].includes(index)
            );

            if (faceOutlinePoints.length > 0) {
              overlayCtx.beginPath();
              overlayCtx.moveTo(
                faceOutlinePoints[0][0] * video.videoWidth,
                faceOutlinePoints[0][1] * video.videoHeight
              );

              faceOutlinePoints.forEach((point: [number, number]) => {
                overlayCtx.lineTo(
                  point[0] * video.videoWidth,
                  point[1] * video.videoHeight
                );
              });

              overlayCtx.closePath();
              overlayCtx.stroke();
            }
          }

          // Draw minimal gaze indicator
          if (result.gaze.gaze_arrow) {
            const { start, end } = result.gaze.gaze_arrow;
            overlayCtx.strokeStyle = "rgba(255, 255, 255, 0.4)"; // Subtle white
            overlayCtx.lineWidth = 2;

            // Draw a simple dot at the eye center
            overlayCtx.beginPath();
            overlayCtx.arc(
              start.x * video.videoWidth,
              start.y * video.videoHeight,
              3,
              0,
              2 * Math.PI
            );
            overlayCtx.fill();

            // Draw a small line indicating direction
            overlayCtx.beginPath();
            overlayCtx.moveTo(
              start.x * video.videoWidth,
              start.y * video.videoHeight
            );
            overlayCtx.lineTo(
              end.x * video.videoWidth,
              end.y * video.videoHeight
            );
            overlayCtx.stroke();
          }

          // Draw minimal gaze direction text
          overlayCtx.font = "16px system-ui"; // Smaller, system font
          overlayCtx.textBaseline = "top";
          const text = result.gaze.direction.toUpperCase();

          overlayCtx.fillStyle = "rgba(255, 255, 255, 0.5)";
          overlayCtx.fillText(text, 10, 10);

          // Restore the original transform
          overlayCtx.restore();
        }
      }
    } catch (error) {
      console.error("Error in combined detection:", error);
      setBackendError(
        error instanceof Error
          ? error.message
          : "Failed to process frame. Please try again."
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
          sampleSize: 16,
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // Wait for video to be ready
        await new Promise((resolve) => {
          if (!videoRef.current) return;
          videoRef.current.onloadedmetadata = () => {
            if (videoRef.current) {
              videoRef.current.play();
              resolve(true);
            }
          };
        });

        // Initialize canvas size
        if (canvasRef.current && videoRef.current) {
          const videoRect = videoRef.current.getBoundingClientRect();
          canvasRef.current.width = videoRect.width;
          canvasRef.current.height = videoRect.height;
        }
      }

      streamRef.current = stream;
      setIsStreaming(true);

      // Start detection loop with combined detection - increase frequency during recording
      intervalRef.current = setInterval(detectCombined, 50); // Increased frequency from 100ms to 50ms
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

  const testApiConnection = async () => {
    try {
      const response = await fetch("http://localhost:5328/api/test", {
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

  const startRecording = async () => {
    if (!streamRef.current) return;

    try {
      setRecordingError(null);
      setRecordingDuration(0);
      // Clear the refs instead of state
      emotionsDataRef.current = [];
      gazeDataRef.current = [];
      setAnalysisReady(false);

      // Clear the canvas when starting recording
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext("2d");
        if (ctx) {
          ctx.clearRect(
            0,
            0,
            canvasRef.current.width,
            canvasRef.current.height
          );
        }
      }

      // Ensure detection interval is running at higher frequency during recording
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      intervalRef.current = setInterval(detectCombined, 50); // 50ms interval during recording

      // Check supported MIME types for MP4
      const mimeTypes = [
        "video/mp4;codecs=avc1.42E01E,mp4a.40.2", // H.264 + AAC
        "video/mp4",
      ];

      let selectedMimeType = "";
      for (const mimeType of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          console.log("Using MIME type:", mimeType);
          selectedMimeType = mimeType;
          break;
        }
      }

      if (!selectedMimeType) {
        throw new Error("No supported MP4 video format found");
      }

      // High quality settings for MP4
      const options = {
        mimeType: selectedMimeType,
        videoBitsPerSecond: 8000000,
        audioBitsPerSecond: 320000,
        videoKeyFrameInterval: 1000,
        videoQuality: 1.0,
        audioSampleRate: 44100,
        audioChannelCount: 2,
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
        // Clear recording timer
        if (recordingTimerRef.current) {
          clearInterval(recordingTimerRef.current);
          recordingTimerRef.current = null;
        }

        const videoBlob = new Blob(chunksRef.current, { type: "video/mp4" });

        const formData = new FormData();
        formData.append("video", videoBlob, "recording.mp4");

        try {
          setIsUploading(true);
          const response = await fetch(
            "http://localhost:5328/api/upload-video",
            {
              method: "POST",
              body: formData,
            }
          );

          if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
          }

          const result = await response.json();
          console.log("Video uploaded successfully:", result.filename);
          setUploadedVideo(result.filename);
          if (result.has_audio) {
            setUploadedAudio(result.audio_filename);
          }

          // Process recording data
          console.log("Processing recording data...");
          console.log("Emotions data:", emotionsDataRef.current);
          console.log("Gaze data:", gazeDataRef.current);

          // Don't return early if no emotions data, just log a warning
          if (emotionsDataRef.current.length === 0) {
            console.warn(
              "No emotions data recorded, but continuing with available data"
            );
          }

          // Always update state with whatever data we have
          setRecordingEmotions(emotionsDataRef.current);
          setRecordingGaze(gazeDataRef.current);

          const analysisData = {
            emotions: emotionsDataRef.current,
            gaze: gazeDataRef.current,
            duration: recordingDuration,
            category: selectedMode,
          };

          // Always save whatever data we have
          console.log("Saving analysis data:", analysisData);
          const storageKey = `recording_analysis_${result.filename}`;
          console.log("Storage key:", storageKey);
          localStorage.setItem(storageKey, JSON.stringify(analysisData));
          console.log("Data saved to localStorage");

          // Also save a dedicated metadata object for this recording
          const metadataKey = `recording_metadata_${result.filename}`;
          const metadata = {
            category: selectedMode,
            timestamp: new Date().toISOString(),
            duration: recordingDuration,
          };
          localStorage.setItem(metadataKey, JSON.stringify(metadata));
          console.log("Metadata saved to localStorage");

          setAnalysisReady(true);

          // Always navigate to the recordings page
          const recordingUrl = `/recordings/${result.filename}${
            result.has_audio ? `?audio=${result.audio_filename}` : ""
          }`;
          console.log("Navigating to:", recordingUrl);
          router.push(recordingUrl);
        } catch (error) {
          console.error("Error uploading video:", error);
          setRecordingError(
            error instanceof Error ? error.message : "Failed to upload video"
          );
        } finally {
          setIsUploading(false);
        }
      };

      mediaRecorder.start(100);
      setIsRecording(true);

      // Start recording timer
      recordingTimerRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (error) {
      console.error("Error starting recording:", error);
      setRecordingError(
        "Failed to start recording. Please check your camera and microphone permissions."
      );
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      // First, stop the media recorder
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      // Reset detection interval to normal frequency
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = setInterval(detectCombined, 100);
      }
    }
  };

  // Format recording duration
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  // Add cleanup for recording timer in useEffect
  useEffect(() => {
    testApiConnection();

    return () => {
      // Clean up camera resources
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }

      // Clean up speech recognition
      if (recognition) {
        recognition.stop();
      }
    };
  }, []);

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
                    <CardTitle className="text-center text-2xl">
                      Choose Your Practice Mode
                    </CardTitle>
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
                    {conversationModes.map(
                      (mode: ConversationMode, index: number) => (
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
                          ${
                            selectedMode === mode.id
                              ? "border-primary"
                              : "border-input"
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
                            <div className="font-semibold text-lg mb-2">
                              {mode.name}
                            </div>
                            <div className="text-sm text-muted-foreground">
                              {mode.description}
                            </div>
                          </div>
                        </motion.button>
                      )
                    )}
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
                  <CardTitle className="text-2xl">
                    {selectedModeData.name} Practice
                  </CardTitle>
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
              className="space-y-6"
            >
              <Card className="border-2">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      <motion.span
                        animate={{ scale: isRecording ? [1, 1.2, 1] : 1 }}
                        transition={{
                          duration: 1,
                          repeat: isRecording ? Infinity : 0,
                        }}
                        className="text-primary"
                      >
                        ●
                      </motion.span>
                      {isRecording ? "Recording..." : "Ready"}
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
                    {/* Camera Feed */}
                    <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted={isRecording}
                        className="absolute top-0 left-0 w-full h-full object-contain"
                      />
                      <canvas
                        ref={canvasRef}
                        className="absolute top-0 left-0 w-full h-full"
                        style={{
                          pointerEvents: "none",
                          zIndex: 10,
                          display: isRecording ? "none" : "block",
                        }}
                      />
                      {isRecording && (
                        <div className="absolute top-4 right-4 bg-red-500 text-white px-3 py-1 rounded-full flex items-center gap-2">
                          <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                          <span>{formatDuration(recordingDuration)}</span>
                        </div>
                      )}
                    </div>

                    {/* Camera and Recording Controls */}
                    <div className="flex justify-center gap-4">
                      {!isRecording && (
                        <Button
                          onClick={isStreaming ? stopCamera : startCamera}
                          variant={isStreaming ? "destructive" : "default"}
                          className="w-32"
                        >
                          {isStreaming ? "Stop Camera" : "Start Camera"}
                        </Button>
                      )}

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
                    </div>

                    {/* Mode-specific tips */}
                    {selectedModeData && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="space-y-2 bg-muted/30 p-4 rounded-lg"
                      >
                        <h3 className="font-semibold mb-2">Practice Tips:</h3>
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

              {/* Upload Overlay */}
              {isUploading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
                >
                  <div className="bg-background p-8 rounded-lg shadow-lg flex flex-col items-center gap-4">
                    <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
                    <p className="text-lg font-semibold">
                      Processing your recording...
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Please wait while we prepare your video
                    </p>
                  </div>
                </motion.div>
              )}

              {/* Error Messages */}
              {(recordingError || backendError) && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="w-full p-4 bg-destructive/10 text-destructive rounded-lg text-center"
                >
                  {recordingError || backendError}
                </motion.div>
              )}
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
        </div>
      </AnimatePresence>
    </motion.main>
  );
}
