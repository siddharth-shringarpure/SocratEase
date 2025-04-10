"use client";

/**
 * @fileoverview Practice page component for SocratEase - handles video recording and analysis
 * for various conversation modes. Integrates with Python backend for real-time processing.
 */

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import { ModeSelection } from "@/components/custom/practice/ModeSelection";
import { TipsAndStart } from "@/components/custom/practice/TipsAndStart";
import { PracticeSession } from "@/components/custom/practice/PracticeSession";
import { ConversationMode, Emotions } from "@/types/practice";
import { checkBackendStatus } from "@/app/actions/backend";
import {
  getOrCreateDeviceId,
  getMetadataKey,
  getAnalysisKey,
  getDeviceIdHashDigest,
} from "@/lib/deviceId";

const CONVERSATION_MODES: ConversationMode[] = [
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
  // TODO: Consider adding more specialised modes based on user feedback
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
      "Use appropriate humour",
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

/**
 * Main practice page component handling video recording and analysis
 * @returns {JSX.Element} The practice page component
 */
export default function PracticePage(): JSX.Element {
  const router = useRouter();

  // Camera and recording states
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [isCheckingBackend, setIsCheckingBackend] = useState<boolean>(false);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);
  const [uploadedAudio, setUploadedAudio] = useState<string | null>(null);
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

  // Refs for tracking recording data
  const durationRef = useRef<number>(0);
  const emotionsDataRef = useRef<
    Array<{ timestamp: number; emotions: Emotions }>
  >([]);
  const gazeDataRef = useRef<Array<{ timestamp: number; direction: string }>>(
    []
  );

  // Analysis states
  const [recordingEmotions, setRecordingEmotions] = useState<
    Array<{ timestamp: number; emotions: Emotions }>
  >([]);
  const [recordingGaze, setRecordingGaze] = useState<
    Array<{ timestamp: number; direction: string }>
  >([]);
  const [analysisReady, setAnalysisReady] = useState<boolean>(false);

  // Media handling refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);
  const backendCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Practice mode states
  const [selectedMode, setSelectedMode] = useState<string | null>(null);
  const [showPractice, setShowPractice] = useState<boolean>(false);

  const selectedModeData = selectedMode
    ? CONVERSATION_MODES.find((mode) => mode.id === selectedMode)
    : null;

  // Helper to track backend connection changes
  const setBackendConnectionState = (
    isConnected: boolean,
    source: string
  ): void => {
    console.log("[Backend Connection] State change:", {
      from: isBackendConnected,
      to: isConnected,
      source,
      timestamp: new Date().toISOString(),
    });
    setIsBackendConnected(isConnected);
  };

  // TODO: Consider splitting camera and analysis logic into separate hooks
  const detectCombined = async (): Promise<void> => {
    if (
      !videoRef.current ||
      !canvasRef.current ||
      isProcessing ||
      !isBackendConnected
    )
      return;

    try {
      setIsProcessing(true);

      // capture current frame
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(videoRef.current, 0, 0);
      const imageData = canvas.toDataURL("image/jpeg", 0.95);

      // Send to backend with retries
      let retries = 0;
      const MAX_RETRIES = 2;
      let response = null;

      while (retries <= MAX_RETRIES) {
        try {
          response = await fetch("http://localhost:5000/api/detect-combined", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json",
            },
            body: JSON.stringify({ image: imageData }),
          });

          if (response.ok) break;

          if (response.status === 500 && retries < MAX_RETRIES) {
            console.warn(
              `Detection attempt ${retries + 1} failed, retrying...`
            );
            retries++;
            await new Promise((resolve) => setTimeout(resolve, 100));
            continue;
          }

          break;
        } catch (fetchError) {
          console.error(`Fetch error on attempt ${retries + 1}:`, fetchError);
          if (retries < MAX_RETRIES) {
            retries++;
            await new Promise((resolve) => setTimeout(resolve, 100));
            continue;
          }
          throw fetchError;
        }
      }

      if (!response?.ok) {
        if (response?.status === 500) {
          if (!backendError) {
            setBackendError(
              "Backend server is not responding. Please ensure the Python server is running."
            );
            if (intervalRef.current) {
              clearInterval(intervalRef.current);
              intervalRef.current = null;
            }
          }
          return;
        }
        throw new Error("Failed to process frame");
      }

      // Clear previous backend error if request succeeds
      if (backendError) {
        setBackendError(null);
        if (!intervalRef.current) {
          intervalRef.current = setInterval(
            detectCombined,
            isRecording ? 50 : 100
          );
        }
      }

      const result = await response.json();

      // Store detection results if face detected
      if (result.success && result.face_detected) {
        setEmotions(result.emotions);

        if (isRecording) {
          const timestamp = Date.now();
          const emotionsData = { timestamp, emotions: result.emotions };
          const gazeData = result.gaze
            ? { timestamp, direction: result.gaze.direction }
            : null;

          emotionsDataRef.current.push(emotionsData);
          if (gazeData) {
            gazeDataRef.current.push(gazeData);
          }
        }

        if (result.gaze) {
          setGazeDirection(result.gaze.direction);
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

  // TODO: Add error handling for failed backend connections
  const testApiConnection = async (): Promise<boolean> => {
    try {
      console.log("[Initial Backend Test] Starting...");
      const status = await checkBackendStatus();
      console.log("[Initial Backend Test] Response:", status);

      setBackendConnectionState(status.isConnected, "initial_test");
      if (!status.isConnected) {
        setBackendError(status.error || "Backend service is not responding");
      }
      return status.isConnected;
    } catch (error) {
      console.error("[Initial Backend Test] Error:", {
        error: error instanceof Error ? error.message : "Unknown error",
        isStreaming,
        isBackendConnected: false,
      });
      setBackendError(
        "Cannot connect to Python server. Make sure it's running."
      );
      setBackendConnectionState(false, "initial_test_error");
      return false;
    }
  };

  const startCamera = async (): Promise<void> => {
    console.log("[Camera Start] Beginning camera initialisation:", {
      isStreaming,
      isBackendConnected,
      isCheckingBackend,
    });

    try {
      // Get camera permissions and initialise
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
        await new Promise((resolve) => {
          if (!videoRef.current) return;
          videoRef.current.onloadedmetadata = () => {
            if (videoRef.current) {
              videoRef.current.play();
              resolve(true);
            }
          };
        });

        // Set canvas size
        if (canvasRef.current && videoRef.current) {
          const videoRect = videoRef.current.getBoundingClientRect();
          canvasRef.current.width = videoRect.width;
          canvasRef.current.height = videoRect.height;
        }
      }

      streamRef.current = stream;
      setIsStreaming(true);

      // Check backend connection
      setIsCheckingBackend(true);
      setBackendError(null);

      try {
        const status = await checkBackendStatus();
        if (!status.isConnected) {
          setBackendError(
            status.error ||
              "Backend server is not responding. Please ensure the Python server is running."
          );
          setBackendConnectionState(false, "camera_start");
        } else {
          setBackendConnectionState(true, "camera_start");
          intervalRef.current = setInterval(detectCombined, 50);

          // Start periodic backend checks
          const checkBackendInterval = setInterval(async () => {
            try {
              const status = await checkBackendStatus();
              setBackendConnectionState(status.isConnected, "periodic_check");
              if (!status.isConnected) {
                if (!backendError) {
                  setBackendError(
                    status.error ||
                      "Backend connection lost. Please ensure the Python server is running."
                  );
                }
              } else {
                if (backendError) {
                  setBackendError(null);
                }
              }
            } catch (error) {
              setBackendConnectionState(false, "periodic_check_error");
              if (!backendError) {
                setBackendError(
                  "Backend connection lost. Please ensure the Python server is running."
                );
              }
            }
          }, 3000);

          backendCheckIntervalRef.current = checkBackendInterval;
        }
      } catch (error) {
        setBackendError(
          "Cannot connect to backend server. Please ensure it's running."
        );
        setBackendConnectionState(false, "camera_start_error");
      }
    } catch (error) {
      console.error("Error starting camera:", error);
      if (error instanceof Error && error.name === "NotAllowedError") {
        setBackendError(
          "Camera access denied. Please allow camera access and try again."
        );
      } else {
        setBackendError(
          "Failed to start camera. Please check your camera permissions."
        );
      }
      setIsStreaming(false);
      setIsBackendConnected(false);
    } finally {
      setIsCheckingBackend(false);
    }
  };

  const stopCamera = (): void => {
    // Stop media tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      streamRef.current = null;
    }

    // Clear intervals
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (backendCheckIntervalRef.current) {
      clearInterval(backendCheckIntervalRef.current);
      backendCheckIntervalRef.current = null;
    }

    // Reset states
    setIsStreaming(false);
    setIsBackendConnected(false);
    setBackendError(null);
    setIsCheckingBackend(false);

    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  };

  const startRecording = async (): Promise<void> => {
    if (!streamRef.current || !isBackendConnected) {
      console.log("[Recording] Cannot start:", {
        hasStream: !!streamRef.current,
        isBackendConnected,
        isStreaming,
      });
      return;
    }

    try {
      setRecordingError(null);
      setRecordingDuration(0);
      emotionsDataRef.current = [];
      gazeDataRef.current = [];
      setAnalysisReady(false);

      // Clear canvas
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

      // Check supported MIME types
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

      // High quality settings
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

      mediaRecorder.onstart = () => {
        setIsRecording(true);
        durationRef.current = 0;
        setRecordingDuration(0);
        recordingTimerRef.current = setInterval(() => {
          durationRef.current += 1;
          setRecordingDuration(durationRef.current);
        }, 1000);
      };

      mediaRecorder.onstop = async () => {
        if (recordingTimerRef.current) {
          clearInterval(recordingTimerRef.current);
          recordingTimerRef.current = null;
        }

        const finalDuration = durationRef.current;

        // Create form data
        const blob = new Blob(chunksRef.current, { type: "video/mp4" });
        const formData = new FormData();

        // Get device ID for filename
        const deviceIdPrefix = getOrCreateDeviceId();
        if (!deviceIdPrefix) {
          setRecordingError(
            "Could not generate device ID for recording. Try refreshing the page, or contact the developers if this issue persists."
          );
          setIsUploading(false);
          return;
        }

        // Generate filename with timestamp
        const now = new Date();
        const formattedDate = now.toISOString().slice(0, 10).replace(/-/g, "");
        const formattedTime = now.toISOString().slice(11, 19).replace(/:/g, "");
        const timestamp = `${formattedDate}T${formattedTime}`;
        const deviceIdHash = getDeviceIdHashDigest();

        const videoFilename = `${deviceIdHash}_${timestamp}.mp4`;
        const audioFilename = `${deviceIdHash}_${timestamp}_audio.wav`;

        formData.append("video", blob, videoFilename);
        formData.append("audio", blob, audioFilename);
        formData.append("deviceId", deviceIdPrefix);

        try {
          setIsUploading(true);

          const response = await fetch("http://localhost:5000/api/recordings", {
            method: "POST",
            headers: {
              Accept: "application/json",
            },
            body: formData,
          });

          if (!response.ok) {
            let errorDetails = "";
            try {
              const errorData = await response.json();
              errorDetails = JSON.stringify(errorData);
            } catch (e) {
              console.error("[Upload] Could not parse error response as JSON");
            }

            throw new Error(
              `Upload failed: ${response.statusText}${
                errorDetails ? ` - ${errorDetails}` : ""
              }`
            );
          }

          const result = await response.json();
          setUploadedVideo(result.filename);
          if (result.has_audio) {
            setUploadedAudio(result.audio_filename);
          }

          // Ensure minimum data points exist
          if (emotionsDataRef.current.length === 0) {
            console.warn("No emotions data recorded, adding dummy data");
            // Add a minimal dummy data point
            emotionsDataRef.current.push({
              timestamp: Date.now(),
              emotions: {
                neutral: 0.7,
                happy: 0.2,
                sad: 0.03,
                angry: 0.02,
                fearful: 0.02,
                disgusted: 0.02,
                surprised: 0.01,
              },
            });
          }

          if (gazeDataRef.current.length === 0) {
            console.warn("No gaze data recorded, adding dummy data");
            // Add a minimal dummy data point
            gazeDataRef.current.push({
              timestamp: Date.now(),
              direction: "center",
            });
          }

          setRecordingEmotions(emotionsDataRef.current);
          setRecordingGaze(gazeDataRef.current);

          // Save metadata and analysis
          const metadataKey = getMetadataKey(result.filename);
          const metadata = {
            category: selectedMode,
            timestamp: new Date().toISOString(),
            duration: finalDuration,
          };
          localStorage.setItem(metadataKey, JSON.stringify(metadata));

          const analysisKey = getAnalysisKey(result.filename);
          const analysisData = {
            category: selectedMode,
            timestamp: new Date().toISOString(),
            duration: finalDuration,
            emotions: emotionsDataRef.current,
            gaze: gazeDataRef.current,
          };
          localStorage.setItem(analysisKey, JSON.stringify(analysisData));

          setAnalysisReady(true);

          // Navigate to recording page
          await new Promise((resolve) => setTimeout(resolve, 100));
          const recordingUrl = `/recordings/${result.filename.replace(
            ".mp4",
            ""
          )}`;
          router.replace(recordingUrl); // Use replace instead of push to prevent back navigation issues
        } catch (error) {
          console.error("Error uploading video:", error);
          setRecordingError(
            error instanceof Error ? error.message : "Failed to upload video"
          );
        } finally {
          setIsUploading(false);
        }
      };

      // Update detection interval for recording
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      intervalRef.current = setInterval(detectCombined, 50);

      mediaRecorder.start(100);
    } catch (error) {
      console.error("Error starting recording:", error);
      setRecordingError(
        "Failed to start recording. Please check your camera and microphone permissions."
      );
      setIsRecording(false);
    }
  };

  const stopRecording = (): void => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = setInterval(detectCombined, 100);
      }

      // Delayed camera shutdown
      setTimeout(() => {
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
          if (videoRef.current) {
            videoRef.current.srcObject = null;
          }
          streamRef.current = null;
          setIsStreaming(false);
        }
      }, 2000);
    }
  };

  useEffect(() => {
    testApiConnection();

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (streamRef.current)
        streamRef.current.getTracks().forEach((track) => track.stop());
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      if (backendCheckIntervalRef.current)
        clearInterval(backendCheckIntervalRef.current);
    };
  }, []);

  return (
    <motion.main
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="container max-w-4xl mx-auto p-4 py-8 mt-16"
    >
      <AnimatePresence mode="wait">
        <div className="space-y-8">
          {/* Mode Selection */}
          {!selectedMode && (
            <ModeSelection
              conversationModes={CONVERSATION_MODES}
              selectedMode={selectedMode}
              onModeSelect={setSelectedMode}
            />
          )}

          {/* Tips and Start Page */}
          {selectedMode && !showPractice && selectedModeData && (
            <TipsAndStart
              selectedModeData={selectedModeData}
              onStartPractice={() => setShowPractice(true)}
            />
          )}

          {/* Practice Session */}
          {showPractice && selectedModeData && (
            <PracticeSession
              selectedModeData={selectedModeData}
              isStreaming={isStreaming}
              isRecording={isRecording}
              isCheckingBackend={isCheckingBackend}
              isUploading={isUploading}
              isBackendConnected={isBackendConnected}
              backendError={backendError}
              recordingError={recordingError}
              recordingDuration={recordingDuration}
              videoRef={videoRef}
              canvasRef={canvasRef}
              onStartCamera={startCamera}
              onStopCamera={stopCamera}
              onStartRecording={startRecording}
              onStopRecording={stopRecording}
            />
          )}
        </div>
      </AnimatePresence>
    </motion.main>
  );
}
