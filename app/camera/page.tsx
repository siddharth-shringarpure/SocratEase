"use client";

/**
 * @fileoverview Camera practice page component for emotion and gaze detection.
 * Implements real-time facial analysis using a webcam feed and backend API.
 */

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BackendStatus } from "@/components/custom/BackendStatus";
import { motion, AnimatePresence } from "framer-motion";

// TODO: Consider moving these types to a separate types.ts file
type Emotions = {
  neutral: number;
  happy: number;
  sad: number;
  angry: number;
  fearful: number;
  disgusted: number;
  surprised: number;
};

/**
 * Main camera page component for emotion and gaze detection
 * @returns {JSX.Element} The rendered camera page
 */
export default function CameraPage() {
  // State management for camera and detection features
  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
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

  // refs for managing video, canvas, and intervals
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const [isCheckingBackend, setIsCheckingBackend] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const detectionErrorShownRef = useRef(false);
  const backendCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);

  /**
   * Checks if the backend service is available and responding
   */
  const checkBackend = async () => {
    if (isCheckingBackend) return;
    setIsCheckingBackend(true);

    try {
      const response = await fetch("/api/test");
      const data = await response.json();
      setIsBackendConnected(response.ok);

      if (!response.ok) {
        setBackendError("Backend service is not responding");
      } else {
        setBackendError(null);
      }
    } catch (error) {
      setIsBackendConnected(false);
      setBackendError("Can't connect to backend service");
    } finally {
      setIsCheckingBackend(false);
    }
  };

  /**
   * Handles combined detection of emotions and gaze
   * @returns {Promise<void>}
   */
  const detectCombined = async () => {
    // Early return if conditions aren't met
    if (
      !videoRef.current ||
      !canvasRef.current ||
      isProcessing ||
      !isBackendConnected
    )
      return;

    // Make sure video is properly loaded
    if (
      !videoRef.current.videoWidth ||
      !videoRef.current.videoHeight ||
      !videoRef.current.getBoundingClientRect().width
    ) {
      return;
    }

    try {
      setIsProcessing(true);

      // Capture current frame
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(videoRef.current, 0, 0);
      const imageData = canvas.toDataURL("image/jpeg", 0.95);

      // TODO: Implement better error handling for image capture failures

      // Send to backend with retries
      let retries = 0;
      const MAX_RETRIES = 2;
      let response = null;

      while (retries <= MAX_RETRIES) {
        try {
          response = await fetch(`/api/detect-combined`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json",
            },
            body: JSON.stringify({ image: imageData }),
          });

          if (response.ok) break;

          // retry on server errors
          if (response.status === 500 && retries < MAX_RETRIES) {
            console.warn(
              `Detection try ${retries + 1} failed, having another go...`
            ); // Casual language
            retries++;
            await new Promise((resolve) => setTimeout(resolve, 100));
            continue;
          }

          break;
        } catch (fetchError) {
          console.error(`Fetch error on try ${retries + 1}:`, fetchError);
          if (retries < MAX_RETRIES) {
            retries++;
            await new Promise((resolve) => setTimeout(resolve, 100));
            continue;
          }
          throw fetchError;
        }
      }

      if (!response || !response.ok) {
        if (!detectionErrorShownRef.current) {
          console.error("Detection API error:", response?.status);
          detectionErrorShownRef.current = true;
        }
        setIsBackendConnected(false);
        return;
      }

      detectionErrorShownRef.current = false;
      setIsBackendConnected(true);
      setBackendError(null);

      const result = await response.json();

      // Process detection results
      if (result.success && result.face_detected) {
        setEmotions(result.emotions);

        if (result.gaze) {
          setGazeDirection(result.gaze.direction);

          // Draw gaze visualisation
          const video = videoRef.current;
          const canvas = canvasRef.current;

          const displayRect = video.getBoundingClientRect();
          const displayWidth = displayRect.width;
          const displayHeight = displayRect.height;

          if (
            canvas.width !== displayWidth ||
            canvas.height !== displayHeight
          ) {
            canvas.width = displayWidth;
            canvas.height = displayHeight;
          }

          // Calculate scaling and positioning
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

          // Draw face outline
          if (result.gaze.landmarks) {
            overlayCtx.strokeStyle = "rgba(255, 255, 255, 0.3)";
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

          // Draw gaze indicator
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

          // Show gaze direction text
          overlayCtx.font = "16px system-ui";
          overlayCtx.textBaseline = "top";
          const text = result.gaze.direction.toUpperCase();
          overlayCtx.fillStyle = "rgba(255, 255, 255, 0.5)";
          overlayCtx.fillText(text, 10, 10);

          overlayCtx.restore();
        }
      }
    } catch (error) {
      if (!detectionErrorShownRef.current) {
        console.error("Something went wrong with detection:", error);
        detectionErrorShownRef.current = true;
      }
      setIsBackendConnected(false);
    } finally {
      setIsProcessing(false);
    }
  };

  /**
   * Starts the camera and initialises detection
   */
  const startCamera = async () => {
    try {
      setIsCheckingBackend(true);
      setBackendError(null);

      const response = await fetch("/api/test");
      if (!response.ok) {
        setBackendError(
          "Backend server is not responding. Please ensure the Python server is running."
        );
        setIsBackendConnected(false);
        setIsCheckingBackend(false);
        return;
      }

      setIsBackendConnected(true);

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
              videoRef.current.play().catch(console.error);
              resolve(true);
            }
          };
        });

        if (canvasRef.current && videoRef.current) {
          const videoRect = videoRef.current.getBoundingClientRect();
          canvasRef.current.width = videoRect.width;
          canvasRef.current.height = videoRect.height;
        }
      }

      streamRef.current = stream;
      setIsStreaming(true);

      intervalRef.current = setInterval(detectCombined, 100);
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

  /**
   * Stops the camera and cleans up resources
   */
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      streamRef.current = null;
    }

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Reset states
    setIsStreaming(false);
    setIsBackendConnected(false);
    setBackendError(null);
    setIsCheckingBackend(false);
    setEmotions({
      neutral: 0,
      happy: 0,
      sad: 0,
      angry: 0,
      fearful: 0,
      disgusted: 0,
      surprised: 0,
    });
    setGazeDirection("center");

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  };

  // Initial backend check
  useEffect(() => {
    checkBackend();

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  /**
   * Gets the dominant emotion from the emotions object
   */
  const getDominantEmotion = (
    emotions: Record<string, number> | null | undefined
  ): string => {
    if (!emotions || Object.keys(emotions).length === 0) return "none";

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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (backendCheckIntervalRef.current) {
        clearInterval(backendCheckIntervalRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <main className="container mx-auto flex flex-col items-center p-4 py-8 mt-16">
      <motion.h1
        className="text-4xl font-bold mb-8 text-center"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        Camera Practice
      </motion.h1>

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-2xl"
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-center">Camera Feed</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-center gap-4">
            <motion.div
              className="relative w-full aspect-video bg-black rounded-lg overflow-hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="absolute top-0 left-0 w-full h-full object-contain"
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full"
                style={{
                  pointerEvents: "none",
                  zIndex: 10,
                }}
              />
            </motion.div>

            <motion.div
              className="flex gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Button
                onClick={isStreaming ? stopCamera : startCamera}
                variant={isStreaming ? "destructive" : "default"}
                className="w-32"
                disabled={isCheckingBackend}
              >
                {isCheckingBackend ? (
                  <span className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                    Checking...
                  </span>
                ) : isStreaming ? (
                  "Stop Camera"
                ) : (
                  "Start Camera"
                )}
              </Button>
            </motion.div>

            <AnimatePresence>
              <motion.div
                className="w-full flex justify-center"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                {isCheckingBackend ? (
                  <div className="text-center text-sm text-muted-foreground">
                    Checking backend connection...
                  </div>
                ) : isStreaming ? (
                  <>
                    <BackendStatus />
                  </>
                ) : null}
              </motion.div>
            </AnimatePresence>

            <AnimatePresence>
              {backendError && (
                <motion.div
                  className="w-full p-4 bg-destructive/10 text-destructive rounded-lg text-center"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                >
                  <p>{backendError}</p>
                  <button
                    onClick={startCamera}
                    className="mt-2 text-sm text-primary hover:text-primary/80 underline underline-offset-2"
                  >
                    Try Again
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {isStreaming && emotions && Object.keys(emotions).length > 0 && (
                <>
                  <motion.div
                    className="w-full p-4 bg-muted rounded-lg"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <h3 className="font-semibold mb-2">Detected Emotions:</h3>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(emotions).map(
                        ([emotion, probability], index) => (
                          <motion.div
                            key={emotion}
                            className="flex justify-between"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.05 }}
                          >
                            <span className="capitalize">{emotion}:</span>
                            <span>{(probability * 100).toFixed(1)}%</span>
                          </motion.div>
                        )
                      )}
                    </div>
                  </motion.div>

                  <motion.div
                    className="w-full p-4 bg-primary/10 text-primary rounded-lg text-center"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ duration: 0.3, delay: 0.2 }}
                  >
                    <h3 className="font-semibold mb-1">Dominant Emotion:</h3>
                    <motion.div
                      className="text-2xl font-bold capitalize"
                      key={getDominantEmotion(emotions)}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.2 }}
                    >
                      {getDominantEmotion(emotions)}
                    </motion.div>
                  </motion.div>
                </>
              )}
            </AnimatePresence>
          </CardContent>
        </Card>
      </motion.div>
    </main>
  );
}
