"use client";

import { useEffect, useState, useRef, useMemo } from "react";
import {
  useParams,
  useRouter,
  useSearchParams,
  notFound,
} from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { RadialBarChart, RadialBar, PolarRadiusAxis, Label } from "recharts";
import type { Props as LabelProps } from "recharts/types/component/Label";
import { ChevronDown, ChevronUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import {
  getMetadataKey,
  getAnalysisKey,
  getAudioFeedbackKey,
  validateDeviceIdHash,
  getDeviceIdHashDigest,
} from "@/lib/deviceId";
import { AccessDenied } from "@/components/custom/AccessDenied";
import { ServiceError } from "@/components/custom/ServiceError";

interface TTRAnalysis {
  ttr: number;
  unique_words: number;
  diversity_level: string;
  emoji: string;
}

interface LogicalFlowAnalysis {
  score: number;
  emoji: string;
}

interface EmotionAnalysis {
  emotion: string;
  percentage: string;
}

interface GazeAnalysis {
  direction: string;
  percentage: string;
}

interface RecordingAnalysis {
  emotions: EmotionAnalysis[];
  gaze: GazeAnalysis[];
  duration: number;
}

interface AnalysisData {
  total_words: number;
  filler_count: number;
  filler_percentage: number;
  found_fillers: string[];
  filler_emoji: string;
  ttr_analysis: {
    ttr: number;
    unique_words: number;
    diversity_level: string;
    emoji: string;
  };
  logical_flow: {
    score: number;
    emoji: string;
  };
  feedback_text?: string;
}

interface Analysis {
  total_words: number;
  filler_count: number;
  filler_percentage: number;
  found_fillers: string[];
  filler_emoji: string;
  ttr_analysis: TTRAnalysis;
  logical_flow: LogicalFlowAnalysis;
  feedback_text?: string;
}

interface FeedbackStep {
  id: string;
  title: string;
  content: string;
  emoji: string;
  score?: number;
}

function SpeechFeedback({
  analysis,
  recordingAnalysis,
  audioUrl,
  onAudioError,
}: {
  analysis: Analysis | null;
  recordingAnalysis: RecordingAnalysis | null;
  audioUrl: string | undefined;
  onAudioError?: () => void;
}) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [practiceCategory, setPracticeCategory] = useState<string | null>(null);
  const [audioLoaded, setAudioLoaded] = useState(false);

  useEffect(() => {
    // Check if the audio URL is a Blob URL (which means it's an enhanced audio)
    if (audioUrl && audioUrl.startsWith("blob:")) {
      // Extract practice category from localStorage based on filename
      const urlParts = window.location.pathname.split("/");
      const filename = urlParts[urlParts.length - 1];

      try {
        // First try metadata
        const metadataKey = getMetadataKey(filename);
        const metadata = localStorage.getItem(metadataKey);
        if (metadata) {
          const parsedMetadata = JSON.parse(metadata);
          if (parsedMetadata.category) {
            setPracticeCategory(parsedMetadata.category);
          }
        } else {
          // Try analysis data
          const analysisKey = getAnalysisKey(filename);
          const analysisData = localStorage.getItem(analysisKey);
          if (analysisData) {
            const parsedData = JSON.parse(analysisData);
            if (parsedData.category) {
              setPracticeCategory(parsedData.category);
            }
          }
        }
      } catch (e) {
        console.warn("Error extracting practice category:", e);
      }
    }
  }, [audioUrl]);

  const getFeedbackSteps = (): FeedbackStep[] => {
    if (!analysis) return [];

    // Create basic steps that don't require recordingAnalysis
    const basicSteps = [
      {
        id: "stats",
        title: "Speech Statistics",
        content: `You used ${analysis.total_words} total words, with ${
          analysis.ttr_analysis.unique_words
        } being unique. Your logical flow score is ${Math.round(
          analysis.logical_flow.score * 100
        )}%.`,
        emoji: "📊",
      },
    ];

    // Only add emotion and gaze steps if recordingAnalysis is available
    if (recordingAnalysis) {
      return [
        ...basicSteps,
        {
          id: "emotions",
          title: "Emotional Tone",
          content: `Your speech mostly sounded ${
            recordingAnalysis.emotions[0]?.emotion.toLowerCase() || "neutral"
          }, with some moments of ${
            recordingAnalysis.emotions[1]?.emotion.toLowerCase() || "variation"
          }.`,
          emoji: "🎭",
        },
        {
          id: "gaze",
          title: "Eye Direction",
          content: `You were mostly looking ${
            recordingAnalysis.gaze[0]?.direction.toLowerCase() || "forward"
          } during your speech.`,
          emoji: "👀",
        },
      ];
    }

    return basicSteps;
  };

  const steps = getFeedbackSteps();

  const startPlayback = async () => {
    if (!audioRef.current) return;

    try {
      setAudioError(null);
      setIsPlaying(true);
      await audioRef.current.play();

      // Progress through steps every 4 seconds
      const interval = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= steps.length - 1) {
            clearInterval(interval);
            return prev;
          }
          return prev + 1;
        });
      }, 4000);

      audioRef.current.onended = () => {
        setIsPlaying(false);
        clearInterval(interval);
        setCurrentStep(steps.length - 1);
      };
    } catch (error) {
      console.error("Error playing audio:", error);
      setAudioError(
        "Sorry, we couldn't play your speech feedback audio. Please contact the developers if this persists."
      );
      setIsPlaying(false);
    }
  };

  // Calculate a score based on analysis data for visual display
  const getOverallScore = (): number => {
    if (!analysis) return 70; // Default score

    // Calculate score based on filler words, vocabulary diversity, and logical flow
    const fillerScore = 100 - Math.min(100, analysis.filler_percentage * 4);

    // Convert string diversity level to score
    const diversityScore =
      analysis.ttr_analysis.diversity_level === "very high"
        ? 95
        : analysis.ttr_analysis.diversity_level === "high"
        ? 85
        : analysis.ttr_analysis.diversity_level === "average"
        ? 70
        : analysis.ttr_analysis.diversity_level === "low"
        ? 50
        : 30;

    // Logical flow is already a percentage
    const flowScore = analysis.logical_flow.score * 100; // Convert from 0-1 to 0-100

    // Weighted average
    return Math.round(
      fillerScore * 0.4 + diversityScore * 0.3 + flowScore * 0.3
    );
  };

  const overallScore = getOverallScore();

  return (
    <Card className="w-full max-w-2xl border-2">
      <CardHeader className="pb-6">
        <div className="flex flex-col items-center space-y-4">
          <CardTitle className="flex items-center gap-2 text-2xl">
            <span className="font-bold">
              AI-Enhanced Speech Analysis Feedback
            </span>
            {practiceCategory && (
              <Badge className="ml-2" variant="outline">
                {practiceCategory.charAt(0).toUpperCase() +
                  practiceCategory.slice(1).replace("-", " ")}
              </Badge>
            )}
          </CardTitle>

          {/* Score display */}
          <div className="mt-6 flex items-center justify-center">
            <div className="relative w-28 h-28">
              <svg className="w-28 h-28" viewBox="0 0 100 100">
                <circle
                  className="text-muted stroke-current"
                  strokeWidth="8"
                  cx="50"
                  cy="50"
                  r="40"
                  fill="transparent"
                />
                <circle
                  className="text-primary stroke-current"
                  strokeWidth="8"
                  strokeLinecap="round"
                  cx="50"
                  cy="50"
                  r="40"
                  fill="transparent"
                  strokeDasharray={`${2 * Math.PI * 40}`}
                  strokeDashoffset={`${
                    2 * Math.PI * 40 * (1 - overallScore / 100)
                  }`}
                  transform="rotate(-90 50 50)"
                />
                <text
                  x="50"
                  y="50"
                  fontFamily="sans-serif"
                  fontSize="22"
                  textAnchor="middle"
                  dy="7"
                  fill="currentColor"
                >
                  {overallScore}
                </text>
              </svg>
            </div>
            <div className="ml-6">
              <h3 className="font-semibold text-lg">Overall Score</h3>
              <p className="text-sm text-muted-foreground">
                {overallScore >= 90
                  ? "Excellent!"
                  : overallScore >= 80
                  ? "Great job!"
                  : overallScore >= 70
                  ? "Good work!"
                  : overallScore >= 60
                  ? "Room for improvement"
                  : "Keep practicing"}
              </p>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-8 pt-0">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentStep((prev) => Math.max(0, prev - 1))}
              disabled={currentStep === 0 || !steps.length}
              className="h-8 w-8 p-0 rounded-full"
            >
              <ChevronUp className="h-4 w-4" />
              <span className="sr-only">Previous</span>
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                setCurrentStep((prev) => Math.min(steps.length - 1, prev + 1))
              }
              disabled={currentStep === steps.length - 1 || !steps.length}
              className="h-8 w-8 p-0 rounded-full"
            >
              <ChevronDown className="h-4 w-4" />
              <span className="sr-only">Next</span>
            </Button>
          </div>

          <Button
            onClick={
              isPlaying
                ? () => {
                    audioRef.current?.pause();
                    setIsPlaying(false);
                  }
                : startPlayback
            }
            disabled={!steps.length}
            variant="default"
          >
            <span className="flex items-center gap-2">
              {isPlaying ? (
                <>
                  <span className="h-2 w-2 rounded-full bg-current animate-pulse" />
                  Pause Audio
                </>
              ) : (
                <>
                  <span className="h-0 w-0 border-y-4 border-y-transparent border-l-8 border-l-current" />
                  Play Audio Feedback
                </>
              )}
            </span>
          </Button>
        </div>

        <audio
          ref={audioRef}
          src={
            audioUrl?.startsWith("blob:")
              ? audioUrl
              : audioUrl
              ? getAudioSource(audioUrl)
              : undefined
          }
          onEnded={() => setIsPlaying(false)}
          onLoadedData={() => setAudioLoaded(true)}
          onError={(e) => {
            console.error("Audio error:", e);
            setIsPlaying(false);
            setAudioError(
              "Sorry, we couldn't play your speech feedback audio. Please contact the developers if this persists."
            );
            // Notify parent component of the error
            if (onAudioError) onAudioError();
          }}
        />

        {audioError && (
          <div className="mb-4 p-3 bg-destructive/10 text-destructive rounded-md">
            {audioError.includes("Could not play speech feedback audio") ? (
              <>
                <p className="mb-2">Could not play speech feedback audio.</p>
                <p>
                  Please contact the developers to let them know about this
                  issue.
                </p>
              </>
            ) : (
              audioError
            )}
          </div>
        )}

        {/* Feedback steps */}
        <div className="mt-8 relative">
          <div className="absolute left-4 inset-y-0 w-0.5 bg-muted" />

          <AnimatePresence mode="wait">
            {steps[currentStep] && (
              <motion.div
                key={steps[currentStep].id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className="ml-6 relative pb-8"
              >
                <span className="flex items-center justify-center w-10 h-10 rounded-full bg-primary text-primary-foreground absolute -left-10">
                  {steps[currentStep].emoji}
                </span>
                <div className="bg-card border rounded-lg p-5 shadow-sm hover:shadow-md transition-shadow">
                  <h3 className="text-lg font-medium mb-3">
                    {steps[currentStep].title}
                  </h3>
                  <p className="text-muted-foreground">
                    {steps[currentStep].content}
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Step indicators */}
          <div className="flex gap-2 mt-6 justify-center">
            {steps.map((_, i) => (
              <button
                key={i}
                className={`w-2.5 h-2.5 rounded-full transition-colors ${
                  i === currentStep ? "bg-primary" : "bg-muted"
                }`}
                onClick={() => setCurrentStep(i)}
                aria-label={`Step ${i + 1}`}
              />
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function RecordingPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [transcription, setTranscription] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [recordingAnalysis, setRecordingAnalysis] =
    useState<RecordingAnalysis | null>(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isGettingFeedback, setIsGettingFeedback] = useState(false);
  const [feedbackAudio, setFeedbackAudio] = useState<string | undefined>(
    undefined
  );
  // Add a state to handle not found cases on the client side
  const [isNotFound, setIsNotFound] = useState(false);
  const filename = params.filename as string;
  // Check for audio in query param first, then derive it if not available
  let audioFilename = searchParams.get("audio");

  // If audio parameter is not provided, derive it from the video filename
  if (!audioFilename) {
    // Make sure we use the base filename (without extension) for deriving audio filename
    const baseFilename = filename.replace(".mp4", "");
    // Don't add .wav extension here - it's added when needed
    audioFilename = `${baseFilename}_audio`;
    console.log("Derived audio filename:", audioFilename);
  }

  const videoRef = useRef<HTMLVideoElement>(null);
  const [showTranscription, setShowTranscription] = useState(false);
  const mountedRef = useRef(true);
  const [feedbackRequestInProgress, setFeedbackRequestInProgress] =
    useState(false);
  const [clientSideRendered, setClientSideRendered] = useState(false);
  // Initialise isValidAccess to null to indicate undetermined status initially
  const [isValidAccess, setIsValidAccess] = useState<boolean | null>(null);
  const [serviceError, setServiceError] = useState<string | null>(null);
  const [videoFileExists, setVideoFileExists] = useState<boolean | null>(null);

  // When component mounts, mark as client-side rendered
  useEffect(() => {
    setClientSideRendered(true);
  }, []);

  // Function to check if the video file exists
  const checkVideoFileExists = async () => {
    if (!filename) return;

    try {
      const videoPath = `/uploads/${filename}.mp4`;
      console.log(`Checking if video file exists at: ${videoPath}`);

      const response = await fetch(videoPath, { method: "HEAD" });
      const exists = response.ok;
      console.log(`Video file exists: ${exists}`);
      setVideoFileExists(exists);
    } catch (error) {
      console.error("Error checking video file:", error);
      setVideoFileExists(false);
    }
  };

  // Check for video file existence when component mounts
  useEffect(() => {
    if (clientSideRendered && isValidAccess === true) {
      checkVideoFileExists();
    }
  }, [clientSideRendered, isValidAccess, filename]);

  // Add validation for filename format and device ID access - client-side only
  useEffect(() => {
    // Only run validation logic after client-side render
    if (!clientSideRendered || !filename) return;

    console.log("Running access validation check...");

    // 1) Check if filename matches the expected format: deviceIdHash_YYYYMMDDTHHMMSS
    const isValidFormat = /^[a-zA-Z0-9]+_\d{8}T\d{6}$/.test(filename);
    console.log(
      `Validation Effect: Filename format valid? ${isValidFormat} for ${filename}`
    );

    if (!isValidFormat) {
      console.error("Invalid recording filename format:", filename);
      // router.push("/404");
      setIsNotFound(true);
      return;
    }

    // 2) Check if the user owns this recording (has recording prefix hash digest can be derived from deviceId)
    const recordingOwnerHashDigest = filename.split("_")[0];
    const userHasAccess = validateDeviceIdHash(recordingOwnerHashDigest);
    console.log(`Validation Effect: User has access? ${userHasAccess}`);

    if (!userHasAccess) {
      console.error("User does not have access to this recording:", filename);
      console.log("Recording owner hash digest:", recordingOwnerHashDigest);
      console.log("Device ID hash:", getDeviceIdHashDigest());
      setIsValidAccess(false);
      return;
    }

    // 3) Check validity of "audio filename" if present
    if (audioFilename) {
      console.log("Checking audio filename validity:", audioFilename);
      const isValidAudioFormat =
        /^[a-zA-Z0-9]+_\d{8}T\d{6}_audio(\.wav)?$/.test(audioFilename);
      if (!isValidAudioFormat) {
        console.error("Invalid audio filename format:", audioFilename);
        setIsNotFound(true);
        return;
      }
    }

    // If all checks pass, set access to true
    console.log("Validation Effect: Setting isValidAccess to true");
    setIsValidAccess(true);
  }, [filename, audioFilename, router, clientSideRendered]);

  // Add this ref to track requests and prevent repetition
  const transcriptionAttempted = useRef(false);
  const feedbackAttempted = useRef(false);
  const analysisLoaded = useRef(false);

  // Add a ref to track the current audio blob URL for cleanup
  const currentAudioBlobUrlRef = useRef<string | undefined>(undefined);

  // Add a key state to force re-render when needed
  const [refreshKey, setRefreshKey] = useState(0);

  // Force a refresh of the component
  const forceRefresh = () => {
    setRefreshKey((prevKey) => prevKey + 1);
  };

  // Function to safely set the feedback audio URL and track it for cleanup
  const setFeedbackAudioUrl = (url: string | undefined) => {
    // Clean up previous blob URL if it exists
    if (
      currentAudioBlobUrlRef.current &&
      currentAudioBlobUrlRef.current.startsWith("blob:")
    ) {
      try {
        URL.revokeObjectURL(currentAudioBlobUrlRef.current);
        console.log("Revoked previous blob URL");
      } catch (e) {
        console.warn("Error revoking previous blob URL:", e);
      }
    }

    // Set the new URL
    setFeedbackAudio(url);
    currentAudioBlobUrlRef.current = url;
  };

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      if (
        currentAudioBlobUrlRef.current &&
        currentAudioBlobUrlRef.current.startsWith("blob:")
      ) {
        try {
          URL.revokeObjectURL(currentAudioBlobUrlRef.current);
          console.log("Cleanup: Revoked blob URL on unmount");
        } catch (e) {
          console.warn("Error during cleanup:", e);
        }
      }
    };
  }, []);

  // Function to get data from localStorage with fallback to old format
  const getLocalStorageItem = (
    key: string,
    fallbackKey: string
  ): string | null => {
    // Try new format first
    let data = localStorage.getItem(key);

    // If not found, try old format as fallback
    if (!data && fallbackKey) {
      data = localStorage.getItem(fallbackKey);

      // If data found in old format, migrate it to new format
      if (data) {
        console.log(
          `Migrating data from old format: ${fallbackKey} to new format: ${key}`
        );
        localStorage.setItem(key, data);
      }
    }

    return data;
  };

  // Function to get transcription
  const getTranscription = async (signal?: AbortSignal) => {
    // Add explicit check for access validity *before* any other logic
    if (isValidAccess !== true) {
      console.log(
        "Skipping transcription - access not definitively valid (isValidAccess: ",
        isValidAccess,
        ")"
      );
      return;
    }

    // Prevent multiple attempts entirely
    if (!audioFilename || isTranscribing || transcriptionAttempted.current) {
      console.log(
        "Skipping transcription - already attempted or in progress or no audio filename"
      );
      return;
    }

    transcriptionAttempted.current = true;
    const MAX_RETRIES = 2;
    let retryCount = 0;

    const attemptTranscription = async (): Promise<void> => {
      retryCount++;
      console.log(`Transcription attempt ${retryCount} of ${MAX_RETRIES}`);
      setIsTranscribing(true);

      try {
        // Debug info
        console.log("Current audioFilename:", audioFilename);

        // Handle extensions properly to avoid duplication
        let cleanFilename = audioFilename;
        // Remove .wav extension if it exists
        if (cleanFilename.endsWith(".wav")) {
          cleanFilename = cleanFilename.slice(0, -4);
          console.log("Removed .wav extension, base filename:", cleanFilename);
        }

        // Try multiple possible formats for the HEAD request
        const urlVariations = [
          `/uploads/${cleanFilename}.wav`, // Correctly add .wav once
          `/uploads/${cleanFilename}`, // Try without extension
        ];

        console.log("Trying these URL variations:", urlVariations);

        let fileExists = false;
        let workingUrl = "";

        for (const url of urlVariations) {
          console.log(`Checking if file exists at: ${url}`);
          try {
            const checkResult = await fetch(url, {
              method: "HEAD",
              signal,
            });

            if (checkResult.ok) {
              console.log(`✅ File found at: ${url}`);
              fileExists = true;
              workingUrl = url;
              break;
            } else {
              console.log(
                `❌ File not found at: ${url} (Status: ${checkResult.status})`
              );
            }
          } catch (err) {
            console.error(`Error checking ${url}:`, err);
          }
        }

        if (!fileExists) {
          console.error("Audio file not found at any of the tried locations");
          setIsTranscribing(false);
          setIsNotFound(true);
          return;
        }

        // Extract the actual filename from the working URL
        const effectiveFilename = workingUrl.split("/").pop();
        console.log(
          `Using effective filename for API request: ${effectiveFilename}`
        );

        // Continue with transcription since the audio file exists
        console.log(
          `Sending transcription request with filename: ${effectiveFilename}`
        );
        const response = await fetch(`/api/speech2text`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            audioFilename: effectiveFilename,
          }),
          signal,
        });

        if (!response.ok) {
          // Try to get error details if available
          let errorMessage = `Failed to fetch transcription: ${response.statusText}`;
          try {
            const errorData = await response.json();
            if (errorData.error) {
              errorMessage = `Speech-to-text error: ${errorData.error}`;
            }
          } catch (e) {
            // If we can't parse the JSON, use the default error message
          }

          // If the service is overloaded or temporarily unavailable, retry after a delay
          if (response.status === 503 || response.status === 429) {
            if (retryCount < MAX_RETRIES) {
              console.log(
                `Will retry transcription after ${retryCount * 2} seconds`
              );
              setTimeout(() => {
                if (mountedRef.current) {
                  attemptTranscription();
                }
              }, retryCount * 2000); // Exponential backoff
              return;
            }
          }

          // If the audio file is missing, redirect to 404
          if (response.status === 404) {
            console.error("Audio file not found on server");
            setIsNotFound(true);
            return;
          }

          // For bad requests or other errors, show the error
          throw new Error(errorMessage);
        }

        // Continue with existing code for successful response
        const result = await response.json();
        console.log("Transcription result:", result);

        if (!result.text) {
          throw new Error("No transcription received from server");
        }

        // Only set state if still mounted
        if (mountedRef.current && !signal?.aborted) {
          setTranscription(result.text);

          // Get the existing analysis data from localStorage
          const analysisKey = getAnalysisKey(filename);
          const oldAnalysisKey = `recording_analysis_${filename}.mp4`;
          const existingAnalysis = getLocalStorageItem(
            analysisKey,
            oldAnalysisKey
          );

          // Get the metadata to ensure we have the correct duration
          const metadataKey = getMetadataKey(filename);
          const oldMetadataKey = `recording_metadata_${filename}.mp4`;
          const metadata = getLocalStorageItem(metadataKey, oldMetadataKey);
          const parsedMetadata = metadata ? JSON.parse(metadata) : null;
          const correctDuration = parsedMetadata?.duration || 0;

          // Merge the analysis data
          const analysisObj = mergeAnalysisData(
            existingAnalysis,
            result.analysis
          );
          console.log("Setting merged analysis object:", analysisObj);
          setAnalysis(analysisObj);

          // Make sure we have a recordingAnalysis object even for new recordings
          if (!recordingAnalysis) {
            console.log("Setting default recordingAnalysis for new recording");
            setRecordingAnalysis({
              emotions: [],
              gaze: [],
              duration: correctDuration,
            });
          }

          // Get the existing recording analysis data
          const recordingAnalysisKey = getAnalysisKey(filename);
          const oldRecordingAnalysisKey = `recording_analysis_${filename}.mp4`;
          const existingRecordingAnalysis = getLocalStorageItem(
            recordingAnalysisKey,
            oldRecordingAnalysisKey
          );
          const parsedRecordingAnalysis = existingRecordingAnalysis
            ? JSON.parse(existingRecordingAnalysis)
            : null;

          // Save the complete analysis data to localStorage with correct duration
          const completeAnalysis = {
            ...analysisObj,
            text: result.text,
            emotions: parsedRecordingAnalysis?.emotions || [],
            gaze: parsedRecordingAnalysis?.gaze || [],
            duration: correctDuration, // Use the duration from metadata
            category: parsedMetadata?.category || null,
          };
          localStorage.setItem(analysisKey, JSON.stringify(completeAnalysis));
          console.log(
            "Saved complete analysis to localStorage:",
            completeAnalysis
          );

          // Now that we have successfully transcribed and saved the data to localStorage,
          // clean up the server-side audio file to save space
          if (effectiveFilename) {
            console.log(
              `Requesting cleanup for audio file: ${effectiveFilename}`
            );
            // Use fetch API to request file cleanup
            fetch("/api/cleanup-audio", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ audioFilename: effectiveFilename }),
            })
              .then((response) => response.json())
              .then((data) => {
                if (data.success) {
                  console.log(
                    `Successfully cleaned up audio file on server: ${effectiveFilename}`
                  );

                  // Delete video file if not already cleaned up
                  const videoFilename = filename + ".mp4";
                  cleanupVideoFile(videoFilename);
                } else {
                  console.warn(
                    `Failed to clean up audio file: ${
                      data.error || "Unknown error"
                    }`
                  );
                }
              })
              .catch((error) => {
                console.error("Error cleaning up audio file:", error);
              });
          }

          // Mark that we should try getting feedback next
          feedbackAttempted.current = true;
        }
      } catch (error) {
        console.error(`Transcription attempt ${retryCount} failed:`, error);

        // Don't retry for server errors marked with noRetry
        // @ts-ignore
        if (error.cause?.noRetry) {
          console.log("Not retrying due to terminal server error");
          throw error;
        }

        // Only retry up to the maximum number of attempts
        if (retryCount < MAX_RETRIES) {
          console.log(`Retrying transcription (${retryCount}/${MAX_RETRIES})`);
          return attemptTranscription();
        }

        throw error;
      } finally {
        if (mountedRef.current && !signal?.aborted) {
          setIsTranscribing(false);
        }
      }
    };

    try {
      await attemptTranscription();
    } catch (error) {
      if (mountedRef.current && !signal?.aborted) {
        console.error("All transcription attempts failed:", error);
        setError(
          error instanceof Error
            ? `Transcription failed: ${error.message}`
            : "Error transcribing audio. Please try again."
        );
      }
    } finally {
      if (mountedRef.current && !signal?.aborted) {
        setIsTranscribing(false);
      }
    }
  };

  // The transcription function - add early return for errors
  useEffect(() => {
    // Clear flags on mount, set mounted flag
    transcriptionAttempted.current = false;
    feedbackAttempted.current = false;
    analysisLoaded.current = false;
    mountedRef.current = true;

    // Wait for isValidAccess to be definitively set before doing anything
    if (isValidAccess === null) {
      return;
    }

    const controller = new AbortController();

    // Create a coordinated function to handle both localStorage check and transcription
    const initialiseData = async () => {
      // Skip data initialisation if access is not valid or not yet determined
      if (isValidAccess !== true) {
        console.log(
          `Skipping data initialisation - access not definitively valid (isValidAccess: ${isValidAccess})`
        );
        return;
      }

      console.log("Initialising data - checking localStorage first");

      // Get the analysis key and data first - we'll use it for multiple checks
      const analysisKey = getAnalysisKey(filename);
      const analysisData = localStorage.getItem(analysisKey);
      const hasCompleteData =
        analysisData &&
        JSON.parse(analysisData)?.text &&
        JSON.parse(analysisData)?.total_words;

      // First check if audio file exists (only if localStorage doesn't have complete data)
      if (!hasCompleteData && audioFilename) {
        try {
          console.log(
            `Checking if audio file exists: /uploads/${audioFilename}.wav`
          );
          const audioFileCheck = await fetch(`/uploads/${audioFilename}.wav`, {
            method: "HEAD",
          });

          if (!audioFileCheck.ok) {
            console.error(
              `Audio file not found: /uploads/${audioFilename}.wav`
            );
            setIsNotFound(true);
            return;
          }
          console.log("Audio file exists, continuing initialization");
        } catch (error) {
          console.error("Error checking audio file:", error);
          setIsNotFound(true);
          return;
        }
      }

      // Check if transcription exists in localStorage
      if (analysisData) {
        try {
          const data = JSON.parse(analysisData);
          console.log("Found analysis data in localStorage");

          // Set recording analysis if available
          if (data.emotions && data.gaze) {
            const topEmotions = getTopEmotions(data.emotions);
            const dominantGaze = getDominantGazeDirection(data.gaze);
            setRecordingAnalysis({
              emotions: topEmotions,
              gaze: dominantGaze,
              duration: data.duration || 0,
            });
          } else {
            // Set default empty recording analysis object for new videos
            console.log(
              "No emotions/gaze data found, setting default recording analysis"
            );
            setRecordingAnalysis({
              emotions: [],
              gaze: [],
              duration: data.duration || 0,
            });
          }

          // Set transcription and analysis if available
          if (data.text && data.total_words) {
            console.log(
              "Found complete transcription data in localStorage, skipping API request"
            );
            setTranscription(data.text);
            const analysisObj = {
              total_words: data.total_words,
              filler_count: data.filler_count || 0,
              filler_percentage: data.filler_percentage || 0,
              found_fillers: data.found_fillers || [],
              filler_emoji: data.filler_emoji || "🎯",
              ttr_analysis: data.ttr_analysis || {
                ttr: 0,
                unique_words: 0,
                diversity_level: "low",
                emoji: "📊",
              },
              logical_flow: data.logical_flow || {
                score: 0,
                emoji: "📈",
              },
            };
            console.log("Setting analysis object:", analysisObj);
            setAnalysis(analysisObj);

            analysisLoaded.current = true;
            return true;
          }
        } catch (e) {
          console.error("Error parsing localStorage data:", e);
        }
      }

      // If we get here, we didn't find complete data in localStorage
      if (audioFilename && !transcriptionAttempted.current) {
        console.log(
          "No complete data in localStorage, requesting transcription"
        );
        getTranscription(controller.signal);
      }

      // Still check for cached audio data regardless
      checkCachedAudioFeedback();
    };

    // Function to check for cached audio feedback
    const checkCachedAudioFeedback = () => {
      // Skip audio feedback check if access is not valid or undetermined
      if (isValidAccess !== true) {
        console.log(
          `Skipping cached audio feedback check - access not definitively valid (isValidAccess: ${isValidAccess})`
        );
        return;
      }

      const cachedAudioKey = getAudioFeedbackKey(filename);
      const cachedAudioData = localStorage.getItem(cachedAudioKey);

      if (cachedAudioData) {
        try {
          console.log("Found cached audio feedback data during data load");

          // Convert Base64 to blob and create a new blob URL
          const byteCharacters = atob(cachedAudioData);
          const byteNumbers = new Array(byteCharacters.length);

          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }

          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], { type: "audio/wav" });
          const feedbackUrl = URL.createObjectURL(blob);

          setFeedbackAudioUrl(feedbackUrl);
          feedbackAttempted.current = true;
          // Clear any error state since we have valid audio
          setError(null);

          // Force refresh to ensure UI updates with the audio
          forceRefresh();
        } catch (e) {
          console.warn("Error converting cached audio data:", e);
          // If there's an error with the cached data, we'll generate new audio later
        }
      }
    };

    // Start the coordinated initialisation process
    initialiseData();

    return () => {
      // Immediately prevent further requests on unmount
      mountedRef.current = false;
      transcriptionAttempted.current = true;
      feedbackAttempted.current = true;
      controller.abort();
    };
  }, [filename, audioFilename, isValidAccess]); // Add isValidAccess to dependencies

  // Function to safely retry transcription
  const retryTranscription = () => {
    if (transcriptionAttempted.current && !isTranscribing) {
      console.log("Manually retrying transcription once");
      setError(null);
      // Allow one more attempt
      transcriptionAttempted.current = false;
      getTranscription();
    }
  };

  // Add a state variable to track TTS service failures
  const [ttsServiceFailed, setTtsServiceFailed] = useState(false);

  // Effect to automatically trigger audio feedback after transcription is complete
  useEffect(() => {
    // Skip if any of these conditions are met
    if (
      !audioFilename ||
      !transcription ||
      feedbackRequestInProgress ||
      feedbackAudio ||
      error || // Skip if there's already an error
      ttsServiceFailed // Skip if TTS service previously failed
    ) {
      return;
    }

    // so if we get here, we need to generate new audio feedback
    console.log(
      "Transcription is available, automatically generating audio feedback"
    );

    const autoGenerateFeedback = async () => {
      // Skip if we already have an error or if the TTS service previously failed
      if (error || ttsServiceFailed) {
        console.log(
          "Skipping audio feedback generation due to previous error:",
          error || "TTS service failure"
        );
        return;
      }

      try {
        setFeedbackRequestInProgress(true);
        setIsGettingFeedback(true);
        await getAudioFeedback();
      } catch (error) {
        console.error("Error automatically generating audio feedback:", error);
        // Check if it's a TTS service error
        if (
          error instanceof Error &&
          (error.message.includes("TTS service") ||
            error.message.includes("Internal Server Error"))
        ) {
          // Mark TTS service as failed to prevent further retries
          setTtsServiceFailed(true);
        }
        // Fall back to showing analysis without audio feedback
        setIsGettingFeedback(false);
        setFeedbackRequestInProgress(false);
      }
    };

    autoGenerateFeedback();
  }, [
    audioFilename,
    transcription,
    feedbackAudio,
    feedbackRequestInProgress,
    filename,
    error, // Add error to the dependency array
    ttsServiceFailed, // Add ttsServiceFailed to the dependency array
  ]);

  // Helper functions for analysis
  const getTopEmotions = (
    emotionsData:
      | Array<{
          timestamp: number;
          emotions: { [key: string]: number };
        }>
      | null
      | undefined
  ) => {
    if (!emotionsData || !Array.isArray(emotionsData)) {
      console.warn("No emotions data available");
      return [];
    }

    const emotionSums: { [key: string]: number } = {};
    const emotionCounts: { [key: string]: number } = {};

    emotionsData.forEach(({ emotions }) => {
      if (!emotions) return;
      Object.entries(emotions).forEach(([emotion, value]) => {
        emotionSums[emotion] = (emotionSums[emotion] || 0) + value;
        emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
      });
    });

    const averageEmotions = Object.entries(emotionSums).map(
      ([emotion, sum]) => ({
        emotion,
        average: sum / emotionCounts[emotion],
      })
    );

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

    // Get all directions sorted by count
    const sortedDirections = Object.entries(directionCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([direction, count]) => ({
        direction,
        percentage: ((count / gazeData.length) * 100).toFixed(1),
      }));

    return sortedDirections;
  };

  const mergeAnalysisData = (
    localStorageData: string | null,
    apiData: Partial<AnalysisData> | null
  ): AnalysisData => {
    const defaultAnalysis: AnalysisData = {
      total_words: 0,
      filler_count: 0,
      filler_percentage: 0,
      found_fillers: [],
      filler_emoji: "🎯",
      ttr_analysis: {
        ttr: 0,
        unique_words: 0,
        diversity_level: "low",
        emoji: "📊",
      },
      logical_flow: {
        score: 0,
        emoji: "📈",
      },
    };

    try {
      // Parse localStorage data if available
      const parsedLocalStorage = localStorageData
        ? JSON.parse(localStorageData)
        : null;

      // Start with default values
      let mergedData = { ...defaultAnalysis };

      // Merge with API data if available
      if (apiData) {
        mergedData = {
          ...mergedData,
          ...apiData,
          ttr_analysis: {
            ...mergedData.ttr_analysis,
            ...(apiData.ttr_analysis || {}),
          },
          logical_flow: {
            ...mergedData.logical_flow,
            ...(apiData.logical_flow || {}),
          },
        };
      }

      // Merge with localStorage data if available
      if (parsedLocalStorage) {
        mergedData = {
          ...mergedData,
          filler_percentage:
            parsedLocalStorage.filler_percentage ??
            mergedData.filler_percentage,
          filler_emoji:
            parsedLocalStorage.filler_emoji ?? mergedData.filler_emoji,
          found_fillers:
            parsedLocalStorage.found_fillers ?? mergedData.found_fillers,
          filler_count:
            parsedLocalStorage.filler_count ?? mergedData.filler_count,
          ttr_analysis: {
            ...mergedData.ttr_analysis,
            ...(parsedLocalStorage.ttr_analysis || {}),
          },
          logical_flow: {
            ...mergedData.logical_flow,
            ...(parsedLocalStorage.logical_flow || {}),
          },
        };
      }

      console.log("Merged analysis data:", mergedData);
      return mergedData;
    } catch (error) {
      console.error("Error merging analysis data:", error);
      return defaultAnalysis;
    }
  };

  // Modify the getAudioFeedback function to force a refresh after generating audio
  const getAudioFeedback = async () => {
    // Add explicit check for access validity *before* any other logic
    if (isValidAccess !== true) {
      console.log(
        "Skipping audio feedback generation - access not definitively valid (isValidAccess: ",
        isValidAccess,
        ")"
      );
      return;
    }

    if (!audioFilename || !transcription || feedbackAudio) {
      console.log(
        "Skipping audio feedback - conditions not met or already received"
      );
      return;
    }

    // Skip audio feedback if access is not valid
    if (!isValidAccess) {
      console.log("Skipping audio feedback - access denied");
      return;
    }

    try {
      // Get the cached audio key first - we'll use this multiple times
      const cachedAudioKey = getAudioFeedbackKey(filename);
      const oldCachedAudioKey = `audio_feedback_data_${filename}.mp4`;

      // Step 1: Try cached audio data from localStorage first
      console.log("Step 1: Checking for cached audio data in localStorage");
      const cachedAudioData = getLocalStorageItem(
        cachedAudioKey,
        oldCachedAudioKey
      );

      if (cachedAudioData) {
        try {
          console.log("Using cached audio feedback data from localStorage");
          // Convert Base64 to blob and create a new blob URL
          const byteCharacters = atob(cachedAudioData);
          const byteNumbers = new Array(byteCharacters.length);

          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }

          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], { type: "audio/wav" });
          const feedbackUrl = URL.createObjectURL(blob);

          setFeedbackAudioUrl(feedbackUrl);

          console.log("Successfully loaded audio from localStorage cache");
          return; // Successfully used cached audio
        } catch (e) {
          console.warn(
            "Error using cached audio data, will try next option:",
            e
          );
          // Don't remove the cached data yet - it might just be a decoding issue
        }
      }

      // Step 2: Check if we have cached feedback text in analysis data
      console.log("Step 2: Checking for cached feedback text in analysis data");
      const analysisKey = getAnalysisKey(filename);
      const analysisData = localStorage.getItem(analysisKey);
      let feedbackText = null;
      let category = null;

      if (analysisData) {
        try {
          const parsedData = JSON.parse(analysisData);
          feedbackText = parsedData.feedback_text;
          category = parsedData.category;
          console.log(
            "Found cached feedback text:",
            feedbackText ? "Yes" : "No"
          );
        } catch (e) {
          console.warn("Error parsing analysis data for feedback text:", e);
        }
      }

      // If we have feedback text, use the direct TTS endpoint
      if (feedbackText) {
        console.log("Using cached feedback text for TTS generation");
        let apiUrl = `/api/tts-core`; // Use the TTS-core endpoint for cached text

        // Send the text directly to the TTS API
        const response = await fetch(apiUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: feedbackText, category: category }),
        });

        if (!response.ok) {
          throw new Error(
            `Failed to generate TTS from cached text: ${response.statusText}`
          );
        }

        // Get the audio response and create a blob URL
        const feedbackBlob = await response.blob();
        const feedbackUrl = URL.createObjectURL(feedbackBlob);
        setFeedbackAudioUrl(feedbackUrl);

        console.log("Successfully generated TTS from cached feedback text");

        // Store the newly generated audio in localStorage
        try {
          // Use the improved cacheAudioAsBase64 function
          const cachingResult = await cacheAudioAsBase64(
            feedbackUrl,
            cachedAudioKey,
            {
              stripBase64Prefix: true, // Strip the data URL prefix
              maxSizeMB: 4, // 4MB max size
            },
            manageLocalStorageSpace
          );

          if (cachingResult) {
            console.log(
              "Successfully cached new audio feedback in localStorage"
            );
          } else {
            console.warn("Failed to cache new audio feedback in localStorage");
          }
        } catch (e) {
          console.warn("Error caching new audio:", e);
        }

        return;
      }

      // Step 3: If no cached data, check if the audio file exists on server
      console.log("Step 3: Checking if the audio file exists on server");

      // Check if the audio file exists before trying to use it
      const recordingPath = `/uploads/${audioFilename}.wav`;
      console.log(`Checking if audio file exists at: ${recordingPath}`);

      let audioFileExists = false;
      try {
        const headResponse = await fetch(recordingPath, { method: "HEAD" });
        audioFileExists = headResponse.ok;
        console.log(`Audio file exists: ${audioFileExists}`);
      } catch (headError) {
        console.error("Error checking audio file existence:", headError);
        audioFileExists = false;
      }

      if (!audioFileExists) {
        console.log(
          `Audio file not found at ${recordingPath}, checking for analysis data`
        );

        // Check if we have analysis data with transcription but just no audio file
        const analysisKey = getAnalysisKey(filename);
        const analysisData = localStorage.getItem(analysisKey);

        if (analysisData) {
          try {
            const parsedData = JSON.parse(analysisData);

            // If we have text in the analysis, we can generate audio from it
            if (parsedData.text) {
              console.log(
                "Found transcription text in analysis data, generating audio from it"
              );

              // Generate descriptive feedback from the analysis data
              let feedbackText = "";

              // Basic feedback template
              feedbackText = `Here's feedback on your recording. You used ${
                parsedData.total_words || 0
              } words. `;

              // Add vocabulary feedback
              if (
                parsedData.ttr_analysis &&
                parsedData.ttr_analysis.diversity_level
              ) {
                feedbackText += `Your vocabulary diversity is ${parsedData.ttr_analysis.diversity_level}. `;
              }

              // Add logical flow feedback
              if (
                parsedData.logical_flow &&
                parsedData.logical_flow.score !== undefined
              ) {
                const flowScore = Math.round(
                  parsedData.logical_flow.score * 100
                );
                feedbackText += `Your logical flow score is ${flowScore}%. `;
              }

              // Add filler word feedback
              if (parsedData.filler_percentage !== undefined) {
                feedbackText += `Your speech contained about ${parsedData.filler_percentage}% filler words. `;
              }

              // Save the generated feedback text for future use
              parsedData.feedback_text = feedbackText;
              localStorage.setItem(analysisKey, JSON.stringify(parsedData));

              // Call TTS API with the generated feedback
              console.log("Calling TTS API with generated feedback text");
              const ttsApiUrl = `/api/tts-core`;
              const ttsResponse = await fetch(ttsApiUrl, {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  text: feedbackText,
                  category: parsedData.category || null,
                }),
              });

              if (!ttsResponse.ok) {
                throw new Error(
                  `Failed to generate audio from analysis: ${ttsResponse.statusText}`
                );
              }

              // Get the audio response and create a blob URL
              const feedbackBlob = await ttsResponse.blob();
              const feedbackUrl = URL.createObjectURL(feedbackBlob);
              setFeedbackAudioUrl(feedbackUrl);

              console.log("Successfully generated audio from analysis data");

              // Cache the audio for future use
              try {
                const cachedAudioKey = getAudioFeedbackKey(filename);
                const cachingResult = await cacheAudioAsBase64(
                  feedbackUrl,
                  cachedAudioKey,
                  {
                    stripBase64Prefix: true,
                    maxSizeMB: 4,
                  },
                  manageLocalStorageSpace
                );

                if (cachingResult) {
                  console.log(
                    "Successfully cached generated audio in localStorage"
                  );
                }
              } catch (e) {
                console.warn("Error caching generated audio:", e);
              }

              return;
            }
          } catch (e) {
            console.error("Error extracting data from analysis:", e);
          }
        }

        // Only show error if we couldn't generate audio from analysis data
        console.error("No audio file and no usable analysis data found");
        setError(`Audio file not found. Using text analysis only.`);
        return;
      }

      // Since the audio file exists, proceed with the full analysis
      console.log("Audio file exists, using full audio analysis flow");
      let apiUrl = `/api/audio-feedback`; // Switch to audio-feedback for full analysis

      // Use GET request to fetch the audio file
      const audioResponse = await fetch(recordingPath);
      if (!audioResponse.ok) {
        throw new Error(
          `Failed to fetch audio file: ${audioResponse.statusText}`
        );
      }

      const audioBlob = await audioResponse.blob();

      // Get any category information stored in localStorage
      let categoryParam = "";
      try {
        const metadataKey = getMetadataKey(filename);
        const oldMetadataKey = `recording_metadata_${filename}.mp4`;
        const metadata = getLocalStorageItem(metadataKey, oldMetadataKey);
        if (metadata) {
          const parsedMetadata = JSON.parse(metadata);
          if (parsedMetadata.category) {
            categoryParam = `?category=${encodeURIComponent(
              parsedMetadata.category
            )}`;
            console.log(
              "Using category from metadata:",
              parsedMetadata.category
            );
          }
        }
      } catch (e) {
        console.warn("Error reading category from localStorage:", e);
      }

      // Add category parameter if available
      apiUrl += categoryParam;

      // Create form data with the audio file
      const formData = new FormData();
      formData.append("file", audioBlob, audioFilename);

      // Send the file to the feedback API with a timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // Increased from 30000 to 60000 (60 seconds)

      try {
        const response = await fetch(apiUrl, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        // Check if the response is JSON (error) or audio (success)
        const contentType = response.headers.get("content-type") || "";
        if (!response.ok) {
          if (contentType.includes("application/json")) {
            const errorData = await response.json().catch(() => ({}));
            console.error("Audio feedback error:", errorData);
            setServiceError(
              "TTS service error: " +
                (errorData.detailed_error ||
                  errorData.error ||
                  `Server returned ${response.status}`)
            );
            return;
          } else {
            setServiceError(
              `Could not generate audio feedback (${response.status})`
            );
            return;
          }
        }

        // Check if JSON error response was received despite 200 OK
        if (contentType.includes("application/json")) {
          const errorData = await response.json().catch(() => ({}));
          if (errorData.error) {
            console.error(
              "Audio feedback returned error with 200 status:",
              errorData
            );
            throw new Error(errorData.error);
          }
        }

        // Check if response includes category in headers
        const responsePracticeCategory = response.headers.get(
          "X-Practice-Category"
        );
        console.log("Response practice category:", responsePracticeCategory);

        // Get speech metrics from response headers if available
        const speechMetrics = response.headers.get("X-Speech-Metrics");
        if (speechMetrics) {
          try {
            const parsedMetrics = JSON.parse(speechMetrics);
            // Extract the feedback text from the feedback generation function
            const analysedText = await fetch(`/api/generate-feedback-text`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                analysis: parsedMetrics,
                category: responsePracticeCategory || category,
              }),
            });

            if (analysedText.ok) {
              const feedbackTextResponse = await analysedText.json();

              // Save the feedback text to localStorage for future use
              if (feedbackTextResponse.feedback_text && analysisData) {
                const parsedAnalysis = JSON.parse(analysisData);
                parsedAnalysis.feedback_text =
                  feedbackTextResponse.feedback_text;
                localStorage.setItem(
                  analysisKey,
                  JSON.stringify(parsedAnalysis)
                );
                console.log(
                  "Saved feedback text to analysis data for future use"
                );
              }
            }
          } catch (e) {
            console.warn("Error generating or saving feedback text:", e);
          }
        }

        // If the server sent back a category, update localStorage with it
        if (
          responsePracticeCategory &&
          responsePracticeCategory !== "unknown"
        ) {
          try {
            // Update metadata if it exists
            const metadataKey = getMetadataKey(filename);
            const oldMetadataKey = `recording_metadata_${filename}.mp4`;
            const metadata = getLocalStorageItem(metadataKey, oldMetadataKey);
            if (metadata) {
              const parsedMetadata = JSON.parse(metadata);
              parsedMetadata.category = responsePracticeCategory;
              localStorage.setItem(metadataKey, JSON.stringify(parsedMetadata));
              console.log(
                "Updated metadata with server category:",
                responsePracticeCategory
              );
            }

            // Update analysis data
            const analysisKey = getAnalysisKey(filename);
            const oldAnalysisKey = `recording_analysis_${filename}.mp4`;
            const analysisData = getLocalStorageItem(
              analysisKey,
              oldAnalysisKey
            );
            if (analysisData) {
              const parsedData = JSON.parse(analysisData);
              parsedData.category = responsePracticeCategory;
              localStorage.setItem(analysisKey, JSON.stringify(parsedData));
              console.log(
                "Updated analysis data with server category:",
                responsePracticeCategory
              );
            }
          } catch (e) {
            console.warn(
              "Error updating practice category in localStorage:",
              e
            );
          }
        }

        // Get the audio feedback as a blob and create URL
        const feedbackBlob = await response.blob();

        // Check if audio feedback was actually received
        if (feedbackBlob.size === 0) {
          throw new Error("Received empty audio feedback from server");
        }

        if (!feedbackBlob.type.includes("audio")) {
          console.warn("Received non-audio content:", feedbackBlob.type);
        }

        const feedbackUrl = URL.createObjectURL(feedbackBlob);
        setFeedbackAudioUrl(feedbackUrl);

        // Store audio as Base64 string instead of blob URL
        try {
          const reader = new FileReader();
          reader.readAsDataURL(feedbackBlob);
          reader.onloadend = function () {
            const base64data = reader.result as string;
            // Remove the data URL prefix (e.g., "data:audio/wav;base64,")
            const base64Audio = base64data.split(",")[1];

            // Check size before attempting to save to localStorage
            const estimatedSize = base64Audio.length * 2; // Rough estimate of size in bytes
            const MAX_SIZE = 4 * 1024 * 1024; // 4MB safe limit for localStorage

            if (estimatedSize < MAX_SIZE) {
              try {
                localStorage.setItem(cachedAudioKey, base64Audio);
                console.log(
                  "Saved audio feedback data to localStorage for future use"
                );
              } catch (storageError) {
                console.warn(
                  "localStorage quota exceeded, attempting to clear older audio data"
                );

                // Try to make room by removing older audio data
                const madeRoom = manageLocalStorageSpace(
                  estimatedSize,
                  cachedAudioKey
                );

                if (madeRoom) {
                  // Try again after clearing space
                  try {
                    localStorage.setItem(cachedAudioKey, base64Audio);
                    console.log(
                      "Successfully saved audio after clearing older data"
                    );
                  } catch (retryError) {
                    console.warn(
                      "Still unable to save audio data after clearing space:",
                      retryError
                    );
                    // Silently fail - the app will regenerate audio next time
                  }
                } else {
                  console.warn(
                    "Could not free enough space for new audio data"
                  );
                }
              }
            } else {
              console.warn(
                `Audio data too large for localStorage (${(
                  estimatedSize /
                  1024 /
                  1024
                ).toFixed(2)}MB)`
              );
            }

            // Force refresh to ensure the UI updates
            forceRefresh();
          };
        } catch (e) {
          console.warn("Error saving audio data to localStorage:", e);
        }

        console.log("Successfully received audio feedback:", {
          size: feedbackBlob.size,
          type: feedbackBlob.type,
        });

        // Force a refresh after successful audio generation
        forceRefresh();
      } catch (abortError: unknown) {
        clearTimeout(timeoutId);
        if (abortError instanceof Error && abortError.name === "AbortError") {
          throw new Error("Audio feedback request timed out");
        }
        // For any other type of error, wrap it in an Error
        if (abortError instanceof Error) {
          throw abortError;
        } else {
          throw new Error("Unknown error during audio feedback request");
        }
      }
    } catch (error) {
      console.error("Error getting audio feedback:", error);

      // Format the error message
      let errorMessage = "Failed to get audio feedback";

      if (error instanceof Error) {
        // Check for specific error patterns
        const errorText = error.message.toLowerCase();

        if (
          errorText.includes("api key") ||
          errorText.includes("unauthorized") ||
          errorText.includes("authentication")
        ) {
          errorMessage = "API key issue: " + error.message;
        } else if (
          errorText.includes("quota") ||
          errorText.includes("limit") ||
          errorText.includes("credit")
        ) {
          errorMessage = "API quota exceeded: " + error.message;
        } else if (
          errorText.includes("internal server") ||
          errorText.includes("status 500")
        ) {
          errorMessage = "TTS service error: " + error.message;
        } else {
          errorMessage = error.message;
        }
      }

      setError(errorMessage);

      // Despite the error, we'll continue showing the analysis
      // We're throwing the error so the calling function knows there was a problem
      throw new Error(errorMessage);
    } finally {
      setFeedbackRequestInProgress(false);
      setIsGettingFeedback(false);
    }
  };

  // Add a function to handle audio errors
  const handleAudioError = () => {
    console.log("Audio error detected, clearing feedbackAudio state");
    setFeedbackAudio(undefined);

    // Remove invalid cached blob URL/data
    const cachedAudioKey = getAudioFeedbackKey(filename);
    localStorage.removeItem(cachedAudioKey);

    // Trigger regeneration of audio feedback if we have transcription
    if (transcription && !feedbackRequestInProgress) {
      console.log("Regenerating audio feedback after error");
      const regenerateFeedback = async () => {
        try {
          setFeedbackRequestInProgress(true);
          setIsGettingFeedback(true);
          await getAudioFeedback();
        } catch (error) {
          console.error("Error regenerating audio feedback:", error);
          setIsGettingFeedback(false);
          setFeedbackRequestInProgress(false);
        }
      };

      regenerateFeedback();
    }
  };

  // Add this new function to manage localStorage space
  const manageLocalStorageSpace = (
    requiredBytes: number,
    currentKey: string
  ): boolean => {
    try {
      console.log(
        `Attempting to free approximately ${(
          requiredBytes /
          1024 /
          1024
        ).toFixed(2)}MB of localStorage space`
      );

      // Estimate current usage (better than using a fixed value)
      const estimateLocalStorageUsage = (): number => {
        let total = 0;
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (key) {
            const value = localStorage.getItem(key);
            if (value) {
              // UTF-16 strings use 2 bytes per character
              total += (key.length + value.length) * 2;
            }
          }
        }
        return total;
      };

      const initialUsage = estimateLocalStorageUsage();
      const MAX_STORAGE = 5 * 1024 * 1024; // 5MB is a common browser limit
      const estimatedAvailable = MAX_STORAGE - initialUsage;

      console.log(
        `Estimated current usage: ${(initialUsage / 1024 / 1024).toFixed(
          2
        )}MB, Available: ${(estimatedAvailable / 1024 / 1024).toFixed(2)}MB`
      );

      // If we already have enough space, no need to clear anything
      if (estimatedAvailable > requiredBytes * 1.2) {
        // 20% safety margin
        console.log("Sufficient space already available");
        return true;
      }

      // STAGE 1: Clear audio feedback data first (oldest first)
      // --------------------------------------------------------
      // Get all keys in localStorage
      const allKeys = Object.keys(localStorage);

      // Filter only audio feedback keys
      const audioFeedbackKeys = allKeys.filter((key) =>
        key.startsWith("audio_feedback_data_")
      );

      // Skip the current key we're trying to save
      const otherAudioKeys = audioFeedbackKeys.filter(
        (key) => key !== currentKey
      );

      if (otherAudioKeys.length > 0) {
        // Enhanced timestamp extraction that handles multiple formats
        const keysByDate = otherAudioKeys.map((key) => {
          let timestamp = 0;

          // Try multiple patterns for timestamp extraction
          // Pattern 1: deviceIdHash_YYYYMMDDTHHMMSS
          const dateMatch = key.match(/[a-z0-9]+_(\d{8})T(\d{6})/);
          if (dateMatch) {
            try {
              const year = dateMatch[1].substring(0, 4);
              const month = dateMatch[1].substring(4, 6);
              const day = dateMatch[1].substring(6, 8);
              const hour = dateMatch[2].substring(0, 2);
              const minute = dateMatch[2].substring(2, 4);
              const second = dateMatch[2].substring(4, 6);

              const date = new Date(
                `${year}-${month}-${day}T${hour}:${minute}:${second}`
              );
              timestamp = date.getTime();
            } catch (e) {
              console.warn(`Could not parse date from ${key}:`, e);
            }
          }

          // If no timestamp is extracted, use a fallback approach - last modified timestamp
          // This uses key extraction time as an approximation
          if (timestamp === 0) {
            // If we can't extract a timestamp, use an estimation based on the key
            // Newer keys typically have higher localStorage indices
            const index = allKeys.indexOf(key);
            timestamp = index; // Use the index as a proxy for time (higher = newer)
          }

          // Get size for this item
          const item = localStorage.getItem(key);
          const size = item ? item.length * 2 : 0;

          return {
            key,
            timestamp,
            size,
          };
        });

        // Sort by timestamp, oldest first
        keysByDate.sort((a, b) => a.timestamp - b.timestamp);

        // Track freed space
        let freedSpace = 0;
        let removedCount = 0;

        // Remove oldest entries until we have enough space or run out of entries
        for (const entry of keysByDate) {
          const item = localStorage.getItem(entry.key);
          if (item) {
            // More accurate size calculation
            const itemSize = item.length * 2;
            localStorage.removeItem(entry.key);
            freedSpace += itemSize;
            removedCount++;

            console.log(
              `Removed old audio feedback: ${entry.key} (${(
                itemSize /
                1024 /
                1024
              ).toFixed(2)}MB, total freed: ${(
                freedSpace /
                1024 /
                1024
              ).toFixed(2)}MB)`
            );

            // Check if we've cleared enough space with a safety margin
            if (freedSpace >= requiredBytes * 1.2) {
              console.log(
                `Successfully freed enough space from audio data: ${(
                  freedSpace /
                  1024 /
                  1024
                ).toFixed(2)}MB`
              );
              return true;
            }
          }
        }

        // If we removed some items but not enough space yet
        console.log(
          `Freed ${(freedSpace / 1024 / 1024).toFixed(
            2
          )}MB from ${removedCount} audio items, but need more space`
        );
      } else {
        console.log("No audio feedback data to clear");
      }

      // STAGE 2: Clear other non-critical localStorage data if needed
      // ------------------------------------------------------------
      // Define priority for clearing (lower priority = remove first)
      const priorityPatterns = [
        { pattern: /^audio_feedback_data_/, priority: 0 }, // Already processed above
        { pattern: /temp_/, priority: 1 }, // Temporary data
        { pattern: /cache_/, priority: 2 }, // Cached data
        { pattern: /^analysis_/, priority: 3 }, // Analysis data (can be regenerated)
        { pattern: /^metadata_/, priority: 4 }, // Important metadata (remove last)
      ];

      // Get remaining keys excluding the current one
      const remainingKeys = allKeys.filter((key) => key !== currentKey);

      // Assign priorities to keys
      const keysByPriority = remainingKeys.map((key) => {
        // Find the priority for this key
        let priority = 10; // Default high priority (don't remove)
        for (const p of priorityPatterns) {
          if (p.pattern.test(key)) {
            priority = p.priority;
            break;
          }
        }

        // Get the item to estimate size
        const item = localStorage.getItem(key);
        const size = item ? item.length * 2 : 0;

        return { key, priority, size };
      });

      // Sort by priority (ascending - remove lowest priority first)
      keysByPriority.sort((a, b) => a.priority - b.priority);

      // Only clear items with priority < 5 (protect most important data)
      const clearableItems = keysByPriority.filter((item) => item.priority < 5);

      if (clearableItems.length > 0) {
        let additionalFreedSpace = 0;
        let additionalRemoved = 0;

        for (const item of clearableItems) {
          const value = localStorage.getItem(item.key);
          if (value) {
            const itemSize = value.length * 2;
            localStorage.removeItem(item.key);
            additionalFreedSpace += itemSize;
            additionalRemoved++;

            console.log(
              `Removed additional item: ${item.key} (${(
                itemSize /
                1024 /
                1024
              ).toFixed(2)}MB, priority: ${item.priority})`
            );

            // Check if we've cleared enough total space
            const currentUsage = estimateLocalStorageUsage();
            const newAvailable = MAX_STORAGE - currentUsage;

            if (newAvailable > requiredBytes * 1.2) {
              console.log(
                `Successfully freed enough total space: ${(
                  newAvailable /
                  1024 /
                  1024
                ).toFixed(2)}MB available`
              );
              return true;
            }
          }
        }

        console.log(
          `Freed additional ${(additionalFreedSpace / 1024 / 1024).toFixed(
            2
          )}MB from ${additionalRemoved} other items`
        );

        // If we removed anything at all, consider it a partial success
        if (additionalRemoved > 0) {
          return true;
        }
      } else {
        console.log("No additional clearable items found");
      }

      // STAGE 3: As a last resort, clear everything except most critical data
      // ------------------------------------------------------------
      if (currentKey.startsWith("audio_feedback_data_")) {
        console.log(
          "Last resort: clearing almost everything to make room for audio feedback"
        );

        // Keep only the current key and any critical data items
        const criticalKeys = allKeys.filter(
          (key) =>
            key === currentKey ||
            key.includes("user_preferences") ||
            key.includes("auth")
        );

        const keysToRemove = allKeys.filter(
          (key) => !criticalKeys.includes(key)
        );

        if (keysToRemove.length > 0) {
          let lastResortFreed = 0;

          for (const key of keysToRemove) {
            const value = localStorage.getItem(key);
            if (value) {
              const itemSize = value.length * 2;
              localStorage.removeItem(key);
              lastResortFreed += itemSize;
            }
          }

          console.log(
            `Emergency cleanup: removed ${
              keysToRemove.length
            } items, freed approximately ${(
              lastResortFreed /
              1024 /
              1024
            ).toFixed(2)}MB`
          );
          return true;
        }
      }

      // If we get here, we tried everything but couldn't free enough space
      console.log(
        "Could not free enough localStorage space despite all attempts"
      );
      return false;
    } catch (e) {
      console.warn("Error managing localStorage space:", e);
      return false;
    }
  };

  // Extract the device ID hash from the filename
  const deviceIdHash = filename?.split("_")[0];

  // Handle API errors with more details
  if (error) {
    return <ServiceError error={error} />;
  }

  // Handle not found state
  if (isNotFound) {
    return (
      <main className="container flex flex-col items-center mx-auto p-4 py-8 mt-16">
        <h1 className="text-4xl font-bold mb-8 text-center">
          Recording Not Found
        </h1>
        <p className="text-muted-foreground mb-6">
          The recording you're looking for doesn't exist or has been deleted.
        </p>
        <Link href="/record" className="button button-primary">
          Record a new video
        </Link>
      </main>
    );
  }

  // Loading state while access is being determined
  if (clientSideRendered && isValidAccess === null) {
    return (
      <main className="container flex flex-col items-center mx-auto p-4 py-8 mt-16">
        <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
        <p className="mt-4 text-muted-foreground">Verifying access...</p>
      </main>
    );
  }

  // Always render a base container first to prevent hydration issues
  return (
    <div key={refreshKey}>
      {clientSideRendered && !isValidAccess ? (
        <AccessDenied />
      ) : (
        // Main content - shown initially and kept if access is valid
        <main className="container flex flex-col items-center mx-auto p-4 py-8 mt-16">
          {/* Only render content if we've verified client-side */}
          {!clientSideRendered ? (
            <>
              <h1 className="text-4xl font-bold mb-8 text-center">
                Loading Recorded Video
              </h1>
              <div className="w-full max-w-2xl mb-8 flex justify-center">
                <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
              </div>
            </>
          ) : (
            // Regular content
            <>
              <h1 className="text-4xl font-bold mb-8 text-center">
                Recorded Video
              </h1>
              {/* Debug state info*/}
              {process.env.NODE_ENV === "development" && (
                <div className="w-full max-w-2xl mb-4 text-xs bg-slate-100 dark:bg-slate-800 p-2 rounded">
                  <pre>
                    {JSON.stringify(
                      {
                        hasTranscription: !!transcription,
                        hasAnalysis: !!analysis,
                        hasRecordingAnalysis: !!recordingAnalysis,
                        recordingAnalysis: recordingAnalysis
                          ? {
                              emotions: recordingAnalysis.emotions.length,
                              gaze: recordingAnalysis.gaze.length,
                              duration: recordingAnalysis.duration,
                            }
                          : null,
                        hasAudioFilename: !!audioFilename,
                        hasFeedbackAudio: !!feedbackAudio,
                        isTranscribing,
                        isGettingFeedback,
                        hasError: !!error,
                        refreshKey,
                      },
                      null,
                      2
                    )}
                  </pre>
                </div>
              )}

              {/* Video file loading state */}
              {videoFileExists === null && (
                <Card className="w-full max-w-2xl mb-8">
                  <CardContent className="flex flex-col items-center justify-center gap-4 py-12">
                    <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
                    <p className="text-sm text-muted-foreground">
                      Checking video availability...
                    </p>
                  </CardContent>
                </Card>
              )}

              {/* Message when video file doesn't exist but we have analysis */}
              {videoFileExists === false && !error && (
                <Card className="w-full max-w-2xl mb-8">
                  <CardContent className="py-6">
                    <div className="text-center">
                      <p className="mb-2">
                        The original video file is not available, but we have
                        your speech analysis.
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Video files may be automatically removed after
                        processing to save space.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Video Card - Only shown if video file exists */}
              {videoFileExists && (
                <Card className="w-full max-w-2xl mb-8">
                  <CardContent className="flex flex-col items-center gap-4 pt-6">
                    <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
                      <video
                        ref={videoRef}
                        src={`/uploads/${filename}.mp4`}
                        controls
                        className="w-full h-full object-contain"
                      />
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Rest of the UI components remain the same... */}
              {/* Loading State - Show while either transcribing or getting feedback, but not if there's an error */}
              {(isTranscribing || isGettingFeedback) && !error && (
                <Card className="w-full max-w-2xl mb-8">
                  <CardHeader>
                    <CardTitle className="text-center">
                      {isTranscribing
                        ? "Analysing Speech"
                        : "Generating Audio Feedback"}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="flex flex-col items-center gap-4">
                    <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
                    <p className="text-sm text-muted-foreground text-center">
                      {isTranscribing
                        ? "Reviewing your speech, coherence, and fluency..."
                        : "Generating audio feedback, this may take a few moments..."}
                    </p>
                  </CardContent>
                </Card>
              )}

              {/* Error State - Only show if no audio feedback is available yet */}
              {error &&
                !feedbackAudio &&
                !isTranscribing &&
                !isGettingFeedback && <ServiceError error={error} />}

              {/* Show partial audio feedback error banner if we have audio but had API errors */}
              {error &&
                feedbackAudio &&
                !isTranscribing &&
                !isGettingFeedback && (
                  <div className="w-full max-w-2xl mb-4 p-3 bg-amber-100 dark:bg-amber-900 text-amber-800 dark:text-amber-200 rounded-md">
                    <p className="text-sm">
                      {error.includes("audio feedback")
                        ? "There was an issue generating complete audio feedback, but some analysis is available."
                        : "There was an issue during analysis, but results are available."}
                    </p>
                  </div>
                )}

              {/* Only show full content when everything is ready or when we have at least some data */}
              {transcription &&
                analysis &&
                ((feedbackAudio && !isTranscribing && !isGettingFeedback) ||
                  // Also show content if we have analysis data even without audio feedback
                  (!isTranscribing && !isGettingFeedback)) && (
                  <>
                    {/* Speech Feedback - Only show if feedbackAudio is available */}
                    {feedbackAudio && (
                      <div
                        className="w-full flex justify-center mb-8"
                        data-audio-feedback
                      >
                        <SpeechFeedback
                          analysis={analysis}
                          recordingAnalysis={recordingAnalysis}
                          audioUrl={feedbackAudio}
                          onAudioError={handleAudioError}
                        />
                      </div>
                    )}

                    {/* Analysis Results */}
                    <div className="w-full max-w-2xl space-y-6">
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-center flex items-center justify-center gap-2">
                            Detailed Speech Analysis {analysis.filler_emoji}
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-8">
                          <div className="flex flex-col items-center space-y-4">
                            <div className="w-[250px] h-[250px] relative">
                              <RadialBarChart
                                width={250}
                                height={250}
                                data={[
                                  {
                                    value:
                                      100 - (analysis?.filler_percentage || 0),
                                  },
                                ]}
                                innerRadius={80}
                                outerRadius={140}
                                startAngle={180}
                                endAngle={
                                  180 *
                                  ((analysis?.filler_percentage || 0) / 100)
                                }
                              >
                                <PolarRadiusAxis
                                  type="number"
                                  domain={[0, 100]}
                                  tick={false}
                                  tickCount={10}
                                  axisLine={false}
                                >
                                  <Label
                                    content={(props) => {
                                      if (!props.viewBox) return null;
                                      const viewBox = props.viewBox as {
                                        cx: number;
                                        cy: number;
                                      };
                                      return (
                                        <text
                                          x={viewBox.cx}
                                          y={viewBox.cy}
                                          textAnchor="middle"
                                          dominantBaseline="middle"
                                        >
                                          <tspan
                                            x={viewBox.cx}
                                            y={viewBox.cy - 10}
                                            className="text-2xl font-bold fill-foreground"
                                          >
                                            {analysis?.filler_percentage || 0}%
                                          </tspan>
                                          <tspan
                                            x={viewBox.cx}
                                            y={viewBox.cy + 15}
                                            className="text-sm fill-muted-foreground"
                                          >
                                            Filler Words/Phrases
                                          </tspan>
                                        </text>
                                      );
                                    }}
                                  />
                                </PolarRadiusAxis>
                                <RadialBar
                                  dataKey="value"
                                  cornerRadius={15}
                                  fill={
                                    (analysis?.filler_percentage || 0) >= 18
                                      ? "rgb(239 68 68)" // red-500 (bad)
                                      : (analysis?.filler_percentage || 0) >= 12
                                      ? "rgb(249 115 22)" // orange-500
                                      : (analysis?.filler_percentage || 0) >= 7
                                      ? "rgb(234 179 8)" // yellow-500
                                      : (analysis?.filler_percentage || 0) >= 3
                                      ? "rgb(132 204 22)" // lime-500
                                      : "rgb(34 197 94)" // green-500 (good)
                                  }
                                />
                              </RadialBarChart>
                            </div>
                            <div className="space-y-4 text-center w-full">
                              <div className="space-y-2">
                                <div className="flex items-center justify-center gap-2">
                                  <span className="text-sm font-medium">
                                    Filler Words:
                                  </span>
                                  <span className="text-sm font-medium">
                                    {analysis.filler_emoji}
                                  </span>
                                </div>
                                <p className="text-sm text-muted-foreground">
                                  {analysis.filler_percentage >= 18
                                    ? "Whoa! You're using quite a few filler words — let's work on that!"
                                    : analysis.filler_percentage >= 12
                                    ? "Not bad, but you could cut back on some of those filler words"
                                    : analysis.filler_percentage >= 7
                                    ? "You're right in the middle — keep practicing!"
                                    : analysis.filler_percentage >= 3
                                    ? "Nice job keeping those filler words in check!"
                                    : "Wow, you're crushing it! Barely any filler words!"}
                                </p>
                              </div>
                              <div className="space-y-2 pt-4">
                                <div className="flex items-center justify-center gap-2">
                                  <span className="text-sm font-medium">
                                    Vocabulary Diversity:
                                  </span>
                                  <span className="text-sm font-medium">
                                    {analysis.ttr_analysis.emoji}
                                  </span>
                                </div>
                                <p className="text-sm text-muted-foreground">
                                  {analysis.ttr_analysis.diversity_level ===
                                  "very high"
                                    ? "Outstanding vocabulary range! You're using a rich and diverse set of words."
                                    : analysis.ttr_analysis.diversity_level ===
                                      "high"
                                    ? "Great word variety! Your vocabulary is quite diverse."
                                    : analysis.ttr_analysis.diversity_level ===
                                      "average"
                                    ? "You're using a good mix of words. Keep expanding your vocabulary!"
                                    : analysis.ttr_analysis.diversity_level ===
                                      "low"
                                    ? "Try incorporating more varied words to enhance your speech."
                                    : "Consider broadening your vocabulary to make your speech more engaging."}
                                </p>
                              </div>
                              <div className="space-y-2 pt-4">
                                <div className="flex items-center justify-center gap-2">
                                  <span className="text-sm font-medium">
                                    Logical Flow:
                                  </span>
                                  <span className="text-sm font-medium">
                                    {analysis.logical_flow.emoji}
                                  </span>
                                </div>
                                <p className="text-sm text-muted-foreground">
                                  {analysis.logical_flow.score >= 80
                                    ? "Excellent logical flow! Your ideas connect seamlessly."
                                    : analysis.logical_flow.score >= 60
                                    ? "Good logical progression. Your points flow well together."
                                    : analysis.logical_flow.score >= 40
                                    ? "Average flow. Try to strengthen the connections between ideas."
                                    : analysis.logical_flow.score >= 20
                                    ? "The logical flow needs work. Focus on transitioning between points."
                                    : "Consider restructuring your speech for better logical progression."}
                                </p>
                              </div>
                            </div>
                            <div className="grid grid-cols-3 gap-4 text-sm w-full pt-4">
                              <div className="space-y-1">
                                <p className="text-muted-foreground">
                                  Total Words
                                </p>
                                <p className="font-medium">
                                  {analysis.total_words}
                                </p>
                              </div>
                              <div className="space-y-1">
                                <p className="text-muted-foreground">
                                  Unique Words
                                </p>
                                <p className="font-medium">
                                  {analysis.ttr_analysis.unique_words}
                                </p>
                              </div>
                              <div className="space-y-1">
                                <p className="text-muted-foreground">
                                  Logical Score
                                </p>
                                <p className="font-medium">
                                  {analysis.logical_flow.score}%
                                </p>
                              </div>
                            </div>
                            {analysis.found_fillers.length > 0 && (
                              <div className="space-y-2 w-full pt-4">
                                <p className="text-sm text-muted-foreground">
                                  Found Filler Words:
                                </p>
                                <div className="flex flex-wrap gap-2">
                                  {analysis.found_fillers.map(
                                    (filler, index) => (
                                      <span
                                        key={index}
                                        className="px-2 py-1 bg-muted rounded-full text-xs"
                                      >
                                        {filler}
                                      </span>
                                    )
                                  )}
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Integrated Emotion and Gaze Analysis here - only show if available */}
                          {recordingAnalysis && (
                            <div className="border-t pt-8 mt-8">
                              <h3 className="font-semibold mb-6 text-center">
                                Emotion & Gaze Analysis
                              </h3>
                              <div className="space-y-8">
                                {/* Top Emotions */}
                                <div>
                                  <h4 className="font-semibold mb-4 flex items-center gap-2">
                                    <span>Top Emotions</span>
                                    <span className="text-2xl">😊</span>
                                  </h4>
                                  <div className="grid gap-3">
                                    {recordingAnalysis.emotions.map(
                                      (emotion, index) => (
                                        <div
                                          key={emotion.emotion}
                                          className="flex items-center gap-4"
                                        >
                                          <div className="w-24 text-sm capitalize">
                                            {emotion.emotion}
                                          </div>
                                          <div className="flex-1">
                                            <Progress
                                              value={parseFloat(
                                                emotion.percentage
                                              )}
                                              max={100}
                                              className="h-2"
                                            />
                                          </div>
                                          <div className="w-16 text-sm text-right">
                                            {emotion.percentage}%
                                          </div>
                                        </div>
                                      )
                                    )}
                                  </div>
                                </div>

                                {/* Dominant Gaze */}
                                <div>
                                  <h4 className="font-semibold mb-4 flex items-center gap-2">
                                    <span>Gaze Direction</span>
                                    <span className="text-2xl">👀</span>
                                  </h4>
                                  <div className="space-y-2">
                                    {recordingAnalysis.gaze.map(
                                      (gaze, index) => (
                                        <div
                                          key={gaze.direction}
                                          className="flex items-center gap-4 bg-primary/10 p-4 rounded-lg"
                                        >
                                          <div className="text-lg capitalize">
                                            {gaze.direction}
                                          </div>
                                          <div className="text-sm text-muted-foreground">
                                            (
                                            {index === 0
                                              ? "most of the time"
                                              : "less of the time"}
                                            )
                                          </div>
                                        </div>
                                      )
                                    )}
                                  </div>
                                </div>
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>

                      {/* Transcription Card */}
                      <Card>
                        <CardHeader
                          className="cursor-pointer hover:bg-muted/50 transition-colors"
                          onClick={() =>
                            setShowTranscription(!showTranscription)
                          }
                        >
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-center">
                              View Transcription
                            </CardTitle>
                            {showTranscription ? (
                              <ChevronUp className="h-5 w-5 text-muted-foreground" />
                            ) : (
                              <ChevronDown className="h-5 w-5 text-muted-foreground" />
                            )}
                          </div>
                        </CardHeader>
                        {showTranscription && (
                          <CardContent>
                            <div className="p-4 bg-muted/30 rounded-lg">
                              <p className="whitespace-pre-wrap font-mono text-sm">
                                {transcription}
                              </p>
                            </div>
                          </CardContent>
                        )}
                      </Card>
                    </div>
                  </>
                )}
            </>
          )}
        </main>
      )}
      {serviceError && <ServiceError error={serviceError} />}
    </div>
  );
}

/**
 * Fetches audio from the given URL and caches it as base64 in localStorage
 * This eliminates the need for server-side storage after initial processing
 *
 * @param audioUrl - The URL of the audio file to cache
 * @param audioKey - The localStorage key to use for caching
 * @param options - Optional configuration
 * @returns Promise<boolean> - True if caching was successful, false otherwise
 */
const cacheAudioAsBase64 = async (
  audioUrl: string,
  audioKey: string,
  options?: {
    maxSizeMB?: number; // Maximum size in MB (default: 5MB)
    stripBase64Prefix?: boolean; // Whether to strip the "data:audio/..." prefix (default: false)
  },
  storageManager?: (requiredBytes: number, currentKey: string) => boolean
): Promise<boolean> => {
  const maxSizeMB = options?.maxSizeMB || 5;
  const stripPrefix = options?.stripBase64Prefix || false;

  try {
    console.log(
      `Attempting to cache audio from ${audioUrl} to localStorage (key: ${audioKey})`
    );

    // Check if we already have it cached
    const existingCache = localStorage.getItem(audioKey);
    if (existingCache) {
      console.log("Audio already cached in localStorage");
      return true;
    }

    // Fetch the audio file
    const response = await fetch(audioUrl);
    if (!response.ok) {
      console.error(
        `Failed to fetch audio: ${response.status} ${response.statusText}`
      );
      return false;
    }

    // Convert to blob and then to base64
    const blob = await response.blob();

    // Check file size before attempting to store
    const fileSizeInMB = blob.size / (1024 * 1024);
    if (fileSizeInMB > maxSizeMB) {
      console.warn(
        `Audio file too large to cache (${fileSizeInMB.toFixed(
          2
        )}MB). Max size: ${maxSizeMB}MB`
      );
      return false;
    }

    // Convert to base64
    return new Promise<boolean>((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        try {
          // Get the base64 data
          let base64data = reader.result as string;

          // Optionally strip the prefix
          if (stripPrefix && base64data.includes(",")) {
            base64data = base64data.split(",")[1];
          }

          // Check size before attempting to save to localStorage
          const estimatedSize = base64data.length * 2; // Rough estimate of size in bytes
          const MAX_SIZE = 4 * 1024 * 1024; // 4MB safe limit for localStorage

          if (estimatedSize > MAX_SIZE) {
            console.warn(
              `Audio data too large for localStorage (${(
                estimatedSize /
                1024 /
                1024
              ).toFixed(2)}MB)`
            );
            resolve(false);
            return;
          }

          // Attempt to store in localStorage
          try {
            localStorage.setItem(audioKey, base64data);
            console.log(
              `Successfully cached audio (${(base64data.length / 1024).toFixed(
                2
              )}KB) in localStorage`
            );
            resolve(true);
          } catch (storageError) {
            console.warn(
              "localStorage quota exceeded, attempting to clear older audio data"
            );

            // Try to make room by removing older audio data if we have a storage management function
            if (typeof storageManager === "function") {
              const madeRoom = storageManager(estimatedSize, audioKey);

              if (madeRoom) {
                // Try again after clearing space
                try {
                  localStorage.setItem(audioKey, base64data);
                  console.log(
                    "Successfully saved audio after clearing older data"
                  );
                  resolve(true);
                  return;
                } catch (retryError) {
                  console.warn(
                    "Still unable to save audio data after clearing space:",
                    retryError
                  );
                }
              } else {
                console.warn("Could not free enough space for new audio data");
              }
            }

            console.error("Error saving audio to localStorage:", storageError);
            resolve(false);
          }
        } catch (error) {
          console.error("Error processing audio data:", error);
          resolve(false);
        }
      };
      reader.onerror = () => {
        console.error("Error reading audio file as base64");
        resolve(false);
      };
      reader.readAsDataURL(blob);
    });
  } catch (error) {
    console.error("Error caching audio:", error);
    return false;
  }
};

// Function to get audio URL from localStorage cache if available
const getAudioSource = (audioFilename: string): string => {
  // Generate the cache key using the audioFilename
  const filenameBase = audioFilename.split("_audio")[0];
  const audioCacheKey = `audio_cache_${filenameBase}`;

  // Check if we have the audio cached in localStorage
  const cachedAudio = localStorage.getItem(audioCacheKey);

  if (cachedAudio) {
    console.log("Using cached audio from localStorage");
    return cachedAudio; // Return the base64 data URL
  }

  // Fallback to the server URL if no cache is available
  console.log("No cached audio found, using server URL");
  return `/uploads/${audioFilename}.wav`;
};

// Add a function to clean up the audio file on the server after successful transcription
const cleanupAudioFile = async (audioFilename: string) => {
  try {
    console.log(`Requesting cleanup for audio file: ${audioFilename}`);
    const response = await fetch("/api/cleanup-audio", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        audioFilename,
      }),
    });

    const result = await response.json();
    if (result.success) {
      console.log(
        `Successfully cleaned up audio file on server: ${audioFilename}`
      );
    } else {
      console.warn(`Failed to clean up audio file: ${result.error}`);
    }
  } catch (error) {
    console.error("Error cleaning up audio file:", error);
  }
};

// Add a function to clean up the video file
const cleanupVideoFile = async (videoFilename: string) => {
  try {
    console.log(`Requesting cleanup for video file: ${videoFilename}`);
    const response = await fetch("/api/cleanup-video", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        videoFilename,
      }),
    });

    const result = await response.json();
    if (result.success) {
      console.log(
        `Successfully cleaned up video file on server: ${videoFilename}`
      );
    } else {
      console.warn(`Failed to clean up video file: ${result.error}`);
    }
  } catch (error) {
    console.error("Error cleaning up video file:", error);
  }
};
