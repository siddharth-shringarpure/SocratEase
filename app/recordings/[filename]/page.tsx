"use client";

import { useEffect, useState, useRef, useMemo } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { RadialBarChart, RadialBar, PolarRadiusAxis, Label } from "recharts";
import type { Props as LabelProps } from "recharts/types/component/Label";
import { ChevronDown, ChevronUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Badge } from "@/components/ui/badge";

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

interface Analysis {
  total_words: number;
  filler_count: number;
  filler_percentage: number;
  found_fillers: string[];
  filler_emoji: string;
  ttr_analysis: TTRAnalysis;
  logical_flow: LogicalFlowAnalysis;
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
}: {
  analysis: Analysis | null;
  recordingAnalysis: RecordingAnalysis | null;
  audioUrl: string;
}) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [practiceCategory, setPracticeCategory] = useState<string | null>(null);

  useEffect(() => {
    // Check if the audio URL is a Blob URL (which means it's an enhanced audio)
    if (audioUrl && audioUrl.startsWith("blob:")) {
      // Extract practice category from localStorage based on filename
      const urlParts = window.location.pathname.split("/");
      const filename = urlParts[urlParts.length - 1];

      try {
        // First try metadata
        const metadataKey = `recording_metadata_${filename}`;
        const metadata = localStorage.getItem(metadataKey);
        if (metadata) {
          const parsedMetadata = JSON.parse(metadata);
          if (parsedMetadata.category) {
            setPracticeCategory(parsedMetadata.category);
          }
        } else {
          // Try analysis data
          const analysisKey = `recording_analysis_${filename}`;
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
    if (!analysis || !recordingAnalysis) return [];

    return [
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
      setAudioError("Could not play enhanced audio. Please try again.");
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
    const flowScore = analysis.logical_flow.score;

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
            <span className="font-bold">AI-Enhanced Speech Analysis</span>
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
                  Play Enhanced Audio
                </>
              )}
            </span>
          </Button>
        </div>

        <audio
          ref={audioRef}
          src={audioUrl}
          onEnded={() => setIsPlaying(false)}
          onError={(e) => {
            console.error("Audio error:", e);
            setIsPlaying(false);
            setAudioError(
              "Error playing enhanced audio. Please try again later."
            );
          }}
        />

        {audioError && (
          <div className="mb-4 p-3 bg-destructive/10 text-destructive rounded-md">
            {audioError}
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
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [enhancedAudio, setEnhancedAudio] = useState<string | null>(null);
  const filename = params.filename as string;
  const audioFilename = searchParams.get("audio");
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [showTranscription, setShowTranscription] = useState(false);

  // Function to get transcription
  const getTranscription = async (signal?: AbortSignal) => {
    if (!audioFilename || isTranscribing) return;

    try {
      setIsTranscribing(true);
      setError(null);

      // Fetch the audio file
      console.log("Fetching audio file:", audioFilename);
      const audioResponse = await fetch(
        `http://localhost:5328/uploads/${audioFilename}`,
        { signal }
      );
      if (!audioResponse.ok) {
        throw new Error(
          `Failed to fetch audio file: ${audioResponse.statusText}`
        );
      }

      const audioBlob = await audioResponse.blob();
      console.log("Audio blob size:", audioBlob.size, "bytes");
      console.log("Audio blob type:", audioBlob.type);

      // Send to transcription endpoint
      const formData = new FormData();
      formData.append("file", audioBlob, audioFilename); // Use original filename to preserve extension

      console.log("Sending transcription request...");
      const response = await fetch("http://localhost:5328/api/speech2text", {
        method: "POST",
        body: formData,
        signal,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error("Transcription error response:", errorData);
        throw new Error(
          errorData.error || `Transcription failed: ${response.statusText}`
        );
      }

      const result = await response.json();
      console.log("Transcription result:", result);

      if (!result.text) {
        throw new Error("No transcription received from server");
      }

      // Only set state if the request wasn't aborted
      if (!signal?.aborted) {
        setTranscription(result.text);
        setAnalysis(result.analysis);
      }
    } catch (error) {
      // Only set error state if the request wasn't aborted
      if (error instanceof Error && error.name === "AbortError") {
        console.log("Transcription request aborted");
        return;
      }
      console.error("Error getting transcription:", error);
      setError(
        error instanceof Error ? error.message : "Failed to get transcription"
      );
      setTranscription(null);
      setAnalysis(null);
    } finally {
      // Only reset transcribing state if the request wasn't aborted
      if (!signal?.aborted) {
        setIsTranscribing(false);
      }
    }
  };

  // Function to enhance audio
  const enhanceAudio = async () => {
    if (!audioFilename || isEnhancing) return;

    try {
      setIsEnhancing(true);
      setError(null);

      // Fetch the audio file
      const audioResponse = await fetch(
        `http://localhost:5328/uploads/${audioFilename}`
      );
      if (!audioResponse.ok) {
        throw new Error(
          `Failed to fetch audio file: ${audioResponse.statusText}`
        );
      }

      const audioBlob = await audioResponse.blob();

      // Get practice category from localStorage if available
      let practiceCategory = null;
      try {
        // Try to get the metadata for this recording
        const metadataKey = `recording_metadata_${filename}`;
        const metadata = localStorage.getItem(metadataKey);
        if (metadata) {
          const parsedMetadata = JSON.parse(metadata);
          practiceCategory = parsedMetadata.category;
          console.log("Found practice category:", practiceCategory);
        } else {
          // If no direct metadata, try to extract from analysis data
          const analysisKey = `recording_analysis_${filename}`;
          const analysisData = localStorage.getItem(analysisKey);
          if (analysisData) {
            const parsedData = JSON.parse(analysisData);
            if (parsedData.category) {
              practiceCategory = parsedData.category;
              console.log(
                "Found practice category from analysis:",
                practiceCategory
              );
            }
          }
        }
      } catch (e) {
        console.warn(
          "Error retrieving practice category from localStorage:",
          e
        );
      }

      // Send to enhancement endpoint
      const formData = new FormData();
      formData.append("file", audioBlob, audioFilename);

      // Add category if available
      if (practiceCategory) {
        formData.append("category", practiceCategory);
      }

      const response = await fetch("http://localhost:5328/api/enhance-audio", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.error || `Enhancement failed: ${response.statusText}`
        );
      }

      // Check if response includes category in headers
      const responsePracticeCategory = response.headers.get(
        "X-Practice-Category"
      );
      console.log("Response practice category:", responsePracticeCategory);

      // If the server sent back a category, update localStorage with it
      if (responsePracticeCategory && responsePracticeCategory !== "unknown") {
        try {
          // Update metadata if it exists
          const metadataKey = `recording_metadata_${filename}`;
          const metadata = localStorage.getItem(metadataKey);
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
          const analysisKey = `recording_analysis_${filename}`;
          const analysisData = localStorage.getItem(analysisKey);
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
          console.warn("Error updating practice category in localStorage:", e);
        }
      }

      // Get the enhanced audio as a blob and create URL
      const enhancedBlob = await response.blob();
      const enhancedUrl = URL.createObjectURL(enhancedBlob);
      setEnhancedAudio(enhancedUrl);
    } catch (error) {
      console.error("Error enhancing audio:", error);
      setError(
        error instanceof Error ? error.message : "Failed to enhance audio"
      );
    } finally {
      setIsEnhancing(false);
    }
  };

  // Get transcription when audio file is available
  useEffect(() => {
    if (audioFilename) {
      console.log("Audio file available:", audioFilename);
      const controller = new AbortController();
      getTranscription(controller.signal);
      return () => {
        controller.abort();
      };
    }
  }, [audioFilename]);

  // Get enhanced audio when audio file is available
  useEffect(() => {
    if (audioFilename) {
      enhanceAudio();
    }
  }, [audioFilename]);

  // Load recording analysis data
  useEffect(() => {
    if (filename) {
      console.log("Loading analysis data for filename:", filename);
      const analysisData = localStorage.getItem(
        `recording_analysis_${filename}`
      );
      console.log("Raw analysis data from localStorage:", analysisData);

      if (analysisData) {
        const data = JSON.parse(analysisData);
        console.log("Parsed analysis data:", data);

        const topEmotions = getTopEmotions(data.emotions);
        console.log("Processed top emotions:", topEmotions);

        const dominantGaze = getDominantGazeDirection(data.gaze);
        console.log("Processed dominant gaze:", dominantGaze);

        setRecordingAnalysis({
          emotions: topEmotions,
          gaze: dominantGaze,
          duration: data.duration,
        });
      } else {
        console.log("No analysis data found in localStorage");
      }
    }
  }, [filename]);

  // Helper functions for analysis
  const getTopEmotions = (
    emotionsData: Array<{
      timestamp: number;
      emotions: { [key: string]: number };
    }>
  ) => {
    const emotionSums: { [key: string]: number } = {};
    const emotionCounts: { [key: string]: number } = {};

    emotionsData.forEach(({ emotions }) => {
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

  return (
    <main className="container py-8 flex flex-col items-center">
      <h1 className="text-4xl font-bold mb-8 text-center">Recorded Video</h1>

      {/* Video Card - Moved to the very top */}
      <Card className="w-full max-w-2xl mb-8">
        <CardContent className="flex flex-col items-center gap-4 pt-6">
          <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              src={`http://localhost:5328/uploads/${filename}`}
              controls
              className="w-full h-full object-contain"
            />
          </div>
        </CardContent>
      </Card>

      {/* Enhanced Speech Analysis */}
      {transcription && analysis && recordingAnalysis && audioFilename && (
        <div className="w-full flex justify-center mb-8">
          <SpeechFeedback
            analysis={analysis}
            recordingAnalysis={recordingAnalysis}
            audioUrl={
              enhancedAudio || `http://localhost:5328/uploads/${audioFilename}`
            }
          />
        </div>
      )}

      {/* The rest of the cards and states, but WITHOUT the separate emotions and gaze card */}
      {audioFilename && (
        <div className="w-full max-w-2xl space-y-6">
          {/* Loading State */}
          {isTranscribing && (
            <Card>
              <CardHeader>
                <CardTitle className="text-center">Analysing Speech</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col items-center gap-4">
                <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
                <p className="text-sm text-muted-foreground">
                  Please wait while we analyse your speech...
                </p>
              </CardContent>
            </Card>
          )}

          {/* Error State */}
          {error && !isTranscribing && (
            <Card>
              <CardContent className="py-6">
                <div className="text-center">
                  <p className="text-destructive mb-2">{error}</p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => getTranscription()}
                  >
                    Retry Analysis
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Analysis Results */}
          {analysis && !isTranscribing && (
            <>
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
                        data={[{ value: analysis.filler_percentage }]}
                        innerRadius={80}
                        outerRadius={140}
                        startAngle={180}
                        endAngle={0}
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
                                    {analysis.filler_percentage}%
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
                            analysis.filler_percentage >= 18
                              ? "rgb(239 68 68)" // red-500 (bad)
                              : analysis.filler_percentage >= 12
                              ? "rgb(249 115 22)" // orange-500
                              : analysis.filler_percentage >= 7
                              ? "rgb(234 179 8)" // yellow-500
                              : analysis.filler_percentage >= 3
                              ? "rgb(132 204 22)" // lime-500
                              : "rgb(34 197 94)" // green-500 (good)
                          }
                          background={{ fill: "hsl(var(--muted))" }}
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
                          {analysis.ttr_analysis.diversity_level === "very high"
                            ? "Outstanding vocabulary range! You're using a rich and diverse set of words."
                            : analysis.ttr_analysis.diversity_level === "high"
                            ? "Great word variety! Your vocabulary is quite diverse."
                            : analysis.ttr_analysis.diversity_level ===
                              "average"
                            ? "You're using a good mix of words. Keep expanding your vocabulary!"
                            : analysis.ttr_analysis.diversity_level === "low"
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
                        <p className="text-muted-foreground">Total Words</p>
                        <p className="font-medium">{analysis.total_words}</p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-muted-foreground">Unique Words</p>
                        <p className="font-medium">
                          {analysis.ttr_analysis.unique_words}
                        </p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-muted-foreground">Logical Score</p>
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
                          {analysis.found_fillers.map((filler, index) => (
                            <span
                              key={index}
                              className="px-2 py-1 bg-muted rounded-full text-xs"
                            >
                              {filler}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Integrated Emotion and Gaze Analysis here */}
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
                                      value={parseFloat(emotion.percentage)}
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
                            {recordingAnalysis.gaze.map((gaze, index) => (
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
                            ))}
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
                  onClick={() => setShowTranscription(!showTranscription)}
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
            </>
          )}
        </div>
      )}
    </main>
  );
}
