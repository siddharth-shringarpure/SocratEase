"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { RadialBarChart, RadialBar, PolarRadiusAxis, Label } from "recharts";
import type { Props as LabelProps } from "recharts/types/component/Label";

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

interface Analysis {
  total_words: number;
  filler_count: number;
  filler_percentage: number;
  found_fillers: string[];
  filler_emoji: string;
  ttr_analysis: TTRAnalysis;
  logical_flow: LogicalFlowAnalysis;
}

export default function RecordingPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [transcription, setTranscription] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const filename = params.filename as string;
  const audioFilename = searchParams.get('audio');
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Function to get transcription
  const getTranscription = async (signal?: AbortSignal) => {
    if (!audioFilename || isTranscribing) return;
    
    try {
      setIsTranscribing(true);
      setError(null);
      
      // Fetch the audio file
      console.log('Fetching audio file:', audioFilename);
      const audioResponse = await fetch(`http://localhost:5328/uploads/${audioFilename}`, { signal });
      if (!audioResponse.ok) {
        throw new Error(`Failed to fetch audio file: ${audioResponse.statusText}`);
      }
      
      const audioBlob = await audioResponse.blob();
      console.log('Audio blob size:', audioBlob.size, 'bytes');
      console.log('Audio blob type:', audioBlob.type);
      
      // Send to transcription endpoint
      const formData = new FormData();
      formData.append('file', audioBlob, audioFilename);  // Use original filename to preserve extension
      
      console.log('Sending transcription request...');
      const response = await fetch('http://localhost:5328/api/speech2text', {
        method: 'POST',
        body: formData,
        signal
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('Transcription error response:', errorData);
        throw new Error(errorData.error || `Transcription failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('Transcription result:', result);
      
      if (!result.text) {
        throw new Error('No transcription received from server');
      }
      
      // Only set state if the request wasn't aborted
      if (!signal?.aborted) {
        setTranscription(result.text);
        setAnalysis(result.analysis);
      }
    } catch (error) {
      // Only set error state if the request wasn't aborted
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Transcription request aborted');
        return;
      }
      console.error('Error getting transcription:', error);
      setError(error instanceof Error ? error.message : 'Failed to get transcription');
      setTranscription(null);
      setAnalysis(null);
    } finally {
      // Only reset transcribing state if the request wasn't aborted
      if (!signal?.aborted) {
        setIsTranscribing(false);
      }
    }
  };

  // Get transcription when audio file is available
  useEffect(() => {
    if (audioFilename) {
      console.log('Audio file available:', audioFilename);
      const controller = new AbortController();
      getTranscription(controller.signal);
      return () => {
        controller.abort();
      };
    }
  }, [audioFilename]);

  return (
    <main className="container py-8 flex flex-col items-center">
      <h1 className="text-4xl font-bold mb-8 text-center">Recorded Video</h1>

      <Card className="w-full max-w-2xl mb-8">
        <CardHeader>
          <CardTitle className="text-center">Recording: {filename}</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-4">
          <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              src={`http://localhost:5328/uploads/${filename}`}
              controls
              className="w-full h-full object-contain"
            />
          </div>

          {audioFilename && (
            <div className="w-full">
              <h3 className="font-semibold mb-2">Audio Track:</h3>
              <audio
                ref={audioRef}
                src={`http://localhost:5328/uploads/${audioFilename}`}
                className="w-full"
                controls
              />
              <p className="text-sm text-muted-foreground mt-2">
                Note: Video and audio can be controlled independently
              </p>
            </div>
          )}

          <div className="flex gap-4">
            <Button
              onClick={() => router.push("/camera")}
              variant="outline"
            >
              Back to Camera
            </Button>
          </div>
        </CardContent>
      </Card>

      {audioFilename && (
        <div className="w-full max-w-2xl space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-center">Transcription</CardTitle>
            </CardHeader>
            <CardContent>
              {isTranscribing ? (
                <div className="space-y-2">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-[90%]" />
                  <Skeleton className="h-4 w-[95%]" />
                  <Skeleton className="h-4 w-[85%]" />
                </div>
              ) : transcription ? (
                <p className="whitespace-pre-wrap">{transcription}</p>
              ) : error ? (
                <div className="text-center">
                  <p className="text-destructive mb-2">{error}</p>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => getTranscription()}
                  >
                    Retry Transcription
                  </Button>
                </div>
              ) : null}
            </CardContent>
          </Card>

          {analysis && (
            <Card>
              <CardHeader>
                <CardTitle className="text-center flex items-center justify-center gap-2">
                  Speech Analysis {analysis.filler_emoji}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
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
                            const viewBox = props.viewBox as { cx: number; cy: number };
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
                            ? "rgb(239 68 68)"   // red-500 (bad)
                            : analysis.filler_percentage >= 12
                            ? "rgb(249 115 22)"  // orange-500
                            : analysis.filler_percentage >= 7
                            ? "rgb(234 179 8)"   // yellow-500
                            : analysis.filler_percentage >= 3
                            ? "rgb(132 204 22)"  // lime-500
                            : "rgb(34 197 94)"   // green-500 (good)
                        }
                        background={{ fill: "hsl(var(--muted))" }}
                      />
                    </RadialBarChart>
                  </div>
                  <div className="space-y-4 text-center w-full">
                    <div className="space-y-2">
                      <div className="flex items-center justify-center gap-2">
                        <span className="text-sm font-medium">Filler Words:</span>
                        <span className="text-sm font-medium">{analysis.filler_emoji}</span>
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
                        <span className="text-sm font-medium">Vocabulary Diversity:</span>
                        <span className="text-sm font-medium">{analysis.ttr_analysis.emoji}</span>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {analysis.ttr_analysis.diversity_level === "very high"
                          ? "Outstanding vocabulary range! You're using a rich and diverse set of words."
                          : analysis.ttr_analysis.diversity_level === "high"
                          ? "Great word variety! Your vocabulary is quite diverse."
                          : analysis.ttr_analysis.diversity_level === "average"
                          ? "You're using a good mix of words. Keep expanding your vocabulary!"
                          : analysis.ttr_analysis.diversity_level === "low"
                          ? "Try incorporating more varied words to enhance your speech."
                          : "Consider broadening your vocabulary to make your speech more engaging."}
                      </p>
                    </div>
                    <div className="space-y-2 pt-4">
                      <div className="flex items-center justify-center gap-2">
                        <span className="text-sm font-medium">Logical Flow:</span>
                        <span className="text-sm font-medium">{analysis.logical_flow.emoji}</span>
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
                      <p className="font-medium">{analysis.ttr_analysis.unique_words}</p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-muted-foreground">Logical Score</p>
                      <p className="font-medium">{analysis.logical_flow.score}%</p>
                    </div>
                  </div>
                  {analysis.found_fillers.length > 0 && (
                    <div className="space-y-2 w-full pt-4">
                      <p className="text-sm text-muted-foreground">Found Filler Words:</p>
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
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </main>
  );
} 