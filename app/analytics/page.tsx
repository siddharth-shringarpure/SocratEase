"use client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
  PolarRadiusAxis,
} from "recharts";

interface EmotionEntry {
  emotion: string;
  percentage: number;
  count?: number;
}

interface GazeEntry {
  direction: string;
  percentage: number;
}

interface SpeechAnalysis {
  ttr_analysis: {
    ttr: number;
    unique_words: number;
    diversity_level: string;
  };
  logical_flow: {
    score: number;
  };
  filler_percentage: number;
}

interface AnalyticsAccumulator {
  totalSessions: number;
  totalDuration: number;
  averageEmotions: EmotionEntry[];
  averageGazeDirections: GazeEntry[];
  speechMetrics: {
    vocabularyScores: number[];
    logicalFlowScores: number[];
    fillerPercentages: number[];
  };
}

interface AggregatedAnalytics {
  totalSessions: number;
  totalDuration: number;
  averageEmotions: { emotion: string; percentage: number }[];
  averageGazeDirections: { direction: string; percentage: number }[];
  averageFillerWords: number;
  averageVocabularyScore: number;
  averageLogicalFlow: number;
}

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<AggregatedAnalytics>({
    totalSessions: 0,
    totalDuration: 0,
    averageEmotions: [],
    averageGazeDirections: [],
    averageFillerWords: 0,
    averageVocabularyScore: 0,
    averageLogicalFlow: 0,
  });

  useEffect(() => {
    // Aggregate data from localStorage
    const recordings = Object.keys(localStorage).filter((key) =>
      key.startsWith("recording_analysis_")
    );

    const aggregated = recordings.reduce<AnalyticsAccumulator>(
      (acc, key) => {
        const data = JSON.parse(localStorage.getItem(key) || "{}");
        if (!data.emotions || !data.gaze) return acc;

        // Update total sessions and duration
        acc.totalSessions++;
        acc.totalDuration += data.duration || 0;

        // Aggregate emotions
        data.emotions.forEach((entry: any) => {
          Object.entries(entry.emotions).forEach(([emotion, value]) => {
            const existing = acc.averageEmotions.find(
              (e) => e.emotion === emotion
            );
            if (existing) {
              existing.percentage += value as number;
              existing.count = (existing.count || 1) + 1;
            } else {
              acc.averageEmotions.push({
                emotion,
                percentage: value as number,
                count: 1,
              });
            }
          });
        });

        // Aggregate gaze directions
        data.gaze.forEach((entry: any) => {
          const existing = acc.averageGazeDirections.find(
            (g) => g.direction === entry.direction
          );
          if (existing) {
            existing.percentage += 1;
          } else {
            acc.averageGazeDirections.push({
              direction: entry.direction,
              percentage: 1,
            });
          }
        });

        // Get speech analysis data if available
        try {
          const speechAnalysisKey = `speech_analysis_${key.split("_").pop()}`;
          const speechAnalysis = JSON.parse(
            localStorage.getItem(speechAnalysisKey) || "{}"
          ) as SpeechAnalysis;

          if (speechAnalysis.ttr_analysis) {
            acc.speechMetrics.vocabularyScores.push(
              speechAnalysis.ttr_analysis.ttr * 100
            );
          }
          if (speechAnalysis.logical_flow) {
            acc.speechMetrics.logicalFlowScores.push(
              speechAnalysis.logical_flow.score
            );
          }
          if (typeof speechAnalysis.filler_percentage === "number") {
            acc.speechMetrics.fillerPercentages.push(
              speechAnalysis.filler_percentage
            );
          }
        } catch (error) {
          console.warn("Could not parse speech analysis for recording:", key);
        }

        return acc;
      },
      {
        totalSessions: 0,
        totalDuration: 0,
        averageEmotions: [],
        averageGazeDirections: [],
        speechMetrics: {
          vocabularyScores: [],
          logicalFlowScores: [],
          fillerPercentages: [],
        },
      }
    );

    // Calculate averages
    aggregated.averageEmotions = aggregated.averageEmotions
      .map((emotion) => ({
        emotion: emotion.emotion,
        percentage: (emotion.percentage / (emotion.count || 1)) * 100,
      }))
      .sort((a, b) => b.percentage - a.percentage)
      .slice(0, 5);

    aggregated.averageGazeDirections = aggregated.averageGazeDirections
      .map((gaze) => ({
        direction: gaze.direction,
        percentage: (gaze.percentage / aggregated.totalSessions) * 100,
      }))
      .sort((a, b) => b.percentage - a.percentage);

    // Calculate speech metric averages
    const getAverage = (arr: number[]) =>
      arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

    setAnalytics({
      totalSessions: aggregated.totalSessions,
      totalDuration: aggregated.totalDuration,
      averageEmotions: aggregated.averageEmotions,
      averageGazeDirections: aggregated.averageGazeDirections,
      averageVocabularyScore: getAverage(
        aggregated.speechMetrics.vocabularyScores
      ),
      averageLogicalFlow: getAverage(
        aggregated.speechMetrics.logicalFlowScores
      ),
      averageFillerWords: getAverage(
        aggregated.speechMetrics.fillerPercentages
      ),
    });
  }, []);

  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <main className="container py-8 space-y-8">
      <div className="flex flex-col items-center">
        <h1 className="text-4xl font-bold mb-2">Communication Analytics</h1>
        <p className="text-muted-foreground">
          Your speaking performance insights
        </p>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-center flex items-center justify-center gap-2">
              <span>Practice Sessions</span>
              <span className="text-2xl">📊</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="text-center">
            <div className="text-3xl font-bold">{analytics.totalSessions}</div>
            <p className="text-muted-foreground">Total recorded sessions</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-center flex items-center justify-center gap-2">
              <span>Total Practice Time</span>
              <span className="text-2xl">⏱️</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="text-center">
            <div className="text-3xl font-bold">
              {formatDuration(analytics.totalDuration)}
            </div>
            <p className="text-muted-foreground">Time invested in practice</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-center flex items-center justify-center gap-2">
              <span>Average Flow Score</span>
              <span className="text-2xl">🌊</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="text-center">
            <div className="text-3xl font-bold">
              {analytics.averageLogicalFlow}%
            </div>
            <p className="text-muted-foreground">Speech coherence rating</p>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Analysis */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Emotions Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-center gap-2">
              <span>Emotional Expression</span>
              <span className="text-2xl">😊</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {analytics.averageEmotions.map((emotion) => (
                <div key={emotion.emotion} className="space-y-1.5">
                  <div className="flex items-center justify-between text-sm">
                    <span className="capitalize">{emotion.emotion}</span>
                    <span>{emotion.percentage.toFixed(1)}%</span>
                  </div>
                  <Progress
                    value={emotion.percentage}
                    max={100}
                    className="h-2"
                  />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Gaze Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-center gap-2">
              <span>Gaze Direction Patterns</span>
              <span className="text-2xl">👀</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analytics.averageGazeDirections}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="direction" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="percentage" fill="hsl(var(--primary))" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Speech Quality */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-center gap-2">
              <span>Speech Quality Metrics</span>
              <span className="text-2xl">🎯</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <RadialBarChart
                  innerRadius="30%"
                  outerRadius="100%"
                  data={[
                    {
                      name: "Vocabulary",
                      value: analytics.averageVocabularyScore,
                      fill: "hsl(var(--primary))",
                    },
                    {
                      name: "Flow",
                      value: analytics.averageLogicalFlow,
                      fill: "hsl(var(--primary) / 0.7)",
                    },
                  ]}
                  startAngle={180}
                  endAngle={0}
                >
                  <PolarRadiusAxis type="number" domain={[0, 100]} />
                  <RadialBar
                    background
                    dataKey="value"
                    cornerRadius={15}
                    label={{
                      fill: "#fff",
                      position: "insideStart",
                    }}
                  />
                  <Tooltip />
                </RadialBarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Tips and Insights */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-center gap-2">
              <span>Insights & Recommendations</span>
              <span className="text-2xl">💡</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="p-4 bg-primary/10 rounded-lg">
                <h3 className="font-semibold mb-2">Emotional Range</h3>
                <p className="text-sm text-muted-foreground">
                  Your expressions show good emotional variety. Focus on
                  maintaining natural transitions between emotions.
                </p>
              </div>
              <div className="p-4 bg-primary/10 rounded-lg">
                <h3 className="font-semibold mb-2">Gaze Patterns</h3>
                <p className="text-sm text-muted-foreground">
                  Try to maintain more consistent eye contact with your
                  audience. Vary your gaze naturally across different
                  directions.
                </p>
              </div>
              <div className="p-4 bg-primary/10 rounded-lg">
                <h3 className="font-semibold mb-2">Speech Flow</h3>
                <p className="text-sm text-muted-foreground">
                  Your logical flow score is strong. Continue practicing smooth
                  transitions between topics.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
