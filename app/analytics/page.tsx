"use client";

/**
 * @fileoverview Analytics dashboard displaying user communication metrics and insights.
 * Provides visualisations for emotions, gaze patterns, and speech quality.
 */

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
import { AnalyticsCard } from "@/components/custom/AnalyticsCard";
import { useAnalyticsData } from "@/hooks/useAnalyticsData";
import { AggregatedAnalytics } from "@/types/analytics";

// TODO: Add error boundary to handle chart rendering failures
// TODO: Implement data caching for better performance
// TODO: Add loading states for initial data fetch

/**
 * Main analytics dashboard component showing communication metrics
 * @returns {JSX.Element} Rendered analytics dashboard
 */
export default function AnalyticsPage() {
  const analytics = useAnalyticsData();

  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;

    // Show only relevant units based on duration
    if (seconds < 60) return `${seconds}s`;
    if (remainingSeconds === 0) {
      if (hours === 0) return `${minutes}m`;
      if (minutes === 0) return `${hours}h`;
      return `${hours}h ${minutes}m`;
    }
    if (hours > 0) {
      const roundedMinutes = Math.round((minutes * 60 + remainingSeconds) / 60);
      return roundedMinutes === 0
        ? `${hours}h`
        : `${hours}h ${roundedMinutes}m`;
    }
    return `${minutes}m ${remainingSeconds}s`;
  };

  // Custom tooltip for bar charts showing percentage values
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload?.length) {
      return (
        <div className="bg-background/95 border border-border p-2 rounded-md shadow-sm text-sm">
          <p className="font-medium capitalize">{label}</p>
          <p className="text-primary font-semibold">
            {payload[0].value.toFixed(1)}%
          </p>
        </div>
      );
    }
    return null;
  };

  // Sort emotions by percentage (highest to lowest)
  const sortedEmotions =
    analytics.averageEmotions.length > 0
      ? [...analytics.averageEmotions].sort(
          (a, b) => b.percentage - a.percentage
        )
      : [];

  return (
    <main className="container mx-auto px-4 py-6 sm:py-8 mt-16">
      <div className="flex flex-col items-center mb-8 sm:mb-12">
        <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-2 sm:mb-3 text-center">
          Communication Analytics
        </h1>
        <p className="text-muted-foreground text-base sm:text-lg max-w-md text-center px-4">
          Insights into your speaking performance
        </p>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-4 sm:gap-6 lg:gap-8 md:grid-cols-2 lg:grid-cols-3 mb-8 sm:mb-12">
        <AnalyticsCard
          title="Practice Sessions"
          icon="📊"
          value={
            analytics.totalSessions > 0
              ? analytics.totalSessions.toString()
              : "N/A"
          }
          description={
            analytics.totalSessions > 0
              ? "Total recorded sessions"
              : "No practice sessions yet"
          }
          trend="neutral"
        />

        <AnalyticsCard
          title="Total Practice Time"
          icon="⏱️"
          value={
            analytics.totalDuration > 0
              ? formatDuration(analytics.totalDuration)
              : "N/A"
          }
          description={
            analytics.totalDuration > 0
              ? "Time invested in practice"
              : "Start practicing to track time"
          }
          trend="neutral"
        />

        <AnalyticsCard
          title="Average Flow Score"
          icon="🌊"
          value={
            analytics.averageLogicalFlow !== null
              ? `${analytics.averageLogicalFlow.toFixed(1)}%`
              : "N/A"
          }
          description={
            analytics.averageLogicalFlow !== null
              ? "Speech coherence rating"
              : "Complete a session to see score"
          }
          trend={
            analytics.averageLogicalFlow === null
              ? "neutral"
              : analytics.averageLogicalFlow > 70
              ? "up"
              : analytics.averageLogicalFlow < 40
              ? "down"
              : "neutral"
          }
        />
      </div>

      {/* Detailed Analysis */}
      <div className="grid gap-4 sm:gap-6 lg:gap-8 sm:grid-cols-1 md:grid-cols-2">
        {/* Emotions Analysis */}
        <Card className="shadow-md hover:shadow-lg transition-shadow">
          <CardHeader className="bg-gradient-to-r from-primary/10 to-primary/5 rounded-t-lg py-4 sm:py-6">
            <CardTitle className="flex items-center justify-center gap-2 text-lg sm:text-xl">
              <span>Emotional Expressions</span>
              <span className="text-xl sm:text-2xl">😊</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4 sm:pt-6 px-4 sm:px-6">
            <div className="space-y-3 sm:space-y-5">
              {sortedEmotions.length > 0 ? (
                sortedEmotions.map((emotion) => (
                  <div key={emotion.emotion} className="space-y-1 sm:space-y-2">
                    <div className="flex items-center justify-between text-xs sm:text-sm font-medium">
                      <span className="capitalize">{emotion.emotion}</span>
                      <span className="text-primary">
                        {emotion.percentage.toFixed(1)}%
                      </span>
                    </div>
                    <Progress
                      value={emotion.percentage}
                      max={100}
                      className="h-2 sm:h-2.5 bg-muted/50"
                    />
                  </div>
                ))
              ) : (
                <div className="flex flex-col items-center justify-center py-8 sm:py-12 text-muted-foreground space-y-2">
                  <span className="text-xl sm:text-2xl">🎭</span>
                  <p className="text-sm sm:text-base">
                    No emotion data available yet
                  </p>
                  <p className="text-xs sm:text-sm text-center">
                    Record a practice session to analyse your emotional
                    expression
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Gaze Analysis */}
        <Card className="shadow-md hover:shadow-lg transition-shadow">
          <CardHeader className="bg-gradient-to-r from-primary/10 to-primary/5 rounded-t-lg py-4 sm:py-6">
            <CardTitle className="flex items-center justify-center gap-2 text-lg sm:text-xl">
              <span>Gaze Direction Patterns</span>
              <span className="text-xl sm:text-2xl">👀</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4 sm:pt-6 px-2 sm:px-4">
            <div className="h-[250px] sm:h-[300px]">
              {analytics.averageGazeDirections.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={analytics.averageGazeDirections}
                    margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="rgba(var(--muted), 0.2)"
                    />
                    <XAxis
                      dataKey="direction"
                      tick={{
                        fill: "hsl(var(--foreground))",
                        fontSize: "0.75rem",
                      }}
                      tickMargin={8}
                    />
                    <YAxis
                      tick={{
                        fill: "hsl(var(--foreground))",
                        fontSize: "0.75rem",
                      }}
                      width={30}
                    />
                    <Tooltip
                      content={<CustomTooltip />}
                      cursor={false}
                      offset={10}
                      allowEscapeViewBox={{ x: true, y: true }}
                    />
                    <Bar
                      dataKey="percentage"
                      fill="hsl(var(--primary))"
                      radius={[4, 4, 0, 0]}
                      barSize={30}
                      animationDuration={800}
                      isAnimationActive={true}
                      activeBar={{ fill: "hsl(var(--primary)/0.8)" }}
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground space-y-2">
                  <span className="text-xl sm:text-2xl">👀</span>
                  <p className="text-sm sm:text-base">
                    No gaze data available yet
                  </p>
                  <p className="text-xs sm:text-sm text-center">
                    Complete a practice session to track your eye movements
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Speech Quality */}
        <Card className="shadow-md hover:shadow-lg transition-shadow">
          <CardHeader className="bg-gradient-to-r from-primary/10 to-primary/5 rounded-t-lg py-4 sm:py-6">
            <CardTitle className="flex items-center justify-center gap-2 text-lg sm:text-xl">
              <span>Speech Quality Metrics</span>
              <span className="text-xl sm:text-2xl">🎯</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4 sm:pt-6 px-2 sm:px-4">
            <div className="h-[250px] sm:h-[300px]">
              {analytics.averageVocabularyScore !== null &&
              analytics.averageLogicalFlow !== null ? (
                <ResponsiveContainer width="100%" height="100%">
                  <RadialBarChart
                    innerRadius="30%"
                    outerRadius="90%"
                    data={[
                      {
                        name: "Vocabulary Diversity",
                        value: analytics.averageVocabularyScore,
                        fill: "hsl(var(--primary))",
                        description: "How varied your word choice is",
                      },
                      {
                        name: "Logical Flow",
                        value: analytics.averageLogicalFlow,
                        fill: "hsl(var(--primary))",
                        description: "How well your ideas connect",
                      },
                    ]}
                    startAngle={180}
                    endAngle={0}
                  >
                    <PolarRadiusAxis
                      type="number"
                      domain={[0, 100]}
                      tick={{ fill: "hsl(var(--foreground))", fontSize: 10 }}
                      tickFormatter={(value) => `${value}%`}
                      tickCount={6}
                    />
                    <RadialBar
                      background={{ fill: "hsl(var(--muted))" }}
                      dataKey="value"
                      cornerRadius={15}
                      label={{
                        fill: "hsl(var(--foreground))",
                        position: "insideStart",
                        fontSize: 14,
                        fontWeight: "600",
                        formatter: (value: number) => `${value.toFixed(1)}%`,
                      }}
                      animationDuration={1000}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div className="bg-background/95 border border-border p-2 sm:p-3 rounded-lg shadow-sm">
                              <p className="font-medium text-xs sm:text-sm">
                                {data.name}
                              </p>
                              <p className="text-primary font-semibold text-sm sm:text-lg">
                                {data.value.toFixed(1)}%
                              </p>
                              <p className="text-muted-foreground text-xs mt-1">
                                {data.description}
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                  </RadialBarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground space-y-2">
                  <span className="text-xl sm:text-2xl">📊</span>
                  <p className="text-sm sm:text-base">
                    No speech data available yet
                  </p>
                  <p className="text-xs sm:text-sm text-center">
                    Complete a practice session to see your metrics
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Tips and Insights */}
        {/* TODO: Make this more dynamic based on user mode choices */}
        <Card className="shadow-md hover:shadow-lg transition-shadow">
          <CardHeader className="bg-gradient-to-r from-primary/10 to-primary/5 rounded-t-lg py-4 sm:py-6">
            <CardTitle className="flex items-center justify-center gap-2 text-lg sm:text-xl">
              <span>Insights & Recommendations</span>
              <span className="text-xl sm:text-2xl">💡</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4 sm:pt-6 px-4 sm:px-6">
            {analytics.totalSessions > 0 ? (
              <div className="space-y-3 sm:space-y-4">
                {analytics.averageEmotions.length > 0 && (
                  <div className="p-3 sm:p-4 rounded-lg bg-gradient-to-r from-primary/10 to-primary/5 border border-primary/10">
                    <h3 className="font-semibold mb-1 sm:mb-2 flex items-center gap-2">
                      <span className="text-base sm:text-lg text-primary">
                        Emotional Range
                      </span>
                    </h3>
                    <p className="text-xs sm:text-sm text-muted-foreground">
                      Your expressions show good emotional variety. Focus on
                      maintaining natural transitions between emotions.
                    </p>
                  </div>
                )}
                {analytics.averageGazeDirections.length > 1 &&
                  analytics.averageGazeDirections.some(
                    (direction) => direction.direction !== "center"
                  ) && (
                    <div className="p-3 sm:p-4 rounded-lg bg-gradient-to-r from-primary/10 to-primary/5 border border-primary/10">
                      <h3 className="font-semibold mb-1 sm:mb-2 flex items-center gap-2">
                        <span className="text-base sm:text-lg text-primary">
                          Gaze Patterns
                        </span>
                      </h3>
                      <p className="text-xs sm:text-sm text-muted-foreground">
                        Try to maintain more consistent eye contact with your
                        audience. Vary your gaze naturally across different
                        directions.
                      </p>
                    </div>
                  )}
                {analytics.averageLogicalFlow !== null && (
                  <div className="p-3 sm:p-4 rounded-lg bg-gradient-to-r from-primary/10 to-primary/5 border border-primary/10">
                    <h3 className="font-semibold mb-1 sm:mb-2 flex items-center gap-2">
                      <span className="text-base sm:text-lg text-primary">
                        Speech Flow
                      </span>
                    </h3>
                    <p className="text-xs sm:text-sm text-muted-foreground">
                      {analytics.averageLogicalFlow > 70
                        ? "Your logical flow score is strong. Continue practicing smooth transitions between topics."
                        : analytics.averageLogicalFlow > 40
                        ? "Your speech flow is developing well. Focus on connecting ideas more clearly."
                        : "Work on improving the flow between ideas. Try using transition phrases."}
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-8 sm:py-12 text-muted-foreground space-y-2">
                <span className="text-xl sm:text-2xl">💡</span>
                <p className="text-sm sm:text-base">
                  No insights available yet
                </p>
                <p className="text-xs sm:text-sm text-center">
                  Practice to receive personalised recommendations
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
