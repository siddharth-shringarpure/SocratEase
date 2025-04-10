import { useState, useEffect } from "react";
import {
  AggregatedAnalytics,
  AnalyticsAccumulator,
  SpeechAnalysis,
} from "@/types/analytics";
import { getAnalysisKey, FILENAME_HASH_LENGTH } from "@/lib/deviceId";

export function useAnalyticsData(): AggregatedAnalytics {
  /**
   * Hook to calculate aggregated analytics data from localStorage
   */
  const [analytics, setAnalytics] = useState<AggregatedAnalytics>({
    totalSessions: 0,
    totalDuration: 0,
    averageEmotions: [],
    averageGazeDirections: [],
    averageFillerWords: null,
    averageVocabularyScore: null,
    averageLogicalFlow: null,
  });

  useEffect(() => {
    try {
      // Find all keys in localStorage that match the format with hash digest
      // The hash digest is exactly FILENAME_HASH_LENGTH (8) characters long
      const recordings = Object.keys(localStorage).filter((key) => {
        // Match keys that start with "analysis_" followed by exactly 8 hex chars and a timestamp
        return /^analysis_[a-f0-9]{8}_\d{8}T\d{6}$/.test(key);
      });

      console.log(
        `Found ${recordings.length} recording entries in localStorage`
      );

      const aggregated = recordings.reduce<AnalyticsAccumulator>(
        (acc, key) => {
          try {
            let data;
            try {
              data = JSON.parse(localStorage.getItem(key) || "{}");
            } catch (parseError) {
              console.error(
                `Failed to parse localStorage data for key ${key}`,
                parseError
              );
              return acc;
            }

            // Validation
            if (!Array.isArray(data.emotions) || !Array.isArray(data.gaze)) {
              console.warn(
                `Missing or invalid emotions/gaze data for key ${key}`
              );
              return acc;
            }

            // Update total sessions and duration
            acc.totalSessions++;
            acc.totalDuration += data.duration || 0;

            // Aggregate emotions
            data.emotions.forEach((entry: any) => {
              // Check if entry and entry.emotions exist and are not null
              if (
                entry &&
                entry.emotions &&
                typeof entry.emotions === "object"
              ) {
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
              }
            });

            // Aggregate gaze directions
            let totalGazeEntries = 0;
            data.gaze.forEach((entry: any) => {
              if (entry && entry.direction) {
                totalGazeEntries++;
                const existing = acc.averageGazeDirections.find(
                  (g) => g.direction === entry.direction
                );
                if (existing) {
                  existing.count = (existing.count || 0) + 1;
                } else {
                  acc.averageGazeDirections.push({
                    direction: entry.direction,
                    count: 1,
                    percentage: 0,
                  });
                }
              }
            });
            acc.totalGazeEntries += totalGazeEntries;

            // Aggregate speech metrics if available
            if (data.ttr_analysis?.ttr) {
              acc.speechMetrics.vocabularyScores.push(data.ttr_analysis.ttr);
            }
            if (data.logical_flow?.score) {
              acc.speechMetrics.logicalFlowScores.push(data.logical_flow.score);
            }
            if (typeof data.filler_percentage === "number") {
              acc.speechMetrics.fillerPercentages.push(data.filler_percentage);
            }

            return acc;
          } catch (error) {
            console.error(`Error processing recording ${key}:`, error);
            return acc;
          }
        },
        {
          totalSessions: 0,
          totalDuration: 0,
          averageEmotions: [],
          averageGazeDirections: [],
          totalGazeEntries: 0,
          speechMetrics: {
            vocabularyScores: [],
            logicalFlowScores: [],
            fillerPercentages: [],
          },
        }
      );

      // Calculate final averages
      const finalAnalytics: AggregatedAnalytics = {
        totalSessions: aggregated.totalSessions,
        totalDuration: aggregated.totalDuration,
        averageEmotions: aggregated.averageEmotions.map((emotion) => ({
          emotion: emotion.emotion,
          percentage: emotion.count
            ? (emotion.percentage / emotion.count) * 100
            : emotion.percentage * 100,
        })),
        averageGazeDirections: aggregated.averageGazeDirections.map((gaze) => ({
          direction: gaze.direction,
          percentage:
            ((gaze.count || 0) / Math.max(1, aggregated.totalGazeEntries)) *
            100,
        })),
        averageFillerWords:
          aggregated.speechMetrics.fillerPercentages.length > 0
            ? aggregated.speechMetrics.fillerPercentages.reduce(
                (a, b) => a + b,
                0
              ) / aggregated.speechMetrics.fillerPercentages.length
            : null,
        averageVocabularyScore:
          aggregated.speechMetrics.vocabularyScores.length > 0
            ? aggregated.speechMetrics.vocabularyScores.reduce(
                (a, b) => a + b,
                0
              ) / aggregated.speechMetrics.vocabularyScores.length
            : null,
        averageLogicalFlow:
          aggregated.speechMetrics.logicalFlowScores.length > 0
            ? aggregated.speechMetrics.logicalFlowScores.reduce(
                (a, b) => a + b,
                0
              ) / aggregated.speechMetrics.logicalFlowScores.length
            : null,
      };

      console.log("==== Speech Analytics Calculation Completed ====");
      console.log(`Sessions processed: ${finalAnalytics.totalSessions}`);
      console.log("Raw data collected:");

      // Log vocabulary scores
      if (aggregated.speechMetrics.vocabularyScores.length > 0) {
        console.log(
          `- Found ${
            aggregated.speechMetrics.vocabularyScores.length
          } vocabulary scores, average: ${finalAnalytics.averageVocabularyScore?.toFixed(
            2
          )}%`
        );
      } else {
        console.log("- No vocabulary scores found in any recordings");
      }

      // Log logical flow scores
      if (aggregated.speechMetrics.logicalFlowScores.length > 0) {
        console.log(
          `- Found ${
            aggregated.speechMetrics.logicalFlowScores.length
          } logical flow scores, average: ${finalAnalytics.averageLogicalFlow?.toFixed(
            2
          )}%`
        );
      } else {
        console.log("- No logical flow scores found in any recordings");
      }

      // Log filler word percentages
      if (aggregated.speechMetrics.fillerPercentages.length > 0) {
        console.log(
          `- Found ${
            aggregated.speechMetrics.fillerPercentages.length
          } filler percentages, average: ${finalAnalytics.averageFillerWords?.toFixed(
            2
          )}%`
        );
      } else {
        console.log("- No filler percentages found in any recordings");
      }

      console.log("Final calculated metrics:");
      console.log(
        `- Vocabulary diversity: ${finalAnalytics.averageVocabularyScore?.toFixed(
          2
        )}%`
      );
      console.log(
        `- Logical flow score: ${finalAnalytics.averageLogicalFlow?.toFixed(
          2
        )}%`
      );
      console.log(
        `- Filler words percentage: ${finalAnalytics.averageFillerWords?.toFixed(
          2
        )}%`
      );

      setAnalytics(finalAnalytics);
    } catch (error) {
      console.error("Error calculating analytics:", error);
    }
  }, []);

  return analytics;
}
