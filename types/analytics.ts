export interface EmotionEntry {
  emotion: string;
  percentage: number;
  count?: number;
}

export interface GazeEntry {
  direction: string;
  percentage: number;
  count?: number;
}

export interface SpeechAnalysis {
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

export interface AnalyticsAccumulator {
  totalSessions: number;
  totalDuration: number;
  averageEmotions: EmotionEntry[];
  averageGazeDirections: GazeEntry[];
  totalGazeEntries: number;
  speechMetrics: {
    vocabularyScores: number[];
    logicalFlowScores: number[];
    fillerPercentages: number[];
  };
}

export interface AggregatedAnalytics {
  totalSessions: number;
  totalDuration: number;
  averageEmotions: Array<{
    emotion: string;
    percentage: number;
  }>;
  averageGazeDirections: Array<{
    direction: string;
    percentage: number;
  }>;
  averageFillerWords: number | null;
  averageVocabularyScore: number | null;
  averageLogicalFlow: number | null;
}
