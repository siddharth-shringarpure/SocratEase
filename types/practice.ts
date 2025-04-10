export interface ConversationMode {
  id: string;
  name: string;
  description: string;
  tips: string[];
  emoji: string;
}

export interface PracticeSessionState {
  isRecording: boolean;
  isVideoOn: boolean;
  selectedMode: ConversationMode | null;
  showTips: boolean;
  backendStatus: {
    isConnected: boolean;
    message: string;
  };
}

export interface EmotionData {
  emotion: string;
  score: number;
}

export interface BackendStatusProps {
  status: {
    isConnected: boolean;
    message: string;
  };
}

export interface CameraFeedProps {
  isVideoOn: boolean;
  videoRef: React.RefObject<HTMLVideoElement>;
}

export interface ModeSelectionProps {
  modes: ConversationMode[];
  selectedMode: ConversationMode | null;
  onModeSelect: (mode: ConversationMode) => void;
}

export interface PracticeTipsProps {
  tips: string[];
  showTips: boolean;
  onToggleTips: () => void;
}

export interface RecordingControlsProps {
  isRecording: boolean;
  isVideoOn: boolean;
  onToggleRecording: () => void;
  onToggleVideo: () => void;
}

export interface TipsAndStartProps {
  selectedMode: ConversationMode | null;
  onStart: () => void;
}

export interface Emotions {
  neutral: number;
  happy: number;
  sad: number;
  angry: number;
  fearful: number;
  disgusted: number;
  surprised: number;
}

export interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList;
  timeStamp: number;
}

export interface SpeechRecognitionResultList {
  length: number;
  [index: number]: SpeechRecognitionResult;
}

export interface SpeechRecognitionResult {
  [index: number]: {
    transcript: string;
  };
}

export interface SpeechRecognition {
  continuous: boolean;
  interimResults: boolean;
  onresult: (event: SpeechRecognitionEvent) => void;
  start: () => void;
  stop: () => void;
}
