"use client";

/**
 * @fileoverview Manages the practice session interface, including camera controls,
 * recording functionality, and status feedback. Provides real-time interaction
 * for users during their practice sessions.
 */

import { motion } from "framer-motion";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ConversationMode } from "@/types/practice";
import { CameraFeed } from "./CameraFeed";
import { RecordingControls } from "./RecordingControls";
import { BackendStatus } from "@/components/custom/BackendStatus";
import { PracticeTips } from "./PracticeTips";

// Animation configuration for smooth transitions
const FADE_ANIMATION = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

interface PracticeSessionProps {
  selectedModeData: ConversationMode;
  isStreaming: boolean;
  isRecording: boolean;
  isCheckingBackend: boolean;
  isUploading: boolean;
  isBackendConnected: boolean;
  backendError: string | null;
  recordingError: string | null;
  recordingDuration: number;
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  onStartCamera: () => void;
  onStopCamera: () => void;
  onStartRecording: () => void;
  onStopRecording: () => void;
}

/**
 * Renders the practice session interface with camera controls and feedback
 * @param {PracticeSessionProps} props - Component configuration options
 * @returns {JSX.Element} Practice session interface
 */
export function PracticeSession({
  selectedModeData,
  isStreaming,
  isRecording,
  isCheckingBackend,
  isUploading,
  isBackendConnected,
  backendError,
  recordingError,
  recordingDuration,
  videoRef,
  canvasRef,
  onStartCamera,
  onStopCamera,
  onStartRecording,
  onStopRecording,
}: PracticeSessionProps): JSX.Element {
  // TODO: Add recording duration display
  // TODO: Add a length limit to the recording

  return (
    <motion.div
      key="practice-session"
      {...FADE_ANIMATION}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      <Card className="border-2">
        <CardHeader>
          <CardTitle className="text-center flex items-center justify-center gap-2">
            <span>{selectedModeData.name}</span>
            <span className="text-2xl">{selectedModeData.emoji}</span>
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Camera feed display */}
          <CameraFeed
            isVideoOn={isStreaming}
            videoRef={videoRef}
            canvasRef={canvasRef}
          />

          <div className="flex flex-col items-center gap-4">
            {/* Recording and camera controls */}
            <RecordingControls
              isRecording={isRecording}
              isVideoOn={isStreaming}
              isBackendConnected={isBackendConnected}
              onToggleRecording={
                isRecording ? onStopRecording : onStartRecording
              }
              onToggleVideo={isStreaming ? onStopCamera : onStartCamera}
            />

            {/* Backend connection status */}
            <div className="flex justify-center w-full">
              {isCheckingBackend ? (
                <div className="text-sm text-muted-foreground">
                  Checking backend connection...
                </div>
              ) : isStreaming ? (
                <BackendStatus
                  isConnected={isBackendConnected}
                  isChecking={isCheckingBackend}
                  onRetry={onStartCamera}
                />
              ) : null}
            </div>
          </div>

          {/* Error display */}
          {recordingError && (
            <div className="w-full p-4 bg-destructive/10 text-destructive rounded-lg text-center">
              {recordingError}
            </div>
          )}

          {/* Upload overlay */}
          {isUploading && (
            <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
              <div className="bg-background p-8 rounded-lg shadow-lg flex flex-col items-center gap-4">
                <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
                <p className="text-lg font-semibold">
                  Processing your recording...
                </p>
                <p className="text-sm text-muted-foreground">
                  Please wait while we prepare your video
                </p>
              </div>
            </div>
          )}

          <PracticeTips selectedModeData={selectedModeData} />
        </CardContent>
      </Card>
    </motion.div>
  );
}
