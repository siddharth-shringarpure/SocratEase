"use client";

/**
 * @fileoverview Provides recording and camera controls for practice sessions.
 * Handles video stream toggling and recording state with tooltips for guidance.
 */

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/components/ui/tooltip";

interface RecordingControlsProps {
  isRecording: boolean;
  isVideoOn: boolean;
  isBackendConnected: boolean;
  onToggleRecording: () => void;
  onToggleVideo: () => void;
}

/**
 * Renders recording and camera control buttons with tooltips
 * @param {RecordingControlsProps} props - Component configuration options
 * @returns {JSX.Element} Recording controls interface
 */
export function RecordingControls({
  isRecording,
  isVideoOn,
  isBackendConnected,
  onToggleRecording,
  onToggleVideo,
}: RecordingControlsProps): JSX.Element {
  // console.log("[RecordingControls] Props received:", {
  //   isRecording,
  //   isVideoOn,
  //   isBackendConnected,
  // });

  return (
    <div className="flex justify-center gap-4">
      {/* Camera toggle only shown when not recording */}
      {!isRecording && (
        <Button
          onClick={onToggleVideo}
          variant={isVideoOn ? "destructive" : "default"}
          className="w-32"
        >
          {isVideoOn ? "Stop Camera" : "Start Camera"}
        </Button>
      )}

      {/* Recording controls only available when camera is on */}
      {isVideoOn && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="w-32">
                <Button
                  onClick={onToggleRecording}
                  variant={isRecording ? "destructive" : "default"}
                  className="w-full"
                  disabled={!isBackendConnected}
                >
                  {isRecording ? "Stop Recording" : "Start Recording"}
                </Button>
              </div>
            </TooltipTrigger>
            {!isBackendConnected && (
              <TooltipContent side="top" align="center" sideOffset={5}>
                <p>Recording is disabled until the backend is connected</p>
              </TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
}
