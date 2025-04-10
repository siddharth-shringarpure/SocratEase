"use client";

/**
 * @fileoverview Handles video feed display and canvas overlay for practice sessions.
 * Provides real-time camera feed with optional visual feedback through canvas.
 */

import { RefObject } from "react";
import { motion } from "framer-motion";

interface CameraFeedProps {
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
  isVideoOn: boolean;
}

/**
 * Formats seconds into MM:SS display format
 * @param {number} seconds - Total seconds to format
 * @returns {string} Formatted time string
 */
const formatDuration = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins.toString().padStart(2, "0")}:${secs
    .toString()
    .padStart(2, "0")}`;
};

/**
 * Renders camera feed with canvas overlay for visual feedback
 * @param {CameraFeedProps} props - Component configuration
 * @returns {JSX.Element} Video feed display
 */
export function CameraFeed({
  videoRef,
  canvasRef,
  isVideoOn,
}: CameraFeedProps): JSX.Element {
  // TODO: Add loading state while camera initialises
  // TODO: Consider adding error boundary for camera failures

  return (
    <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
      {/* Main video feed display */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute top-0 left-0 w-full h-full object-contain"
      />

      {/* Canvas overlay */}
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full"
        style={{
          pointerEvents: "none",
          zIndex: 10,
        }}
      />
    </div>
  );
}
