"use client";

import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5328";

export default function EyeTracking() {
  const imgRef = useRef<HTMLImageElement>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Request camera permission
    async function requestCameraPermission() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        // Stop the stream right away since we don't need it (backend handles camera)
        stream.getTracks().forEach((track) => track.stop());
        setHasPermission(true);

        // Start the video feed
        if (imgRef.current) {
          imgRef.current.src = `${API_URL}/api/video_feed?t=${Date.now()}`;
        }
      } catch (err) {
        console.error("Camera permission error:", err);
        setHasPermission(false);
        setError(
          "Camera access denied. Please enable camera permissions to use eye tracking."
        );
      }
    }

    requestCameraPermission();

    // Refresh feed periodically
    let interval: NodeJS.Timeout;
    if (hasPermission) {
      interval = setInterval(() => {
        if (imgRef.current) {
          imgRef.current.src = `${API_URL}/api/video_feed?t=${Date.now()}`;
        }
      }, 5000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [hasPermission]);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Eye Tracking Demo</h1>
      <div className="relative aspect-video w-full max-w-2xl mx-auto">
        {hasPermission === null ? (
          <div className="flex items-center justify-center h-full bg-gray-100 rounded-lg">
            <p>Requesting camera permission...</p>
          </div>
        ) : hasPermission === false ? (
          <div className="flex items-center justify-center h-full bg-red-50 rounded-lg p-4">
            <p className="text-red-600 text-center">{error}</p>
          </div>
        ) : (
          <div className="relative">
            <img
              ref={imgRef}
              className="w-full h-full object-contain rounded-lg"
              alt="Eye tracking feed"
              onError={() =>
                setError(
                  "Failed to connect to video feed. Please ensure the backend server is running."
                )
              }
            />
            {error && (
              <div className="absolute inset-0 flex items-center justify-center bg-red-50 bg-opacity-90 rounded-lg">
                <p className="text-red-600 text-center p-4">{error}</p>
              </div>
            )}
          </div>
        )}
      </div>
      {hasPermission && !error && (
        <div className="mt-4 space-y-2">
          <div className="text-center text-sm text-gray-600">
            <p>
              The green dots show facial landmarks, and blue rectangles show
              detected faces.
            </p>
            <p>
              Red arrows indicate gaze direction, and text shows the yaw/pitch
              angles and gaze region.
            </p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h2 className="font-semibold mb-2">Gaze Detection Guide:</h2>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• "center": Looking directly at the camera</li>
              <li>• "up": Looking up</li>
              <li>• "down": Looking down</li>
              <li>• "left": Looking left</li>
              <li>• "right": Looking right</li>
              <li>• Combinations like "up-left", "down-right", etc.</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
