"use client";

/**
 * @fileoverview Monitors and displays backend connectivity status with retry functionality.
 * Provides real-time feedback about system health and core service availability.
 */

import { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";
import { checkBackendStatus } from "@/app/actions/backend";
import type { BackendStatus as BackendStatusType } from "@/app/actions/backend";

interface BackendStatusProps {
  isConnected?: boolean; // External control of connection state
  isChecking?: boolean; // Whether status check is in progress
  onRetry?: () => void; // Optional external retry handler
}

// Configuration for different connection states
const STATUS_CONFIG = {
  checking: {
    colour: "bg-amber-400 animate-pulse",
    message: "Checking connection...",
  },
  online: {
    colour: "bg-green-400",
    message: "Connected",
  },
  partial: {
    colour: "bg-yellow-400",
    message: "Partially Connected",
  },
  offline: {
    colour: "bg-amber-400 animate-pulse",
    message: "Not connected",
  },
};

type ConnectionStatus = keyof typeof STATUS_CONFIG;

/**
 * Displays backend connection status with visual indicators and retry option
 * @param {BackendStatusProps} props - Component configuration options
 * @returns {JSX.Element} Status indicator with optional retry functionality
 */
export function BackendStatus({
  isConnected: externalIsConnected,
  isChecking: externalIsChecking,
  onRetry: externalOnRetry,
}: BackendStatusProps): JSX.Element {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("checking");
  const [isRetrying, setIsRetrying] = useState<boolean>(false);
  const [details, setDetails] = useState<BackendStatusType["details"] | null>(
    null
  );

  // TODO: Consider adding error message state for more detailed feedback

  const isControlled = externalIsConnected !== undefined;

  // Attempts to connect to the backend
  const checkConnection = async (): Promise<void> => {
    if (isRetrying) return;
    setIsRetrying(true);

    try {
      const status = await checkBackendStatus();

      if (status.isConnected) {
        setDetails(status.details);

        // Check if core services are available
        const hasAllServices =
          status.details?.face_detection && status.details?.emotion_model;
        setConnectionStatus(hasAllServices ? "online" : "partial");
      } else {
        setConnectionStatus("offline");
        setDetails(null);
      }
    } catch {
      setConnectionStatus("offline");
      setDetails(null);
    } finally {
      setIsRetrying(false);
    }
  };

  // Check connection on mount and poll every 3s
  useEffect(() => {
    if (!isControlled) {
      checkConnection();
      const pollInterval = setInterval(checkConnection, 3000);
      return () => clearInterval(pollInterval);
    }
  }, [isControlled]);

  useEffect(() => {
    if (isControlled) {
      setConnectionStatus(
        externalIsChecking
          ? "checking"
          : externalIsConnected
          ? "online"
          : "offline"
      );
    }
  }, [isControlled, externalIsConnected, externalIsChecking]);

  const handleRetry = () => {
    externalOnRetry ? externalOnRetry() : checkConnection();
  };

  return (
    <div className="inline-flex flex-col items-center gap-1.5 select-none">
      <div className="flex items-center gap-3 rounded-full bg-background/50 px-3 py-1.5 shadow-sm border border-border/50">
        <div className="flex items-center gap-2">
          <div
            className={`h-2.5 w-2.5 rounded-full ${STATUS_CONFIG[connectionStatus].colour}`}
          />
          <span className="text-sm font-medium">
            {STATUS_CONFIG[connectionStatus].message}
          </span>
        </div>

        {(connectionStatus === "offline" || connectionStatus === "partial") && (
          <button
            onClick={handleRetry}
            disabled={isRetrying}
            className="text-sm text-primary hover:text-primary/80 disabled:text-primary/50 flex items-center gap-1"
          >
            {isRetrying ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                <span>Retrying</span>
              </>
            ) : (
              "Retry"
            )}
          </button>
        )}
      </div>

      {connectionStatus === "offline" && (
        <span className="text-xs text-muted-foreground/70">
          Backend may be starting up...
        </span>
      )}
      {connectionStatus === "partial" && details && (
        <span className="text-xs text-muted-foreground/70">
          {!details.face_detection && "Face detection unavailable. "}
          {!details.emotion_model && "Emotion detection unavailable."}
        </span>
      )}
    </div>
  );
}
