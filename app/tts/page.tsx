"use client";

/**
 * @fileoverview Developer testing page for TTS with Neuphonic.
 */

import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import Link from "next/link";
import { Loader2, AlertCircle } from "lucide-react";

const MAX_TEXT_LENGTH = 2000;
const DEFAULT_SPEED = 1.0;
const VOICE_OPTIONS = [
  {
    id: "f8698a9e-947a-43cd-a897-57edd4070a78",
    name: "Albert (British Male)",
  },
  {
    id: "79ffd956-872a-4b89-b25b-d99bb4335b82",
    name: "Liz (British Female)",
  },
];

/**
 * Text-to-Speech page component that provides speech synthesis functionality
 * @returns {JSX.Element} The rendered TTS page
 */
export default function TTSPage(): JSX.Element {
  const [text, setText] = useState("");
  const [audio, setAudio] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rawError, setRawError] = useState<string | null>(null);
  const [errorType, setErrorType] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [voice, setVoice] = useState(VOICE_OPTIONS[0].id);
  const [speed, setSpeed] = useState([DEFAULT_SPEED]);
  const requestInProgress = useRef(false);

  // Populate text area with sample content
  const setTestText = () => setText("Test text for TTS");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Prevent duplicate requests
    if (requestInProgress.current) {
      return;
    }

    requestInProgress.current = true;
    setError(null);
    setRawError(null);
    setErrorType(null);
    setIsLoading(true);
    setAudio(null);

    try {
      if (text.length > MAX_TEXT_LENGTH) {
        setError(
          `Text is too long (${text.length} characters). Please limit to ${MAX_TEXT_LENGTH} characters.`
        );
        setIsLoading(false);
        requestInProgress.current = false;
        return;
      }

      console.log(
        "Sending TTS request for text:",
        text.slice(0, 50) + (text.length > 50 ? "..." : "")
      );
      console.log(`TTS input text length: ${text.length} chars.`);

      const ttsResponse = await fetch("/api/tts-core", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "audio/wav",
        },
        body: JSON.stringify({
          text,
          voice,
          speed: speed[0],
        }),
      });

      console.log("Response status:", ttsResponse.status);
      console.log(
        "Response headers:",
        Object.fromEntries(ttsResponse.headers.entries())
      );

      // Check if the response is JSON (error) or audio (success)
      const contentType = ttsResponse.headers.get("content-type");
      if (contentType?.includes("application/json")) {
        const errorData = await ttsResponse.json();
        console.log("Error response data:", errorData);

        // Extract detailed error information
        let errorMessage = errorData.error || "Unknown error";
        const detailedError = errorData.detailed_error || errorMessage;
        const errorData2 = errorData.error_data || {};

        setErrorType(errorData2.reason || null);
        setRawError(detailedError);

        if (errorData2.reason === "api_key_invalid") {
          errorMessage = "The API key appears to be invalid or expired.";
        } else if (errorData2.reason === "quota_exceeded") {
          errorMessage = "The API usage quota has been exceeded.";
        } else if (errorData2.reason === "server_error") {
          errorMessage = detailedError;
        }

        setError(errorMessage);
        throw new Error(errorMessage);
      }

      if (!ttsResponse.ok) {
        if (ttsResponse.status === 401) {
          setErrorType("api_key_invalid");
          setError("API key is invalid or expired");
        } else if (ttsResponse.status === 403) {
          setErrorType("quota_exceeded");
          setError("API usage quota has been exceeded");
        } else if (ttsResponse.status === 502 || ttsResponse.status === 504) {
          setError("Cannot connect to TTS service");
        } else {
          setError(`TTS error, status: ${ttsResponse.status}`);
        }
        return;
      }

      const audioBlob = await ttsResponse.blob();
      console.log("Received audio blob:", {
        type: audioBlob.type,
        size: audioBlob.size,
      });

      const audio = new Audio();
      audio.src = URL.createObjectURL(audioBlob);

      audio.onerror = (e) => {
        console.error("Audio error:", e);
        setError("Error loading audio");
      };

      audio.onloadedmetadata = () => {
        console.log("Audio metadata loaded:", {
          duration: audio.duration,
          readyState: audio.readyState,
        });
      };

      setAudio(audio.src);
    } catch (err) {
      let errorMessage = "An error occurred";
      if (err instanceof Error) {
        errorMessage = err.message;
      } else if (err instanceof Response) {
        try {
          const text = await err.text();
          errorMessage = text || "Failed to connect to TTS service";
        } catch {
          errorMessage = "Failed to connect to TTS service";
        }
      }
      console.error("Error details:", err);
      setError(errorMessage);
    } finally {
      setIsLoading(false);
      setTimeout(() => {
        requestInProgress.current = false;
      }, 500);
    }
  };

  useEffect(() => {
    return () => {
      if (audio) {
        URL.revokeObjectURL(audio);
      }
    };
  }, [audio]);

  const getErrorHelpText = () => {
    switch (errorType) {
      case "api_key_invalid":
        return "The Neuphonic API key appears to be invalid or has expired. Please contact an administrator to update the API key.";
      case "quota_exceeded":
        return "The Neuphonic API usage quota has been exceeded. Please try again later or contact an administrator.";
      case "server_error":
        return "The speech synthesis server encountered an error. This might be temporary; please try again with shorter text.";
      default:
        return null;
    }
  };

  return (
    <main className="container mx-auto p-8">
      <h1 className="text-4xl font-bold mb-8 text-center">
        Text-to-Speech Testing
      </h1>

      <div className="max-w-2xl mx-auto">
        <Card>
          <CardHeader>
            <CardTitle>Generate Speech</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <label className="text-sm font-medium">Voice Selection</label>
                <Select value={voice} onValueChange={setVoice}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select voice" />
                  </SelectTrigger>
                  <SelectContent>
                    {VOICE_OPTIONS.map((v) => (
                      <SelectItem key={v.id} value={v.id}>
                        {v.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Speech Speed</label>
                <Slider
                  value={speed}
                  onValueChange={setSpeed}
                  min={0.7}
                  max={2.0}
                  step={0.1}
                />
                <div className="text-sm text-muted-foreground text-center">
                  {speed[0].toFixed(1)}x
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium">Text Input</label>
                  <span className="text-xs text-muted-foreground">
                    {text.length} / 2000 characters
                  </span>
                </div>
                <Textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter text to convert to speech..."
                  className="h-32"
                  maxLength={2000}
                />
              </div>

              <div className="flex gap-2">
                <Button
                  type="submit"
                  className="flex-1"
                  disabled={isLoading || !text.trim()}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    "Generate Speech"
                  )}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={setTestText}
                  className="whitespace-nowrap"
                >
                  Use Test Text
                </Button>
              </div>

              {error && (
                <div className="text-red-500 text-sm mt-2 p-3 bg-red-50 dark:bg-red-950/20 rounded-md">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-semibold">Error:</p>

                      {errorType === "api_key_invalid" ||
                      errorType === "quota_exceeded" ||
                      errorType === "server_error" ? (
                        <div>
                          <p>
                            There was an issue with the text-to-speech service.
                          </p>
                          <p className="mt-2">
                            Please contact the developers to let them know about
                            this issue.
                          </p>

                          {errorType === "quota_exceeded" && (
                            <p className="mt-1 text-sm">
                              (It appears the service usage limit has been
                              reached)
                            </p>
                          )}
                        </div>
                      ) : (
                        <p>{error}</p>
                      )}

                      {error === "Cannot connect to TTS service" && (
                        <p className="mt-2">
                          Please check the status of the backend on the{" "}
                          <Link
                            href="/"
                            className="text-primary hover:underline"
                          >
                            home page
                          </Link>
                          .
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {audio && (
                <div className="mt-4">
                  <audio controls src={audio} className="w-full">
                    Your browser does not support the audio element.
                  </audio>
                </div>
              )}
            </form>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
