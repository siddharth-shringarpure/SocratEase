"use client";

import { useState, useEffect } from "react";
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

export default function TTSPage() {
  const [text, setText] = useState("");
  const [audio, setAudio] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [voice, setVoice] = useState("f8698a9e-947a-43cd-a897-57edd4070a78");
  const [speed, setSpeed] = useState([1.0]); // Default to normal speed (1.0)

  const voices = [
    {
      id: "f8698a9e-947a-43cd-a897-57edd4070a78",
      name: "Albert (British Male)",
    },
    {
      id: "79ffd956-872a-4b89-b25b-d99bb4335b82",
      name: "Liz (British Female)",
    },
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);
    setAudio(null);

    try {
      console.log("Sending TTS request for text:", text);

      const ttsResponse = await fetch("http://localhost:5328/api/tts", {
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

      if (!ttsResponse.ok) {
        const errorData = await ttsResponse.json();
        throw new Error(
          `TTS error! status: ${ttsResponse.status}, message: ${
            errorData.error || "Unknown error"
          }`
        );
      }

      const audioBlob = await ttsResponse.blob();
      console.log("Received audio blob:", {
        type: audioBlob.type,
        size: audioBlob.size,
        slice: await audioBlob.slice(0, 4).text(),
      });

      // Create and test the audio element programmatically
      const audio = new Audio();
      audio.src = URL.createObjectURL(audioBlob);

      // Listen for errors
      audio.onerror = (e) => {
        console.error("Audio error:", e);
        setError("Error loading audio");
      };

      // Listen for metadata loaded
      audio.onloadedmetadata = () => {
        console.log("Audio metadata loaded:", {
          duration: audio.duration,
          readyState: audio.readyState,
        });
      };

      setAudio(audio.src);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "An error occurred";
      console.error("Error details:", err);
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // Clean up object URL when component unmounts or audio changes
  useEffect(() => {
    return () => {
      if (audio) {
        URL.revokeObjectURL(audio);
      }
    };
  }, [audio]);

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
                    {voices.map((v) => (
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
                <label className="text-sm font-medium">Text Input</label>
                <Textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter text to convert to speech..."
                  className="h-32"
                />
              </div>

              <Button
                type="submit"
                className="w-full"
                disabled={isLoading || !text.trim()}
              >
                {isLoading ? "Generating..." : "Generate Speech"}
              </Button>

              {error && (
                <div className="text-red-500 text-sm mt-2">Error: {error}</div>
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
