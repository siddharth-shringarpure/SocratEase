"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export default function RecordingPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [transcription, setTranscription] = useState<string | null>(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const filename = params.filename as string;
  const audioFilename = searchParams.get('audio');
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Function to get transcription
  const getTranscription = async () => {
    if (!audioFilename) return;
    
    try {
      setIsTranscribing(true);
      setError(null);
      
      // Fetch the audio file
      console.log('Fetching audio file:', audioFilename);
      const audioResponse = await fetch(`http://localhost:5328/uploads/${audioFilename}`);
      if (!audioResponse.ok) {
        throw new Error(`Failed to fetch audio file: ${audioResponse.statusText}`);
      }
      
      const audioBlob = await audioResponse.blob();
      console.log('Audio blob size:', audioBlob.size, 'bytes');
      console.log('Audio blob type:', audioBlob.type);
      
      // Send to transcription endpoint
      const formData = new FormData();
      formData.append('file', audioBlob, audioFilename);  // Use original filename to preserve extension
      
      console.log('Sending transcription request...');
      const response = await fetch('http://localhost:5328/api/speech2text', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('Transcription error response:', errorData);
        throw new Error(errorData.error || `Transcription failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('Transcription result:', result);
      
      if (!result.text) {
        throw new Error('No transcription received from server');
      }
      setTranscription(result.text);
    } catch (error) {
      console.error('Error getting transcription:', error);
      setError(error instanceof Error ? error.message : 'Failed to get transcription');
      setTranscription(null);
    } finally {
      setIsTranscribing(false);
    }
  };

  // Get transcription when audio file is available
  useEffect(() => {
    if (audioFilename) {
      console.log('Audio file available:', audioFilename);
      getTranscription();
    }
  }, [audioFilename]);

  return (
    <main className="container py-8 flex flex-col items-center">
      <h1 className="text-4xl font-bold mb-8 text-center">Recorded Video</h1>

      <Card className="w-full max-w-2xl mb-8">
        <CardHeader>
          <CardTitle className="text-center">Recording: {filename}</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-4">
          <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              src={`http://localhost:5328/uploads/${filename}`}
              controls
              className="w-full h-full object-contain"
            />
          </div>

          {audioFilename && (
            <div className="w-full">
              <h3 className="font-semibold mb-2">Audio Track:</h3>
              <audio
                ref={audioRef}
                src={`http://localhost:5328/uploads/${audioFilename}`}
                className="w-full"
                controls
              />
              <p className="text-sm text-muted-foreground mt-2">
                Note: Video and audio can be controlled independently
              </p>
            </div>
          )}

          <div className="flex gap-4">
            <Button
              onClick={() => router.push("/camera")}
              variant="outline"
            >
              Back to Camera
            </Button>
          </div>
        </CardContent>
      </Card>

      {audioFilename && (
        <Card className="w-full max-w-2xl">
          <CardHeader>
            <CardTitle className="text-center">Transcription</CardTitle>
          </CardHeader>
          <CardContent>
            {isTranscribing ? (
              <div className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-[90%]" />
                <Skeleton className="h-4 w-[95%]" />
                <Skeleton className="h-4 w-[85%]" />
              </div>
            ) : transcription ? (
              <p className="whitespace-pre-wrap">{transcription}</p>
            ) : error ? (
              <div className="text-center">
                <p className="text-destructive mb-2">{error}</p>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => getTranscription()}
                >
                  Retry Transcription
                </Button>
              </div>
            ) : null}
          </CardContent>
        </Card>
      )}
    </main>
  );
} 