"use client";

/**
 * @fileoverview Displays a list of user recordings and provides navigation
 * to individual recording details.
 */

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { format } from "date-fns";
import { motion } from "framer-motion";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Calendar, AlertCircle, ArrowLeft, ChevronRight } from "lucide-react";
import { getDeviceIdHashDigest, validateDeviceIdHash } from "@/lib/deviceId";

// TODO: Add proper error handling for localStorage access failures
// TODO: Consider adding pagination for large numbers of recordings
// TODO: Add ability to delete recordings

interface Recording {
  id: string;
  filename: string;
  timestamp: string;
  duration: number;
  category: string | null;
  formattedDate: string;
  formattedDuration: string;
}

/**
 * Displays a list of user recordings
 * @returns {JSX.Element} The recordings page component
 */
export default function RecordingsPage(): JSX.Element {
  const router = useRouter();
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadRecordings = async () => {
      try {
        const deviceIdHash = getDeviceIdHashDigest();
        const allKeys = Object.keys(localStorage);

        // Filter for analysis entries that match the format with the correct hash digest
        const recordingKeys = allKeys.filter((key) => {
          const matchesFormat = /^analysis_[a-f0-9]{8}_\d{8}T\d{6}$/.test(key);
          if (!matchesFormat) return false;

          const keyHash = key.split("_")[1];
          return validateDeviceIdHash(keyHash);
        });

        const recordingsData: Recording[] = [];

        // Process each recording entry
        for (const key of recordingKeys) {
          try {
            const filename = key.replace("analysis_", "");
            const analysisData = localStorage.getItem(key);
            if (!analysisData) continue;

            const parsedData = JSON.parse(analysisData);
            const date = parsedData.timestamp
              ? new Date(parsedData.timestamp)
              : new Date();

            // Format duration into minutes:seconds
            const duration = parsedData.duration || 0;
            const mins = Math.floor(duration / 60);
            const secs = duration % 60;

            recordingsData.push({
              id: filename,
              filename,
              timestamp: parsedData.timestamp || "",
              duration,
              category: parsedData.category || null,
              formattedDate: format(date, "dd MMM yyyy, HH:mm"),
              formattedDuration: `${mins}:${secs.toString().padStart(2, "0")}`,
            });
          } catch (err) {
            console.error(`Failed to process recording ${key}:`, err); // Log but continue processing others
          }
        }

        // Sort by newest first
        recordingsData.sort((a, b) => {
          return (
            new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
          );
        });

        setRecordings(recordingsData);
      } catch (err) {
        console.error("Error loading recordings:", err);
        setError(
          "Failed to load your recordings. Please refresh and try again."
        );
      } finally {
        setIsLoading(false);
      }
    };

    loadRecordings();
  }, []);

  const formatCategory = (category: string | null): string => {
    if (!category) return "General";
    return category
      .split("-")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  // animation variants for container and items
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 24,
      },
    },
  };

  return (
    <div className="container mx-auto py-6 px-4 mt-12 max-w-3xl">
      <div className="flex items-center mb-8">
        <h1 className="text-3xl font-bold">Your Recordings</h1>
      </div>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-destructive/15 text-destructive p-4 rounded-md mb-6 flex items-center"
        >
          <AlertCircle className="h-5 w-5 mr-2" />
          {error}
        </motion.div>
      )}

      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <Card key={i} className="overflow-hidden">
              <CardHeader className="pb-3 pt-4">
                <div className="flex justify-between items-start">
                  <div className="space-y-2 w-full">
                    <Skeleton className="h-6 w-3/4" />
                    <Skeleton className="h-4 w-1/2" />
                  </div>
                  <Skeleton className="h-6 w-16" />
                </div>
              </CardHeader>
            </Card>
          ))}
        </div>
      ) : recordings.length > 0 ? (
        <motion.div
          className="space-y-3"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {recordings.map((recording) => (
            <motion.div key={recording.id} variants={itemVariants}>
              <Link
                href={`/recordings/${recording.filename}`}
                className="block"
              >
                <Card className="hover:bg-accent/50 transition-colors">
                  <CardHeader className="pb-3 pt-4">
                    <div className="flex justify-between items-center">
                      <div>
                        <CardTitle className="text-xl">
                          {formatCategory(recording.category)}
                        </CardTitle>
                        <CardDescription className="flex items-center mt-1">
                          <Calendar className="h-4 w-4 mr-1" />
                          {recording.formattedDate}
                        </CardDescription>
                      </div>
                      <div className="flex items-center">
                        <Badge variant="outline" className="mr-2">
                          {recording.formattedDuration}
                        </Badge>
                        <ChevronRight className="h-5 w-5 text-muted-foreground" />
                      </div>
                    </div>
                  </CardHeader>
                </Card>
              </Link>
            </motion.div>
          ))}
        </motion.div>
      ) : (
        <motion.div
          className="text-center py-12"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="text-2xl font-semibold mb-2">No Recordings Yet</h2>
          <p className="text-muted-foreground mb-6">
            You haven't made any recordings yet. Start practicing to see your
            recordings here.
          </p>
          <Button onClick={() => router.push("/practice")}>
            Start Practicing
          </Button>
        </motion.div>
      )}
    </div>
  );
}
