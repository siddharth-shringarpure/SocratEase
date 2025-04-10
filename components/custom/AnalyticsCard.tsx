/**
 * @fileoverview Displays analytics data in a card format with trend indicators and styling.
 * Provides visual feedback about performance metrics through colour-coded badges.
 */

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface AnalyticsCardProps {
  title: string; // Card header text
  icon: string; // Emoji or icon character
  value: string; // Main metric value
  description: string; // Explanatory text
  trend?: "up" | "down" | "neutral"; // Performance trend indicator
}

// Styling configurations for different trend states
const TREND_STYLES = {
  up: {
    label: "Good",
    bgColour: "bg-green-100 dark:bg-green-900/30",
    textColour: "text-green-700 dark:text-green-400",
    borderColour: "border-green-200 dark:border-green-800",
  },
  down: {
    label: "Needs Work",
    bgColour: "bg-amber-100 dark:bg-amber-900/30",
    textColour: "text-amber-700 dark:text-amber-400",
    borderColour: "border-amber-200 dark:border-amber-800",
  },
  neutral: {
    label: "Neutral",
    bgColour: "bg-blue-100 dark:bg-blue-900/30",
    textColour: "text-blue-700 dark:text-blue-400",
    borderColour: "border-blue-200 dark:border-blue-800",
  },
};

/**
 * Renders an analytics card with trend indicators and styling
 * @param {AnalyticsCardProps} props - Component configuration options
 * @returns {JSX.Element} Styled analytics card
 */
export function AnalyticsCard({
  title,
  icon,
  value,
  description,
  trend = "neutral",
}: AnalyticsCardProps): JSX.Element {
  // TODO: Consider adding animations for value changes
  const styles = TREND_STYLES[trend];

  return (
    <Card className="shadow-md hover:shadow-lg transition-shadow overflow-hidden h-full">
      <CardHeader className="bg-gradient-to-r from-primary/10 to-primary/5 rounded-t-lg pb-4">
        <CardTitle className="text-center flex items-center justify-center gap-2 text-xl">
          <span>{title}</span>
          <span className="text-2xl">{icon}</span>
        </CardTitle>
      </CardHeader>

      <CardContent className="text-center pt-6 pb-6 relative flex flex-col h-[160px]">
        <div className="flex-1 flex flex-col justify-center items-center">
          {/* primary metric display */}
          <div className="text-3xl font-bold">{value}</div>

          {/* Performance indicator - only shown for non-neutral trends */}
          {/* todo: rethink this in general */}
          {trend !== "neutral" && (
            <div
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium 
                ${styles.bgColour} ${styles.textColour} border ${styles.borderColour} mt-2 mx-auto`}
            >
              <span>{styles.label}</span>
            </div>
          )}
        </div>

        <p className="text-muted-foreground mt-auto">{description}</p>
      </CardContent>
    </Card>
  );
}
