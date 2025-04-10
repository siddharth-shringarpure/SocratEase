"use client";

/**
 * @fileoverview Root layout component
 */

import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";
import { NavigationBar } from "@/components/custom/NavigationBar";
import { Footer } from "@/components/custom/Footer";
import { useEffect } from "react";
import { DEVICE_ID_KEY, getOrCreateDeviceId } from "@/lib/deviceId";

const inter = Inter({ subsets: ["latin"] });

/**
 * Root layout component that provides common structure and styling
 * @param {Object} props - Component properties
 * @param {React.ReactNode} props.children - Child components to render within layout
 * @returns {JSX.Element} The wrapped application layout
 */
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>): JSX.Element {
  // Generate a device ID on component mount if one doesn't exist
  useEffect(() => {
    if (typeof window === "undefined") {
      return; // skip during SSR
    }

    const deviceId = getOrCreateDeviceId();
    console.log("Device ID:", deviceId); // TODO: Remove console.log
  }, []);

  return (
    <html lang="en" className="dark">
      <body
        className={cn(
          inter.className,
          "min-h-screen bg-background flex flex-col"
        )}
      >
        <NavigationBar />
        <main className="flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
