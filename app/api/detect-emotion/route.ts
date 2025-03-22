import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import fs from "fs";
import path from "path";
// import { v4 as uuidv4 } from "uuid";

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();

    if (!data.image) {
      return NextResponse.json(
        { success: false, error: "No image data provided" },
        { status: 400 }
      );
    }

    // Generate random emotions for demo purposes
    // In a real app, you'd call your Python script here
    const emotions = {
      angry: Math.random() * 0.2,
      disgust: Math.random() * 0.1,
      fear: Math.random() * 0.1,
      happy: Math.random() * 0.5,
      sad: Math.random() * 0.2,
      surprise: Math.random() * 0.2,
      neutral: Math.random() * 0.4,
    };

    return NextResponse.json({
      success: true,
      face_detected: true,
      emotions,
    });
  } catch (error) {
    console.error("Error in emotion detection:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
