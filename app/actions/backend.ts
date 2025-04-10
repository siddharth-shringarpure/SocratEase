"use server";

export interface BackendStatus {
  isConnected: boolean;
  details?: {
    face_detection: boolean;
    emotion_model: boolean;
    version: string;
  };
  error?: string;
}

export async function checkBackendStatus(): Promise<BackendStatus> {
  try {
    const response = await fetch("http://localhost:5000/api/test", {
      method: "GET",
      cache: "no-store",
      headers: {
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      return {
        isConnected: false,
        error: "Backend service is not responding",
      };
    }

    const data = await response.json();
    return {
      isConnected: true,
      details: data.details,
    };
  } catch (error) {
    return {
      isConnected: false,
      error: "Cannot connect to Python server",
    };
  }
}
