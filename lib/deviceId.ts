/**
 * @fileoverview Device ID utilities for recording access control
 */

// Import proper hashing from crypto-js
import { SHA256 } from "crypto-js";

// Constants for device ID generation and storage
export const DEVICE_ID_KEY = "socratease_device_id";
export const DEVICE_ID_LENGTH = 32;
export const FILENAME_HASH_LENGTH = 8;

/**
 * Generates a secure random string of specified length
 * @param {number} length - Length of the string to generate
 * @returns {string} A random string of the specified length
 */
export function generateRandomString(length: number): string {
  const chars =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  const charLength = chars.length; // 62 chars
  let result = "";

  const randomValues = new Uint8Array(length);
  crypto.getRandomValues(randomValues);

  for (let i = 0; i < length; i++) {
    result += chars.charAt(randomValues[i] % charLength); // NB: Won't be evenly distributed as 256 % 62 != 0
  }

  return result;
}

/**
 * Creates a hash from a string using SHA-256 from crypto-js
 * @param {string} input - The string to hash
 * @param {number} length - The desired length of the hash
 * @returns {string} A hash of the specified length
 */
export function createHash(input: string, length: number): string {
  // Generate SHA-256 hash
  const hash = SHA256(input).toString();

  // Truncate to desired length, or pad if needed
  if (hash.length <= length) {
    return hash.padEnd(length, "0");
  }

  return hash.substring(0, length);
}

/**
 * Generates a device ID for recording access control
 * @returns {string} A unique device ID
 */
export function generateDeviceId(): string {
  // Combine multiple sources of entropy
  const timestamp = Date.now().toString(36);
  const randomPart = generateRandomString(16);

  // Create a fingerprint from browser info (without collecting PII)
  const userAgentData =
    typeof navigator !== "undefined" ? navigator.userAgent : "";
  const screenData =
    typeof window !== "undefined"
      ? `${window.screen.width}x${window.screen.height}`
      : "";

  // Combine all data
  const rawId = `${timestamp}-${randomPart}-${userAgentData}-${screenData}`;

  // Create a hash
  return createHash(rawId, DEVICE_ID_LENGTH);
}

/**
 * Gets the current device ID from localStorage or generates a new one
 * @returns {string} The device ID
 */
export function getOrCreateDeviceId(): string {
  if (typeof window === "undefined") return "";

  let deviceId =
    localStorage.getItem(DEVICE_ID_KEY) ??
    (() => {
      let newId = generateDeviceId();
      localStorage.setItem(DEVICE_ID_KEY, newId);
      return newId;
    })();

  return deviceId;
}

/**
 * Generates a short hash from the device ID for use in filenames
 * This creates a one-way transformation that can't be easily reversed
 * @returns {string} A short hash derived from the device ID
 */
export function getDeviceIdHashDigest(): string {
  const deviceId = getOrCreateDeviceId();
  return createHash(deviceId, FILENAME_HASH_LENGTH);
}

/**
 * Validates if a given hash matches the current device's hash
 * @param {string} hash - The hash to validate
 * @returns {boolean} Whether the hash is valid for this device
 */
export function validateDeviceIdHash(hash: string): boolean {
  return hash === getDeviceIdHashDigest().substring(0, FILENAME_HASH_LENGTH);
}

/**
 * Generates a key for storing recording metadata in localStorage
 * @param {string} filename - The recording filename
 * @returns {string} The localStorage key for the recording metadata
 */
export function getMetadataKey(filename: string): string {
  // Remove file extensions if present
  const cleanFilename = filename.replace(/\.(mp4|wav)$/, "");
  return `metadata_${cleanFilename}`;
}

/**
 * Generates a key for storing recording analysis in localStorage
 * @param {string} filename - The recording filename
 * @returns {string} The localStorage key for the recording analysis
 */
export function getAnalysisKey(filename: string): string {
  // Remove file extensions if present
  const cleanFilename = filename.replace(/\.(mp4|wav)$/, "");
  return `analysis_${cleanFilename}`;
}

/**
 * Generates a key for storing audio feedback data in localStorage
 * @param {string} filename - The recording filename
 * @returns {string} The localStorage key for the audio feedback data
 */
export function getAudioFeedbackKey(filename: string): string {
  // Remove file extensions if present
  const cleanFilename = filename.replace(/\.(mp4|wav)$/, "");
  return `audio_feedback_data_${cleanFilename}`;
}
