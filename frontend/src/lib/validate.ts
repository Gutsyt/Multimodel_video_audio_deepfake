import type { Capabilities, MediaKind } from "../api";
import { formatBytes } from "./format";

export interface FileCheck { kind: MediaKind | null; errors: string[] }

const FALLBACK: Record<MediaKind, string[]> = {
  video: [".mp4", ".mov", ".mkv", ".webm", ".avi"],
  audio: [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"],
  image: [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
};

/** Accepted extensions per kind, from the backend's capabilities (never hardcoded when it answered). */
export function acceptedExtensions(caps: Capabilities | null): Record<MediaKind, string[]> {
  const a = caps?.limits.accepted;
  return {
    video: a?.video?.extensions ?? FALLBACK.video,
    audio: a?.audio?.extensions ?? FALLBACK.audio,
    image: a?.image?.extensions ?? FALLBACK.image,
  };
}

export function detectKind(name: string, accepted: Record<MediaKind, string[]>): MediaKind | null {
  const dot = name.lastIndexOf(".");
  const ext = dot < 0 ? "" : name.slice(dot).toLowerCase();
  for (const kind of ["video", "audio", "image"] as const) if (accepted[kind].includes(ext)) return kind;
  return null;
}

/** Cheap checks before upload. The server still validates by content; this only saves a round trip. */
export function validateFile(file: { name: string; size: number }, caps: Capabilities | null): FileCheck {
  const accepted = acceptedExtensions(caps);
  const errors: string[] = [];
  if (file.size === 0) errors.push("The file is empty.");
  const kind = detectKind(file.name, accepted);
  if (!kind) {
    errors.push(`Unsupported file type. Accepted: ${Object.values(accepted).flat().join(" ")}`);
  }
  const max = caps?.limits.max_upload_mb;
  if (max != null && file.size > max * 1e6) errors.push(`The file is ${formatBytes(file.size)}; the server accepts up to ${max} MB.`);
  return { kind, errors };
}

export const acceptAttr = (caps: Capabilities | null): string => Object.values(acceptedExtensions(caps)).flat().join(",");
