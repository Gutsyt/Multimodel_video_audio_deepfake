import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the Vite server proxies /api to the FastAPI backend (no CORS needed).
// In production the backend serves the built files from the same origin.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist" },
  test: { environment: "jsdom", globals: true, setupFiles: ["./src/setupTests.ts"] },
});
