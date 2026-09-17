import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Express server (server/index.ts) mounts Vite in middleware mode in dev and
// serves dist/ in production, so there is one process and one port (5173).
export default defineConfig({
  plugins: [react()],
  appType: "custom",
  build: { outDir: "dist", emptyOutDir: true },
});
