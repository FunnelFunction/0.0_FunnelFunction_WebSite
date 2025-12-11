import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/0.0_FunnelFunction_WebSite/simulators/campaign-commander/',
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  resolve: {
    alias: {
      '@': '/5.1.1_src',
    },
  },
});
