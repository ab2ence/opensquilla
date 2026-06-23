import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

const gatewayTarget = process.env.OPENSQUILLA_GATEWAY_DEV_TARGET || 'http://127.0.0.1:18791'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  base: './',
  server: {
    proxy: {
      '/ws': {
        target: gatewayTarget,
        ws: true,
      },
      '/artifacts': gatewayTarget,
      '/attachments': gatewayTarget,
      '/uploads': gatewayTarget,
      '/api': gatewayTarget,
    },
  },
  build: {
    outDir: resolve(__dirname, '../src/opensquilla/gateway/static/dist'),
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      output: {
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name?.split('.') || []
          const ext = info[info.length - 1]
          if (/\.(png|jpe?g|gif|svg|webp|ico)$/i.test(assetInfo.name || '')) {
            return `assets/img/[name]-[hash][extname]`
          }
          if (/\.(woff2?|ttf|otf|eot)$/i.test(assetInfo.name || '')) {
            return `assets/fonts/[name]-[hash][extname]`
          }
          return `assets/[name]-[hash][extname]`
        },
        chunkFileNames: 'assets/[name]-[hash].js',
        entryFileNames: 'assets/[name]-[hash].js',
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
})
