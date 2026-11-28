import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

const local = path => fileURLToPath(new URL(path, import.meta.url));

export default defineConfig({
  base: './',
  server: { fs: { allow: [local('.'), local('../resources')] } },
  build: { rollupOptions: { input: { home: local('index.html'), tasks: local('tasks.html') } } }
});
