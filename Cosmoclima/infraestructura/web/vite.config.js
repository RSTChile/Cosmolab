import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// ★ La versión y el momento de compilación se sellan aquí, no se escriben a
//   mano: un número de versión que hay que acordarse de subir queda desfasado a
//   la segunda semana. Es lo mismo que muestra la barra de App Captura.
const VERSION = '1.0';
const COMPILADO = new Date()
  .toISOString()
  .slice(0, 16)
  .replace('T', ' ');

export default defineConfig({
  plugins: [svelte()],
  define: {
    __VERSION__: JSON.stringify(VERSION),
    __COMPILADO__: JSON.stringify(COMPILADO),
  },
  // `publico` en vez de `public`: el proyecto nombra en castellano.
  publicDir: 'publico',
  build: { target: 'es2022', outDir: 'dist' },
});
