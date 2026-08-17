// Worker de cola dinámica (IPC con el proceso padre): pide un job, lo corre,
// devuelve las filas, pide el siguiente. Reparto dinámico en vez de sharding
// estático porque el costo por barrido varía mucho (300-520+ s medido) y un
// reparto fijo dejaría procesos ociosos mientras otros cargan con los puntos
// lentos de la grilla.
import { correrBarridoV75 } from './correr_barrido_v75.mjs';

process.send({ tipo: 'listo' });

process.on('message', (msg) => {
  if (msg.tipo === 'fin') { process.exit(0); }
  if (msg.tipo === 'job') {
    const { job } = msg;
    const t0 = Date.now();
    let rows, error = null;
    try {
      rows = correrBarridoV75({
        seed: job.seed, modo: job.modo,
        from: job.from, to: job.to, steps: job.steps,
        settle: job.settle, measure: job.measure, params: job.params,
      });
    } catch (e) {
      error = String(e && e.stack || e);
      rows = [];
    }
    const seg = (Date.now() - t0) / 1000;
    process.send({ tipo: 'resultado', tag: job.tag, seed: job.seed, modo: job.modo, params: job.params, rows, seg, error });
    process.send({ tipo: 'listo' });
  }
});
