// Worker de cola dinámica (IPC con el proceso padre) — igual patrón que
// bateria2/worker_v75_ipc.mjs, usando el motor de v7.6.1.
import { correrBarridoV76 } from './correr_barrido_v76.mjs';

process.send({ tipo: 'listo' });

process.on('message', (msg) => {
  if (msg.tipo === 'fin') { process.exit(0); }
  if (msg.tipo === 'job') {
    const { job } = msg;
    const t0 = Date.now();
    let rows, error = null;
    try {
      rows = correrBarridoV76({
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
