// Lista de barridos (jobs) para Experimento D, A' y B' de la tercera batería
// (v7.6.1). Mismo diseño que bateria2/jobs_v75.mjs, eje angostado a
// 0,60->1,40 (el que v7.6.1 declara válido para luminosidad).
export const EJE = { from: 0.60, to: 1.40, steps: 60, settle: 300, measure: 120 };

const BASE = { powerBase: 0.47, beta: 0.94, sigma: 6.8, noise: 0.0079, band: 1.105, tOpt: 25, ptcTc: 18, ptcSharp: 4.1 };

export function buildJobsD() {
  const jobs = [];
  for (const modo of ['parada', 'inicio']) {
    for (let seed = 1; seed <= 10; seed++) {
      jobs.push({ tag: 'D', seed, modo, params: { ...BASE }, ...EJE });
    }
  }
  return jobs;
}

export function buildJobsAprima() {
  const jobs = [];
  for (let seed = 1; seed <= 30; seed++) {
    jobs.push({ tag: 'Aprima', seed, modo: 'parada', params: { ...BASE }, ...EJE });
  }
  return jobs;
}

export function buildJobsBprima() {
  const jobs = [];
  const betas = [0.80, 0.88, 0.94, 0.98];
  const tOpts = [22, 25, 28];
  const ptcSharps = [3.0, 4.1, 6.0];
  const powerBases = [0.30, 0.47, 0.65];
  for (const beta of betas) for (const tOpt of tOpts) for (const ptcSharp of ptcSharps) for (const powerBase of powerBases) {
    const params = { powerBase, beta, sigma: 6.8, noise: 0.0079, band: 1.105, tOpt, ptcTc: 18, ptcSharp };
    for (let seed = 1; seed <= 10; seed++) {
      jobs.push({ tag: 'Bprima', seed, modo: 'parada', params, ...EJE });
    }
  }
  return jobs;
}

export function buildAllJobs() {
  return [...buildJobsD(), ...buildJobsAprima(), ...buildJobsBprima()];
}
