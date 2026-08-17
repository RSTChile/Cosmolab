import fs from 'node:fs';

export function readCsv(path) {
  const lines = fs.readFileSync(path, 'utf8').trim().split('\n');
  const header = lines[0].split(',');
  return lines.slice(1).map(line => {
    const vals = line.split(',');
    const row = {};
    header.forEach((h, i) => {
      const v = vals[i];
      row[h] = (v === 'banda<=ruido' || v === 'banda>ruido') ? v : Number(v);
    });
    return row;
  });
}

export function writeCsv(path, rows) {
  if (!rows.length) { fs.writeFileSync(path, ''); return; }
  const header = Object.keys(rows[0]);
  const lines = [header.join(',')];
  for (const r of rows) lines.push(header.map(h => r[h]).join(','));
  fs.writeFileSync(path, lines.join('\n'));
}

export function pearson(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }
  const den = Math.sqrt(dx2 * dy2);
  return den > 0 ? num / den : 0;
}

export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffleWith(rng, arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function mean(a) { return a.reduce((x, y) => x + y, 0) / a.length; }
export function std(a) { const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); }
export function percentileOfValue(value, distribution) {
  const sorted = distribution.slice().sort((a, b) => a - b);
  let count = 0;
  for (const v of sorted) if (v <= value) count++;
  return (100 * count) / sorted.length;
}
