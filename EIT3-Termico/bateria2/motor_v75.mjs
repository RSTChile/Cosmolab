// Motor de v7.5.html. La física NO cambió respecto a v7.4.1 (confirmado leyendo
// pasoFisica/evolveField/computeDaisyworld/etc. en el archivo nuevo — son
// idénticas línea a línea), así que esto EXTIENDE Motor de motor2.mjs en vez de
// duplicar la física, agregando solo lo nuevo de v7.5: medirRecuperacion()
// (ralentización crítica) y varianzaYAutocorr().
import { Motor, clamp } from './motor2.mjs';

export class MotorV75 extends Motor {
  // Réplica exacta de medirRecuperacion(golpe,tope) del HTML (línea ~161):
  // golpea black/white, cuenta pasos hasta que |black-black_antes|<=golpe*0.2
  // o se acaba el tope, 5 repeticiones con 50 pasos de por medio entre ellas.
  medirRecuperacion(golpe, tope) {
    const REPS = 5, UMBRAL = 0.2;
    let suma = 0, fallos = 0;
    for (let r = 0; r < REPS; r++) {
      const bB = this.state.black, bW = this.state.white;
      this.state.black = clamp(bB + golpe, 0, 0.9);
      this.state.white = clamp(bW - golpe * 0.5, 0, 0.9);
      let i = 1;
      for (; i <= tope; i++) {
        this.paso();
        if (Math.abs(this.state.black - bB) <= golpe * UMBRAL) break;
      }
      if (i > tope) fallos++;
      suma += Math.min(i, tope);
      for (let s = 0; s < 50; s++) this.paso();
    }
    return { pasos: suma / REPS, convergio: fallos === 0 ? 1 : 0 };
  }

  // Réplica exacta de varianzaYAutocorr(xs0) del HTML (línea ~187): quita
  // tendencia lineal (mínimos cuadrados) y calcula varianza + autocorrelación
  // de retardo 1 sobre la serie sin tendencia.
  varianzaYAutocorr(xs0) {
    const n = xs0.length;
    if (n < 3) return { varianza: 0, autocorr1: 0 };
    let sx = 0, sy = 0, sxy = 0, sxx = 0;
    for (let i = 0; i < n; i++) { sx += i; sy += xs0[i]; sxy += i * xs0[i]; sxx += i * i; }
    const den = n * sxx - sx * sx;
    const b = den !== 0 ? (n * sxy - sx * sy) / den : 0;
    const a = sy / n - b * sx / n;
    const xs = xs0.map((y, i) => y - (a + b * i));
    let m = 0; for (const x of xs) m += x; m /= n;
    let s2 = 0; for (const x of xs) s2 += (x - m) * (x - m); s2 /= n;
    if (s2 <= 0) return { varianza: 0, autocorr1: 0 };
    let c = 0; for (let i = 1; i < n; i++) c += (xs[i] - m) * (xs[i - 1] - m);
    return { varianza: s2, autocorr1: (c / (n - 1)) / s2 };
  }
}
