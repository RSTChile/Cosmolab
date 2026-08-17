// ============================================================================
// GLOSARIO DE LA UI — el nombre en castellano al lado de cada sigla
// ============================================================================
// Por qué existe (8-ago-2026): «el organismo tiene tantas variables y siglas que
// simplemente me confundo y no entiendo qué estás midiendo». Las siglas NO se
// renombran (romperían el CSV canónico, los replays y la comparabilidad con
// 131.000 pasos de historia): se les ADOSA el nombre descriptivo.
//
// Este archivo es la MEMBRANA, no el diccionario. El diccionario vive en
// `celula_madre/glosario.py` y llega por `/glosario`. Si el servidor no responde,
// o una sigla no está en el glosario, el nombre ES la sigla: nada se rompe, sólo
// se explica menos.
//
// Expone (globales, disponibles para todas las cajas de /Cajas):
//   GLOS            mapa sigla -> ficha
//   cjFicha(col)    ficha completa, siempre con las seis claves
//   cjNom(col)      nombre descriptivo (o la sigla)
//   cjDef(col)      definición larga (o '')
//   cjFmt(col,v)    valor legible: '23,456 %' / '8,7° (de ±90°)' / '21 voces'
//   cjRowG(col,v)   fila sigla+nombre / valor formateado, para las cajas
//   glosPintar(el)  pinta la pestaña GLOSARIO en `el`
(function () {
  window.GLOS = window.GLOS || {};
  var CARGADO = false;

  var FICHA_VACIA = function (col) {
    return {
      sigla: col, nombre: col, definicion: '', unidad: '', nodo: '',
      rango_min: null, rango_max: null, rango_medido: false, cosa: ['', ''], fraccion: false,
      obs_n: 0, obs_min: null, obs_max: null, obs_texto: '',
      obs_min_fmt: '', obs_max_fmt: ''
    };
  };

  window.cjFicha = function (col) {
    return (window.GLOS && window.GLOS[col]) || FICHA_VACIA(col);
  };
  window.cjNom = function (col) {
    var n = window.cjFicha(col).nombre;
    return n || col;
  };
  window.cjDef = function (col) { return window.cjFicha(col).definicion || ''; };

  // ---- número en castellano: miles con punto, decimales con coma --------------
  // Mismo criterio que `glosario._numero()`, incluido el signo menos tipográfico.
  function esNum(x, dec) {
    x = Number(x);
    if (!isFinite(x)) return String(x);
    var neg = x < 0, s = Math.abs(x).toFixed(dec), p = s.split('.');
    p[0] = p[0].replace(/\B(?=(\d{3})+(?!\d))/g, '.');
    return (neg ? '−' : '') + p[0] + (p[1] ? ',' + p[1] : '');
  }

  // ---- formatear(): las MISMAS reglas que `glosario.formatear()` del servidor --
  // La unidad no es un sufijo suelto sino un TIPO ('fraccion', 'grados', 'conteo',
  // 'acumulador', 'rms'…), y cada tipo se imprime como lo que es. La regla que lo
  // motivó: no convertir a porcentaje lo que no es una fracción — 8,7 grados no
  // son «870 %» ni 21 voces son «2.100 %».
  var SUFIJO = { grados: '°', segundos: ' s', hz: ' Hz', rms: ' RMS', porcentaje: ' %' };
  var DECIMALES = { grados: 1, segundos: 1, hz: 1, rms: 4, porcentaje: 1, adimensional: 4, fraccion: 3, conteo: 0, acumulador: 0 };

  function rangoTexto(f) {
    var lo = f.rango_min, hi = f.rango_max;
    if (lo === null || hi === null || !isFinite(lo) || !isFinite(hi)) return '';
    var u = f.unidad || '', dec = DECIMALES[u], suf = SUFIJO[u] || '';
    if (dec === undefined) dec = 4;
    if (lo === -hi && hi > 0) return ' (de ±' + esNum(hi, dec) + suf + ')';
    return ' (de ' + esNum(lo, dec) + ' a ' + esNum(hi, dec) + suf + ')';
  }

  window.cjFmt = function (col, v, corto) {
    var f = window.cjFicha(col), u = f.unidad || '';
    if (v === null || v === undefined || v === '') return '—';
    var x = Number(String(v).trim().replace(',', '.'));
    if (u === 'texto' || !isFinite(x)) {
      var t = String(v).trim();
      return t ? t : '—';
    }
    if (u === 'booleano') return x === 0 ? 'no' : 'sí';
    if (u === 'fraccion') return esNum(x * 100, 3) + ' %';
    if (u === 'porcentaje') return esNum(x, 1) + ' %';
    if (u === 'grados') return esNum(x, 1) + '°' + (corto ? '' : rangoTexto(f));
    if (u === 'conteo') {
      var n = Math.round(x), cosa = (f.cosa || ['', ''])[n === 1 ? 0 : 1] || '';
      return (esNum(n, 0) + ' ' + cosa).trim();
    }
    if (u === 'acumulador') {
      var dec = Math.abs(x) >= 10 ? 0 : 4;
      if (corto || f.rango_max === null || !isFinite(f.rango_max)) return esNum(x, dec);
      var techo = esNum(f.rango_max, Math.abs(f.rango_max) >= 10 ? 0 : 4);
      return esNum(x, dec) + ' de ' + techo + (f.rango_medido && !corto ? ' (máx. visto)' : '');
    }
    if (u === 'segundos' || u === 'hz' || u === 'rms') {
      return esNum(x, DECIMALES[u]) + SUFIJO[u] + (corto ? '' : rangoTexto(f));
    }
    var d = DECIMALES[u];
    return esNum(x, d === undefined ? 4 : d) + (corto ? '' : rangoTexto(f));
  };

  // El rango viable declarado, ya legible ('±90°', '0 a 10.730'), para la pestaña.
  window.cjRangoViable = function (col) {
    var f = window.cjFicha(col);
    if (f.unidad === 'fraccion' || f.unidad === 'booleano' || f.unidad === 'texto') return '';
    var t = rangoTexto(f);
    return t ? t.replace(/^ \(de /, '').replace(/\)$/, '') : '';
  };

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  window.cjEsc = esc;

  // Tooltip completo de una sigla: nombre + definición + unidad + rango viable.
  window.cjTitulo = function (col) {
    var f = window.cjFicha(col), t = col;
    if (f.nombre && f.nombre !== col) t += ' — ' + f.nombre;
    if (f.definicion) t += '\n' + f.definicion;
    if (f.unidad) t += '\nunidad: ' + f.unidad;
    var rv = window.cjRangoViable(col);
    if (rv) t += '\nrango: ' + rv + (f.rango_medido ? ' (máx. visto, no declarado)' : '');
    if (f.nodo) t += '\nnodo: ' + f.nodo;
    return t;
  };

  // La etiqueta de una columna: SIEMPRE la sigla visible y, debajo, su nombre descriptivo.
  // `alt` es el rótulo que la caja ya usaba: sirve de nombre mientras el glosario no
  // conozca esa sigla, para que la caja nunca quede peor explicada que antes.
  function etiqueta(col, alt) {
    var nom = window.cjNom(col);
    if (!nom || nom === col) nom = alt || '';
    return '<span class="obsk glosk"><b>' + esc(col) + '</b>'
      + (nom ? '<span class="glosnom">' + esc(nom) + '</span>' : '') + '</span>';
  }

  // Fila de caja POR COLUMNA: sigla + nombre descriptivo + valor formateado.
  window.cjRowG = function (col, v, alt) {
    return '<div class="obsrow glosrow" title="' + esc(window.cjTitulo(col)) + '">'
      + etiqueta(col, alt) + '<span class="obsv">' + esc(window.cjFmt(col, v)) + '</span></div>';
  };

  // Igual que cjRowG pero el valor lo pone la caja (texto, veredicto, sí/no, unidades propias…).
  window.cjRowT = function (col, texto, alt) {
    return '<div class="obsrow glosrow" title="' + esc(window.cjTitulo(col)) + '">'
      + etiqueta(col, alt)
      + '<span class="obsv">' + esc(texto === null || texto === undefined || texto === '' ? '—' : texto)
      + '</span></div>';
  };

  // ---- estilos (inyectados: una caja menos que versionar) --------------------
  function estilos() {
    if (document.getElementById('glos-css')) return;
    var st = document.createElement('style');
    st.id = 'glos-css';
    st.textContent = [
      '.glosrow{align-items:flex-start}',
      '.glosk b{color:#cfe0f5;font-weight:600}',
      '.glosnom{display:block;color:#7f93aa;font-size:9px;line-height:1.25;max-width:150px}',
      '.glostabla{width:100%;border-collapse:collapse;font-size:11px}',
      '.glostabla th{position:sticky;top:0;background:#0e1a28;color:#e8b86d;text-align:left;',
      'padding:6px 7px;border-bottom:1px solid #2a3a4e;cursor:pointer;white-space:nowrap;user-select:none}',
      '.glostabla th:hover{color:#ffd79a}',
      '.glostabla td{padding:5px 7px;border-bottom:1px solid #172333;vertical-align:top;color:#cfe0f0}',
      '.glostabla tr:hover td{background:#101c2b}',
      '.glostabla .gsig{color:#7ddefa;font-weight:bold;white-space:nowrap;font-family:ui-monospace,Menlo,Consolas,monospace}',
      '.glostabla .gnom{color:#e6eefb}',
      '.glostabla .gdef{color:#93a7bd;font-size:10px;max-width:430px}',
      '.glostabla .grng{color:#cfe0f0;white-space:nowrap;font-variant-numeric:tabular-nums}',
      '.glostabla .gvac{color:#5d6f84;font-style:italic}',
      '.glostabla .gvia{color:#6b7f96;font-size:10px;white-space:nowrap}',
      '.glostabla .gnod{color:#b58cff;white-space:nowrap;font-size:10px}',
      '.gloswrap{max-height:62vh;overflow:auto;border:1px solid #22314a;border-radius:8px}',
      '.glosbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:7px}',
      '.glosbar input{flex:1;min-width:180px;background:#0c121b;border:1px solid #2a3a4e;color:#dfe7f0;',
      'border-radius:6px;padding:5px 8px;font-size:12px}',
      '.glosbar .gcount{color:#8aa0b8;font-size:10.5px}'
    ].join('');
    document.head.appendChild(st);
  }

  // ---- la pestaña GLOSARIO ---------------------------------------------------
  var COLUMNAS = [
    { k: 'sigla', t: 'Sigla' },
    { k: 'nombre', t: 'Nombre descriptivo' },
    { k: 'definicion', t: 'Definición' },
    { k: 'unidad', t: 'Unidad' },
    { k: 'obs', t: 'Rango observado' },
    { k: 'nodo', t: 'Nodo de la Teoría' }
  ];
  var orden = { k: 'sigla', asc: true };
  var filtro = '';
  var DATOS = [];
  var TOTAL_PASOS = 0;

  function claveOrden(f, k) {
    if (k === 'obs') {
      if (f.obs_min === null || f.obs_min === undefined) return Number.POSITIVE_INFINITY;
      return Number(f.obs_min);
    }
    return String(f[k] || '').toLocaleLowerCase('es');
  }

  function rangoHTML(f) {
    if (f.obs_min !== null && f.obs_min !== undefined && f.obs_max !== null && f.obs_max !== undefined) {
      var a = window.cjFmt(f.sigla, f.obs_min, true);
      var b = window.cjFmt(f.sigla, f.obs_max, true);
      var txt = (a === b) ? a : (a + '  …  ' + b);
      var viable = window.cjRangoViable(f.sigla);
      var marca = f.rango_medido ? ' (máx. visto)' : '';
      var tit = f.obs_n + ' muestras de esta sesión'
        + (viable ? ' · escala' + (f.rango_medido ? ' medida' : ' declarada') + ': ' + viable : '')
        + (f.obs_max_fmt ? ' · máximo tal como se lee: ' + f.obs_max_fmt : '');
      return '<span class="grng" title="' + esc(tit) + '">' + esc(txt) + '</span>'
        + (viable ? '<span class="gvia"> · de ' + esc(viable + marca) + '</span>' : '');
    }
    if (f.obs_texto) return '<span class="grng">' + esc(f.obs_texto) + '</span>';
    return '<span class="gvac">sin datos aún</span>';   // NUNCA un cero inventado
  }

  // Todo el contenido de la pestaña se arma como UNA cadena y se asigna de una vez: sin
  // querySelector ni handlers por celda. Los clics se delegan en el contenedor (más abajo),
  // que es también lo que sobrevive a que GridStack u otro se coma un evento.
  function pintarHTML() {
    var q = filtro.trim().toLocaleLowerCase('es');
    var vis = DATOS.filter(function (f) {
      if (!q) return true;
      return (String(f.sigla) + ' ' + String(f.nombre) + ' ' + String(f.definicion))
        .toLocaleLowerCase('es').indexOf(q) >= 0;
    });
    vis = vis.slice().sort(function (a, b) {
      var x = claveOrden(a, orden.k), y = claveOrden(b, orden.k);
      if (x < y) return orden.asc ? -1 : 1;
      if (x > y) return orden.asc ? 1 : -1;
      return String(a.sigla).localeCompare(String(b.sigla), 'es');
    });
    var th = COLUMNAS.map(function (c) {
      var fl = (orden.k === c.k) ? (orden.asc ? ' ▲' : ' ▼') : '';
      return '<th data-k="' + c.k + '">' + esc(c.t) + fl + '</th>';
    }).join('');
    var tr = vis.map(function (f) {
      return '<tr>'
        + '<td class="gsig">' + esc(f.sigla) + '</td>'
        + '<td class="gnom">' + esc(f.nombre === f.sigla ? '—' : f.nombre) + '</td>'
        + '<td class="gdef">' + esc(f.definicion) + '</td>'
        + '<td>' + esc(f.unidad === 'fraccion' ? 'fracción (%)' : (f.unidad || '')) + '</td>'
        + '<td>' + rangoHTML(f) + '</td>'
        + '<td class="gnod">' + esc(f.nodo) + '</td>'
        + '</tr>';
    }).join('');
    var cuenta = vis.length + ' de ' + DATOS.length + ' columnas'
      + (TOTAL_PASOS
        ? ' · rango observado sobre ' + TOTAL_PASOS.toLocaleString('es') + ' pasos de esta sesión'
        : ' · el organismo aún no ha dado ningún paso en esta sesión');
    var cuerpo = CARGADO
      ? ('<table class="glostabla"><thead><tr>' + th + '</tr></thead><tbody>'
        + (tr || '<tr><td colspan="6" class="gvac">nada coincide con la búsqueda</td></tr>')
        + '</tbody></table>')
      : 'cargando glosario…';
    return '<div class="glosbar">'
      + '<input type="text" class="gbusca" placeholder="buscar por sigla, nombre o definición…" value="' + esc(filtro) + '">'
      + '<button type="button" class="sm gref">↻ actualizar rangos</button>'
      + '<a class="sm gcsv" href="/glosario.csv" download style="text-decoration:none;padding:5px 8px;'
      + 'border:1px solid #2a3a4e;border-radius:6px;color:#e8b86d;font-size:11px">⬇ glosario.csv</a>'
      + '<span class="gcount">' + esc(cuenta) + '</span>'
      + '</div><div class="gloswrap">' + cuerpo + '</div>';
  }

  function pintar(cont, foco) {
    cont.innerHTML = pintarHTML();
    if (foco && cont.querySelector) {
      var i = cont.querySelector('.gbusca');
      if (i && i.focus) { try { i.focus(); i.setSelectionRange(i.value.length, i.value.length); } catch (e) { } }
    }
  }

  window.glosPintar = function (cont) {
    if (!cont) return;
    estilos();
    if (!cont.__glosBound) {
      cont.__glosBound = 1;
      // Delegado y en fase de CAPTURA: el mismo motivo por el que el "?" de las cajas tuvo que
      // capturar — si algún contenedor (GridStack, el portal) se come el clic, aquí ya se atendió.
      cont.addEventListener('click', function (e) {
        var t = e.target;
        if (!t || !t.closest) return;
        var h = t.closest('th[data-k]');
        if (h) {
          e.preventDefault(); e.stopPropagation();
          var k = h.dataset.k;
          if (orden.k === k) orden.asc = !orden.asc; else { orden.k = k; orden.asc = true; }
          pintar(cont);
          return;
        }
        if (t.closest('.gref')) {
          e.preventDefault(); e.stopPropagation();
          cargar(true).then(function () { pintar(cont); });
        }
      }, true);
      cont.addEventListener('input', function (e) {
        if (e.target && e.target.closest && e.target.closest('.gbusca')) {
          filtro = e.target.value;
          pintar(cont, true);
        }
      }, true);
    }
    pintar(cont);
    if (!CARGADO) cargar(false).then(function () { pintar(cont); });
  };

  // ---- carga del diccionario -------------------------------------------------
  function cargar(forzar) {
    if (CARGADO && !forzar) return Promise.resolve();
    return fetch('/glosario', { cache: 'no-cache' })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d || !Array.isArray(d.columnas)) return;
        DATOS = d.columnas;
        TOTAL_PASOS = d.n || 0;
        var m = {};
        DATOS.forEach(function (f) { m[f.sigla] = f; });
        window.GLOS = m;
        CARGADO = true;
        // Lo que ya está pintado con siglas a secas se vuelve a pintar CON nombre.
        try { if (window.glosRefrescarLeyendas) window.glosRefrescarLeyendas(); } catch (e) { }
        try { if (window.renderCajas && window._ultimaFila) window.renderCajas(window._ultimaFila); } catch (e) { }
        try { window.dispatchEvent(new Event('anima-glosario')); } catch (e) { }
      })
      .catch(function () { /* sin glosario la UI sigue: la sigla es su propio nombre */ });
  }
  window.glosCargar = cargar;

  if (document.readyState !== 'loading') cargar(false);
  else document.addEventListener('DOMContentLoaded', function () { cargar(false); });
})();
