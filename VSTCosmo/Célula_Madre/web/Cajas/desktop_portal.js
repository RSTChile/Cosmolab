(function () {
  'use strict';
  if (window.__ANIMA_DESKTOP_PORTAL__) return;
  window.__ANIMA_DESKTOP_PORTAL__ = true;
  window.ANIMA_PORTAL_ACTIVE = 'organismo';

  const PUBLIC_OBSERVATORY = 'https://observatorio.cosmosemiotica.cl/';
  const COSMOSEMIOTICA_URL = 'https://cosmosemiotica.cl/';
  const BITACORA_SOCIAL_URL = 'https://bitacora.cosmosemiotica.cl/#ahora';
  const ORIGINAL_OBSERVATORY = '/observatorio-original';
  // URL LAN opcional solo si el host la configura (meta o window). Nunca hardcode de lab.
  const SOCIEDAD_LAN = (window.ANIMA_SOCIEDAD_URL || document.querySelector('meta[name="anima-sociedad-url"]')?.content || '').replace(/\/$/, '');
  const originalViews = {vivo: 'vivo', historia: 'hist', circuito: 'circ'};
  const tabs = [
    ['organismo', '🧬 Organismo'],
    ['vivo', '🟢 En vivo'],
    ['historia', '🕮 Historia'],
    ['circuito', '🫀 Circuito vivo'],
    ['campo', '📡 Campo local'],
    ['observatorio', '🌐 Observatorio'],
    ['bitacora', '📓 Bitácora Social'],
    ['cosmosemiotica', '✦ Cosmosemiótica']
  ];
  let active = 'organismo';
  let organismView;
  let originalFrame = null;
  let publicFrame = null;
  let cosmoFrame = null;
  let bitacoraFrame = null;
  let campoTimer = null;

  function installStyle() {
    const style = document.createElement('style');
    style.textContent = `
      .anima-portal-nav{position:sticky;top:0;z-index:10020;display:flex;gap:6px;align-items:center;padding:10px 14px;background:rgba(7,11,17,.97);border-bottom:1px solid #26374b;box-shadow:0 8px 24px rgba(0,0,0,.28);overflow-x:auto}
      .anima-portal-tab{appearance:none;border:1px solid #2b3d53;border-radius:999px;background:#111a26;color:#aebed0;padding:9px 15px;font:600 13px/1 system-ui,sans-serif;white-space:nowrap;cursor:pointer}
      .anima-portal-tab:hover{border-color:#5d83ad;color:#fff}.anima-portal-tab.on{background:#e8b86d;color:#111820;border-color:#e8b86d}
      .anima-portal-view{display:none;min-height:calc(100vh - 56px);padding:12px;box-sizing:border-box;background:#0a0e14;color:#dfe7f0;font-family:system-ui,-apple-system,Segoe UI,sans-serif}
      .anima-portal-view.on{display:block}.anima-original-frame,.anima-public-frame,.anima-cosmo-frame{display:block;width:100%;height:calc(100vh - 82px);min-height:680px;border:1px solid #243246;border-radius:12px;background:#172642}
      .anima-public-wrap{max-width:1500px;margin:0 auto}.anima-frame-note{display:flex;justify-content:space-between;gap:12px;align-items:center;margin:0 2px 10px;color:#8aa0b8;flex-wrap:wrap}.anima-frame-note a{color:#6db6ff}
      .anima-loading{display:flex;align-items:center;justify-content:center;min-height:65vh;color:#8aa0b8;text-align:center}
      .anima-honest{max-width:720px;margin:16px auto;padding:14px 16px;border:1px solid #2b3d53;border-radius:12px;background:#111a26;color:#aebed0;line-height:1.45;font-size:14px}
      .anima-honest strong{color:#e8b86d}
      .anima-campo-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;max-width:1200px;margin:0 auto}
      .anima-campo-card{border:1px solid #243246;border-radius:12px;background:#141e2e;padding:14px}
      .anima-campo-card h3{margin:0 0 6px;font-size:15px;color:#e8b86d}
      .anima-campo-card .mut{color:#8aa0b8;font-size:12px}
      .anima-campo-card a{color:#6db6ff}
      .anima-campo-self{border-color:#3d6b4a}
      @media(max-width:760px){.anima-portal-view{padding:6px}.anima-original-frame,.anima-public-frame,.anima-cosmo-frame{height:calc(100vh - 74px);min-height:620px}.anima-frame-note{font-size:12px}}
    `;
    document.head.appendChild(style);
  }

  function build() {
    const nav = document.createElement('nav');
    nav.className = 'anima-portal-nav';
    nav.setAttribute('aria-label', 'Secciones de ANIMA');
    tabs.forEach(([id, label]) => {
      const button = document.createElement('button');
      button.className = 'anima-portal-tab' + (id === active ? ' on' : '');
      button.dataset.view = id;
      button.textContent = label;
      button.addEventListener('click', () => activate(id));
      nav.appendChild(button);
    });
    document.body.insertBefore(nav, document.body.firstChild);

    // Interlocutor en vivo (Atención Social): a quién ESCUCHA/HABLA el organismo por su propia
    // decisión. Banner siempre visible (chrome del portal), independiente de la vista embebida.
    const focoSocial = document.createElement('div');
    focoSocial.id = 'anima-foco-social';
    focoSocial.style.cssText = 'padding:5px 14px;font-size:12px;color:#cfe0f0;background:#0c1622;border-bottom:1px solid #243246';
    focoSocial.textContent = '🧭 Atención social: —';
    nav.insertAdjacentElement('afterend', focoSocial);
    async function refrescarFocoSocial() {
      try {
        const r = await fetch('/ultima_fila', {cache: 'no-store'});
        const f = (await r.json()).fila || {};
        if (f.as_modo == null) { focoSocial.style.display = 'none'; return; }
        focoSocial.style.display = '';
        const esc = f.as_esc_nombre, hab = f.as_habla_nombre;
        const sesgo = (f.as_esc_sesgo && f.as_esc_sesgo !== '-') ? ' · ' + f.as_esc_sesgo : '';
        let txt;
        if (esc && hab && esc !== hab) txt = '👂 escucha a ' + esc + sesgo + ' · 🗣 habla a ' + hab;
        else if (esc || hab) txt = 'conversando con ' + (esc || hab) + sesgo;
        else txt = 'sin interlocutor (libertad funcional) · ' + (f.as_n_candidatos || 0) + ' en el campo';
        focoSocial.textContent = '🧭 Atención social: ' + txt;
      } catch (e) {}
    }
    refrescarFocoSocial();
    setInterval(refrescarFocoSocial, 2000);

    organismView = document.querySelector('.wrap') || document.body.children[1];
    const portal = document.createElement('div');
    portal.innerHTML = `
      <section class="anima-portal-view" id="anima-portal-original">
        <div id="anima-original-holder" class="anima-loading">El Observatorio original se cargará al abrir una de sus vistas.</div>
      </section>
      <section class="anima-portal-view" id="anima-portal-campo">
        <div class="anima-public-wrap">
          <div class="anima-frame-note"><span>Campo local — organismos visibles por presencia (mDNS/UDP) en tu red.</span>
            <button type="button" id="anima-campo-refresh" class="anima-portal-tab" style="padding:6px 12px">Actualizar</button>
          </div>
          <div id="anima-campo-holder" class="anima-loading">Consultando /presencia…</div>
        </div>
      </section>
      <section class="anima-portal-view" id="anima-portal-observatorio">
        <div class="anima-public-wrap">
          <div class="anima-frame-note">
            <span>Observatorio público del ecosistema ANIMA.</span>
            <span>
              <a href="${PUBLIC_OBSERVATORY}" target="_blank" rel="noopener">Abrir en ventana nueva ↗</a>
              ${SOCIEDAD_LAN ? ` · <a href="${SOCIEDAD_LAN}/" target="_blank" rel="noopener">Sociedad LAN</a>` : ''}
            </span>
          </div>
          <div id="anima-obs-honest" class="anima-honest" style="display:none"></div>
          <div id="anima-public-holder" class="anima-loading">El Observatorio público se cargará al abrir esta pestaña.</div>
        </div>
      </section>
      <section class="anima-portal-view" id="anima-portal-bitacora">
        <div class="anima-public-wrap">
          <div class="anima-frame-note">
            <span>Bitácora Social — registro compartido de lo que hacen todos los organismos.</span>
            <span>
              <a href="${BITACORA_SOCIAL_URL}" target="_blank" rel="noopener">Abrir en ventana nueva ↗</a>
            </span>
          </div>
          <div id="anima-bitacora-holder" class="anima-loading">La bitácora se cargará al abrir esta pestaña.</div>
        </div>
      </section>
      <section class="anima-portal-view" id="anima-portal-cosmosemiotica">
        <div class="anima-public-wrap">
          <div class="anima-frame-note">
            <span>Sitio madre del experimento — Teoría Cosmosemiótica · Cosmolab / VST Cosmo.</span>
            <span>
              <a href="${COSMOSEMIOTICA_URL}" target="_blank" rel="noopener">Abrir en ventana nueva ↗</a>
            </span>
          </div>
          <div id="anima-cosmo-holder" class="anima-loading">El sitio cosmosemiotica.cl se cargará al abrir esta pestaña.</div>
        </div>
      </section>`;
    document.body.appendChild(portal);
    const refreshBtn = document.getElementById('anima-campo-refresh');
    if (refreshBtn) refreshBtn.addEventListener('click', () => renderCampo(true));
  }

  function ensureOriginal(view) {
    if (!originalFrame) {
      const holder = document.getElementById('anima-original-holder');
      holder.className = '';
      originalFrame = document.createElement('iframe');
      originalFrame.className = 'anima-original-frame';
      originalFrame.title = 'Observatorio original de la conversación ANIMA';
      originalFrame.loading = 'eager';
      originalFrame.src = ORIGINAL_OBSERVATORY + '#embed=1&vista=' + encodeURIComponent(view);
      holder.replaceChildren(originalFrame);
      originalFrame.addEventListener('load', () => sendOriginalView(view));
    } else {
      sendOriginalView(view);
    }
  }

  function sendOriginalView(view) {
    if (originalFrame && originalFrame.contentWindow) {
      originalFrame.contentWindow.postMessage({animaVista: view}, location.origin);
    }
  }

  async function probePublicObservatory() {
    const box = document.getElementById('anima-obs-honest');
    if (!box) return;
    try {
      const r = await fetch('/api/desktop/observatorio/datos', {cache: 'no-store'});
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const data = await r.json();
      const roster = Array.isArray(data.roster) ? data.roster : [];
      const others = roster.filter(e => {
        const name = (e.name || e.organism_id || '').toLowerCase();
        return name && !name.includes('self');
      });
      // Si solo hay 0-1 entradas o falla la lectura de "otros", mensaje honesto
      if (roster.length <= 1) {
        box.style.display = 'block';
        box.innerHTML = '<strong>Presencia local.</strong> Tu organismo está en modo local. ' +
          'Solo aparece en el Observatorio público si el anfitrión de la sociedad te descubre en la LAN ' +
          'o si publicas presencia (futuro). Mientras tanto usa la pestaña <strong>Campo local</strong> ' +
          'para ver vecinos en tu red.';
      } else {
        box.style.display = 'none';
        box.innerHTML = '';
      }
    } catch (e) {
      console.warn('[ANIMA] No se pudo consultar el Observatorio público:', e);
      box.style.display = 'block';
      box.innerHTML = '<strong>No se pudo consultar el Observatorio público.</strong> ' +
        'Tu organismo puede seguir vivo en local. Revisa la conexión a Internet o abre la pestaña ' +
        '<strong>Campo local</strong> (funciona offline en tu LAN).';
    }
  }

  function ensurePublic() {
    probePublicObservatory();
    if (publicFrame) return;
    const holder = document.getElementById('anima-public-holder');
    holder.className = '';
    publicFrame = document.createElement('iframe');
    publicFrame.className = 'anima-public-frame';
    publicFrame.title = 'Observatorio público ANIMA';
    publicFrame.loading = 'lazy';
    publicFrame.src = PUBLIC_OBSERVATORY;
    holder.replaceChildren(publicFrame);
  }

  function ensureBitacora() {
    if (bitacoraFrame) return;
    const holder = document.getElementById('anima-bitacora-holder');
    if (!holder) return;
    holder.className = '';
    bitacoraFrame = document.createElement('iframe');
    bitacoraFrame.className = 'anima-cosmo-frame';
    bitacoraFrame.title = 'Bitácora Social ANIMA';
    bitacoraFrame.loading = 'lazy';
    bitacoraFrame.referrerPolicy = 'no-referrer-when-downgrade';
    bitacoraFrame.src = BITACORA_SOCIAL_URL;
    holder.replaceChildren(bitacoraFrame);
  }

  function ensureCosmosemiotica() {
    if (cosmoFrame) return;
    const holder = document.getElementById('anima-cosmo-holder');
    if (!holder) return;
    holder.className = '';
    cosmoFrame = document.createElement('iframe');
    cosmoFrame.className = 'anima-cosmo-frame';
    cosmoFrame.title = 'Cosmosemiótica — cosmosemiotica.cl';
    cosmoFrame.loading = 'lazy';
    cosmoFrame.referrerPolicy = 'no-referrer-when-downgrade';
    cosmoFrame.src = COSMOSEMIOTICA_URL;
    holder.replaceChildren(cosmoFrame);
  }

  async function renderCampo(force) {
    const holder = document.getElementById('anima-campo-holder');
    if (!holder) return;
    if (!force && holder.dataset.loaded === '1' && active !== 'campo') return;
    try {
      const [presR, estR, idR] = await Promise.all([
        fetch('/presencia', {cache: 'no-store'}),
        fetch('/estado', {cache: 'no-store'}),
        fetch('/identidad', {cache: 'no-store'})
      ]);
      const pres = presR.ok ? await presR.json() : {vecinos: []};
      const est = estR.ok ? await estR.json() : {};
      const id = idR.ok ? await idR.json() : {};
      const vecinos = Array.isArray(pres.vecinos) ? pres.vecinos : [];
      const cards = [];
      cards.push(
        `<article class="anima-campo-card anima-campo-self">
          <h3>${escapeHtml(est.organismo || id.name || 'Tú')}</h3>
          <div class="mut">este organismo · ${escapeHtml(id.organism_id || est.organismo_id || '')}</div>
          <div class="mut">vivo: ${est.vivo ? 'sí' : 'no'} · <a href="/">abrir</a></div>
        </article>`
      );
      vecinos.forEach(v => {
        const base = (v.base_url || '').replace(/\/$/, '');
        cards.push(
          `<article class="anima-campo-card">
            <h3>${escapeHtml(v.name || v.organism_id || 'vecino')}</h3>
            <div class="mut">${escapeHtml(v.organism_id || '')}</div>
            <div class="mut">${escapeHtml(v.estado_presencia || '')} · fuente ${escapeHtml(v.source || '?')} · frescura ${v.frescura != null ? v.frescura : '—'}</div>
            ${base ? `<div class="mut"><a href="${escapeAttr(base)}/" target="_blank" rel="noopener">${escapeHtml(base)}</a></div>` : ''}
          </article>`
        );
      });
      if (vecinos.length === 0) {
        cards.push(
          `<article class="anima-campo-card"><h3>Sin vecinos aún</h3>
            <div class="mut">Si hay otro ANIMA en la misma LAN, debería aparecer aquí por mDNS/UDP en menos de un minuto. No se usan IPs de laboratorio fijas.</div></article>`
        );
      }
      holder.className = 'anima-campo-grid';
      holder.innerHTML = cards.join('');
      holder.dataset.loaded = '1';
    } catch (e) {
      holder.className = 'anima-loading';
      holder.textContent = 'No se pudo leer /presencia o /estado. ¿Sigue vivo el organismo?';
    }
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function escapeAttr(s) {
    return escapeHtml(s).replace(/'/g, '&#39;');
  }

  function activate(id) {
    active = id;
    window.ANIMA_PORTAL_ACTIVE = id;
    document.querySelectorAll('.anima-portal-tab').forEach(button => {
      if (button.dataset.view) button.classList.toggle('on', button.dataset.view === id);
    });
    if (organismView) organismView.style.display = id === 'organismo' ? '' : 'none';
    const original = document.getElementById('anima-portal-original');
    const publicView = document.getElementById('anima-portal-observatorio');
    const cosmoView = document.getElementById('anima-portal-cosmosemiotica');
    const campoView = document.getElementById('anima-portal-campo');
    if (original) original.classList.toggle('on', Object.prototype.hasOwnProperty.call(originalViews, id));
    if (publicView) publicView.classList.toggle('on', id === 'observatorio');
    const bitView = document.getElementById('anima-portal-bitacora');
    if (bitView) bitView.classList.toggle('on', id === 'bitacora');
    if (cosmoView) cosmoView.classList.toggle('on', id === 'cosmosemiotica');
    if (campoView) campoView.classList.toggle('on', id === 'campo');
    if (originalViews[id]) ensureOriginal(originalViews[id]);
    if (id === 'observatorio') ensurePublic();
    if (id === 'bitacora') ensureBitacora();
    if (id === 'cosmosemiotica') ensureCosmosemiotica();
    if (id === 'campo') {
      renderCampo(true);
      if (campoTimer) clearInterval(campoTimer);
      campoTimer = setInterval(() => { if (active === 'campo') renderCampo(true); }, 8000);
    } else if (campoTimer) {
      clearInterval(campoTimer);
      campoTimer = null;
    }
    history.replaceState(null, '', id === 'organismo' ? location.pathname : '#' + id);
  }

  function boot() {
    installStyle();
    build();
    const requested = location.hash.replace('#', '');
    activate(tabs.some(tab => tab[0] === requested) ? requested : 'organismo');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, {once: true});
  else boot();
})();
