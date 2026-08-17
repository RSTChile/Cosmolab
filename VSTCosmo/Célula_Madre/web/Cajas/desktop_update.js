/* ANIMA Desktop — actualizaciones (auto por defecto para colaboradores) */
(function () {
  const STYLE = `
  #anima-update-banner{
    position:fixed;right:14px;bottom:14px;z-index:99999;
    max-width:360px;background:#0f1720;color:#e8f1f8;
    border:1px solid #2a6f8f;border-radius:12px;padding:12px 14px;
    font:13px/1.4 system-ui,Segoe UI,sans-serif;
    box-shadow:0 8px 28px rgba(0,0,0,.35);display:none
  }
  #anima-update-banner b{color:#7ddefa}
  #anima-update-banner .row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
  #anima-update-banner button{
    border:0;border-radius:8px;padding:7px 12px;cursor:pointer;font-weight:600
  }
  #anima-update-banner .ok{background:#1f9d6a;color:#fff}
  #anima-update-banner .no{background:#243040;color:#cfe3f2}
  #anima-update-banner .mut{opacity:.75;font-size:11px;margin-top:6px}
  `;

  function el(tag, attrs, html) {
    const n = document.createElement(tag);
    if (attrs) Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
    if (html != null) n.innerHTML = html;
    return n;
  }

  function ensureUi() {
    if (document.getElementById("anima-update-banner")) return;
    document.head.appendChild(el("style", null, STYLE));
    const box = el("div", { id: "anima-update-banner" });
    box.innerHTML = `
      <div><b id="anima-update-title">Actualización</b></div>
      <div id="anima-update-msg" class="mut"></div>
      <div class="row" id="anima-update-actions">
        <button class="ok" id="anima-update-yes">Actualizar ahora</button>
        <button class="no" id="anima-update-later">Después</button>
      </div>
      <div class="mut" id="anima-update-foot"></div>`;
    document.body.appendChild(box);
    document.getElementById("anima-update-later").onclick = () => {
      box.style.display = "none";
    };
    document.getElementById("anima-update-yes").onclick = () => applyUpdate();
  }

  async function applyUpdate() {
    ensureUi();
    const box = document.getElementById("anima-update-banner");
    const foot = document.getElementById("anima-update-foot");
    const title = document.getElementById("anima-update-title");
    const actions = document.getElementById("anima-update-actions");
    const yes = document.getElementById("anima-update-yes");
    box.style.display = "block";
    title.textContent = "Actualizando…";
    if (actions) actions.style.display = "none";
    if (yes) yes.disabled = true;
    foot.textContent = "Descargando e instalando en segundo plano…";
    try {
      const r = await fetch("/api/desktop/update/apply", { method: "POST" });
      const d = await r.json();
      if (d.ok) {
        foot.textContent = (d.message || "Listo.") + " Reiniciando…";
        setTimeout(() => location.reload(), 1200);
      } else {
        title.textContent = "No se pudo actualizar";
        foot.textContent = d.message || "Error desconocido.";
        if (actions) actions.style.display = "flex";
        if (yes) yes.disabled = false;
      }
    } catch (e) {
      title.textContent = "Error de red";
      foot.textContent = "No se pudo contactar el servicio de actualizaciones.";
      if (actions) actions.style.display = "flex";
      if (yes) yes.disabled = false;
    }
  }

  async function check() {
    try {
      const r = await fetch("/api/desktop/update/check", { cache: "no-store" });
      const d = await r.json();
      if (!d || !d.ok) return;
      if (!d.update_available) return;

      ensureUi();
      const box = document.getElementById("anima-update-banner");
      const title = document.getElementById("anima-update-title");
      const msg = document.getElementById("anima-update-msg");
      const foot = document.getElementById("anima-update-foot");
      const actions = document.getElementById("anima-update-actions");

      msg.textContent =
        d.message ||
        (d.remote && d.remote.version
          ? "Versión " + d.remote.version
          : "Hay una nueva versión de ANIMA.");
      foot.textContent =
        "Local: " + (d.local_version || "?") +
        (d.remote && d.remote.version ? " → " + d.remote.version : "");

      // Colaboradores: auto por defecto
      const auto = d.auto_update !== false && d.mandatory !== false
        ? (d.auto_update === true || d.mandatory === true || d.auto_update == null)
        : false;
      // Simplificado: si auto_update es true O no viene desactivado → auto
      const doAuto = d.auto_update !== false;

      if (doAuto) {
        title.textContent = "Actualización automática";
        if (actions) actions.style.display = "none";
        box.style.display = "block";
        await applyUpdate();
      } else {
        title.textContent = "Actualización disponible";
        if (actions) actions.style.display = "flex";
        box.style.display = "block";
      }
    } catch (_) {
      /* silencioso sin red */
    }
  }

  // Primera comprobación pronto; luego cada 6 h si la página sigue abierta
  const firstDelay = 2000;
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(check, firstDelay));
  } else {
    setTimeout(check, firstDelay);
  }
  setInterval(check, 6 * 60 * 60 * 1000);
})();
