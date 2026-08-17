# Plan — cosmosemiotica.cl + Observatorio ANIMA (partiendo de cero)

**Fecha:** 2026-07-08  
**Decisión:** No tocar nameservers de `geografiasagrada.cl`.  
**Camino:** dominio propio `cosmosemiotica.cl` → Cloudflare → túnel al observatorio vivo (`:9101`).

---

## 1. Objetivo

| URL | Contenido |
|-----|-----------|
| `https://cosmosemiotica.cl/` | (futuro) sede / landing del proyecto |
| `https://observatorio.cosmosemiotica.cl/` | **Observatorio ANIMA vivo** (cabezas Three.js, `:9101` vía túnel) |

Geografía Sagrada puede enlazar al observatorio; no hospeda el stream en vivo.

---

## 2. Compra del dominio (.cl)

1. Ir a [https://www.nic.cl](https://www.nic.cl) (o un registrar autorizado NIC Chile).
2. Buscar **`cosmosemiotica.cl`**.
3. Si está libre: registrar a nombre del proyecto / persona jurídica que corresponda.
4. Completar datos de contacto (NIC Chile es estricto con datos).
5. Pagar y esperar confirmación de inscripción (suele ser rápido).

**No** configures aún un hosting web “completo” en el registrar si vas a usar solo Cloudflare + túnel.  
Basta con **poseer el dominio**.

---

## 3. Añadir el dominio a Cloudflare (sí aquí; no en GS)

1. Cuenta en [https://dash.cloudflare.com](https://dash.cloudflare.com).
2. **Add a site** → `cosmosemiotica.cl`.
3. Plan **Free** alcanza para el túnel + DNS.
4. Cloudflare muestra **2 nameservers** (ej. `xxx.ns.cloudflare.com`).
5. En **NIC Chile** (o el panel del registrar):
   - Cambiar **servidores de nombres** del dominio `cosmosemiotica.cl` a esos de Cloudflare.
6. En Cloudflare, esperar estado **Active** (minutos a pocas horas).

Esto **no afecta** geografiasagrada.cl.

### DNS mínimo inicial (en Cloudflare → DNS)

Puedes dejar solo lo del túnel; no hace falta página aún.

| Tipo | Nombre | Contenido | Proxy |
|------|--------|-----------|--------|
| (lo crea el túnel) | `observatorio` | → cloudflared | Proxied (naranja) |

Opcional después:

| Tipo | Nombre | Contenido |
|------|--------|-----------|
| A/AAAA o CNAME | `@` | landing futura |
| CNAME | `www` | `@` o landing |

---

## 4. Túnel Cloudflare — **solo PC (Abraxas `.38`)**

**Decisión 2026-07-09:** el Mac del lab **no** hospeda ni corre Cloudflare.  
`cloudflared` se desinstaló del Mac; no reinstalar ahí.

Origen del sitio público:

| Pieza | Dónde |
|-------|--------|
| `vst_sociedad` (`:9101`) | PC `192.168.86.38` |
| `cloudflared` túnel `anima-observatorio` | PC (`C:\Users\adale\ANIMA\cloudflared\`) |
| Lab A–D Docker | Mac (solo sujetos de prueba; no origen público) |

Config de referencia en el PC (`config.yml`):

```yaml
tunnel: anima-observatorio
credentials-file: C:\Users\adale\ANIMA\cloudflared\<TUNNEL_ID>.json

ingress:
  - hostname: observatorio.cosmosemiotica.cl
    service: http://192.168.86.38:9101   # o http://127.0.0.1:9101 en el mismo host
  - service: http_status:404
```

Arranque en el PC: `packaging/observatorio-abraxas/start_all.ps1` (sociedad + túnel).
---

## 5. Observatorio en Abraxas (requisitos)

**Separación:** el Mac es lab interno; el sitio público vive solo en Abraxas (`.38`).

Siempre que el túnel esté up:

1. `vst_sociedad.py` en **:9101** con `ANIMA_SOCIEDAD_PUBLIC=1`
2. Semillas alcanzables desde Abraxas: organismo local (`127.0.0.1:7788`, p. ej. **Abraxas** del instalable) + Pis `.22` / `.33` si están en red
3. `cloudflared tunnel run anima-observatorio` en el mismo PC

Arranque (scripts en repo):

```powershell
cd C:\Users\adale\ANIMA\celula_madre
.\packaging\observatorio-abraxas\start_all.ps1
```

Salud local: `http://127.0.0.1:9101/salud`  
Config: `packaging/observatorio-abraxas/config/observatorio.env`

---

## 6. Qué hacer con geografiasagrada.cl

| Acción | Recomendación |
|--------|----------------|
| Página `observatorio-anima` (POC estática) | Borrador / “Próximamente” + enlace a `https://observatorio.cosmosemiotica.cl` |
| Nameservers GS | **No tocar** |
| Enlace menú | “Observatorio ANIMA (en vivo)” → dominio nuevo |

---

## 7. Checklist para Alexis

- [ ] Comprar/inscribir `cosmosemiotica.cl` en NIC Chile  
- [ ] Add site en Cloudflare (plan Free)  
- [ ] Cambiar nameservers **solo** de cosmosemiotica.cl  
- [ ] Esperar Active  
- [ ] Avisar al agente (o ejecutar §4): tunnel + DNS `observatorio`  
- [ ] Probar desde el móvil (datos móviles): cabezas en movimiento  
- [ ] Actualizar enlace en GS  

---

## 8. Costes orientativos

| Ítem | Orden de magnitud |
|------|-------------------|
| `.cl` (NIC) | tarifa anual NIC Chile |
| Cloudflare Free | $0 |
| cloudflared + Mac lab | $0 (usa el equipo del lab) |

---

## 9. Después (no bloquea el día 1)

- Landing en `https://cosmosemiotica.cl` (teoría, papers, equipo)  
- Plaza/API en subdominio `plaza.cosmosemiotica.cl` (Fase 5 spec)  
- TLS y WAF ya vienen con proxy naranja de Cloudflare  

---

## 10. Estado actual del lab (2026-07-08)

- Dominio **Active** en Cloudflare Free  
- Túnel nombrado **`anima-observatorio`** (`f4f48999-9268-470a-8c73-29ef5f9dcb6b`)  
- DNS: `observatorio.cosmosemiotica.cl` → túnel  
- Config: `~/.cloudflared/config.yml`  
- Local: `vst_sociedad` `:9101` con `ANIMA_SOCIEDAD_PUBLIC=1`  
- URL viva (HTTP OK al montar; HTTPS espera Universal SSL de CF, minutos–horas):  
  - `http://observatorio.cosmosemiotica.cl/`  
  - `https://observatorio.cosmosemiotica.cl/` (cuando Edge Certificate esté Active)  
- Arranque túnel: `cloudflared tunnel --config ~/.cloudflared/config.yml run anima-observatorio`  
- Log: `Célula_Madre/docker/cloudflared-named.log`
