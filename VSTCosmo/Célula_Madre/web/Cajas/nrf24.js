// Caja: RADIO DIGITAL (nRF24) — la 3a radio del organismo, el canal SHANNON.
// Paquetes discretos, direccionados, con ACK: E (nRF24 en el ATmega) ↔ A (nRF24 en su
// Arduino Uno). Complementa la voz analógica anti-Shannon (HackRF). Lee columnas nrf_*.
// Latente en organismos sin la radio digital (solo E la tiene por hardware).
window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push({
  id:'nrf24', tit:'Radio digital (nRF24)', w:4, h:8, render:(b,r,bf)=>{
    const activo = (r.nrf_ok !== undefined && r.nrf_ok !== null);
    const chip = Number(r.nrf_ok)>0.5;
    const conn = Number(r.nrf_connected)>0.5;
    const vivo = Number(r.nrf_vivo)>0.5;
    const rx = Number(r.nrf_rx)||0, tx = Number(r.nrf_tx)||0;
    const rxd = Number(r.nrf_rx_delta)||0;
    const lastrx = (r.nrf_last_rx==null?'':String(r.nrf_last_rx)).trim();
    const lasttx = (r.nrf_last_tx==null?'':String(r.nrf_last_tx)).trim();
    const manual = Number(r.nrf_tx_manual)>0.5;
    const backend = (r.nrf_backend==null?'':String(r.nrf_backend)).trim();

    // --- indicador del ENLACE (verde vivo; azul pulsante al recibir; ámbar chip-sin-enlace; gris) ---
    const box = (bg,bd,fg,txt,glow)=>'<div style="text-align:center;padding:9px;margin-bottom:8px;'
      +'border-radius:8px;background:'+bg+';border:1px solid '+bd+';color:'+fg+';font-weight:700;'
      +'letter-spacing:.3px'+(glow?';box-shadow:0 0 14px '+glow:'')+'">'+txt+'</div>';
    let air;
    if(!activo){
      air = box('#12181c','#2a3a44','#6a7a84','radio digital latente','');
    } else if(rxd>0){
      air = box('#0d2a3a','#4aa0ff','#8ac6ff','◉ recibiendo paquete','#4aa0ff55');
    } else if(vivo){
      air = box('#12241a','#3a8c5a','#8ad3a0','● enlace vivo (canal Shannon)','');
    } else if(chip){
      air = box('#241a10','#8c6a3a','#d3b58a','chip OK · sin enlace','');
    } else {
      air = box('#3a0d10','#ff5a4a','#ff8c7a','radio digital no responde','');
    }

    const status = air
      + '<div class="obsSub">nodo (E) ↔ par (A · por el Uno)</div>'
      + cjRow('chip SPI', si(r.nrf_ok))
      + cjRow('conectado', si(r.nrf_connected))
      + cjRow('nodo', backend || (activo ? 'radio' : '—'))
      + cjRow('TX manual', si(r.nrf_tx_manual))
      + cjRow('direcciones', 'ANE01 ↔ ANA01')
      + '<div class="obsSub">recepción — paquetes del otro</div>'
      + cjRow('recibidos', rx)
      + cjRow('último recibido', lastrx || '—')
      + '<div class="obsSub">transmisión — paquetes al otro</div>'
      + cjRow('enviados', tx)
      + cjRow('último envío', lasttx || '—');

    // --- construir el DOM UNA vez (status + enviador); en cada refresco SOLO se actualiza el status ---
    if(!b._nrf){
      b.innerHTML =
          '<div class="nrfStatus"></div>'
        + '<div class="obsSub" style="margin-top:8px">enviar mensaje digital →</div>'
        + '<div style="display:flex;gap:6px;margin-top:4px">'
        +   '<input class="nrfMsg" type="text" maxlength="30" placeholder="texto (≤30 chars)…" '
        +     'style="flex:1;min-width:0;background:#0c141a;border:1px solid #2a3a44;color:#cfe6f0;'
        +     'border-radius:6px;padding:6px 8px;font-size:12px;outline:none">'
        +   '<button class="nrfSend" title="Transmite un paquete nRF24 al otro organismo, con independencia de la conducta autónoma" '
        +     'style="background:#3a8c5a;color:#fff;border:none;border-radius:6px;padding:6px 13px;'
        +     'font-size:12px;font-weight:600;cursor:pointer;white-space:nowrap">Enviar ▸</button>'
        + '</div>'
        + '<div class="nrfSendMsg" style="font-size:11px;opacity:.8;margin-top:5px;min-height:14px"></div>';
      const inp=b.querySelector('.nrfMsg'), btn=b.querySelector('.nrfSend'), out=b.querySelector('.nrfSendMsg');
      const enviar=()=>{
        const t=(inp.value||'').trim();
        if(!t){ if(out)out.textContent='escribe un texto primero'; return; }
        btn.disabled=true; if(out)out.textContent='enviando "'+t+'" por el aire…';
        fetch('/nrf/tx',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})})
          .then(r=>r.json())
          .then(j=>{
            if(out) out.textContent = j.ok
              ? ('✓ transmitido ('+(j.via||'radio')+', '+j.len+' B)')
              : (j.error==='firmware_sin_tx_manual'
                  ? '✗ Arduino visible, pero el sketch no implementa TX manual'
                  : '✗ no hay transmisor nRF24 en este organismo');
            if(j.ok)inp.value='';
          })
          .catch(e=>{ if(out)out.textContent='error: '+e; })
          .finally(()=>{ btn.disabled=false; setTimeout(()=>{ if(out)out.textContent=''; },7000); });
      };
      if(btn) btn.onclick=enviar;
      if(inp) inp.onkeydown=(e)=>{ if(e.key==='Enter'){ e.preventDefault(); enviar(); } };
      b._nrf=true;
    }
    const st=b.querySelector('.nrfStatus');
    if(st) st.innerHTML = status;
  }
});
