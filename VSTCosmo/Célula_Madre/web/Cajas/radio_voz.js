// Caja: VOZ DE RADIO (TX) — el fonador de radio (OrganoFonadorRadio).
// Observa las columnas radiotx_* que el órgano pone en la fila. Es la boca de radio del
// organismo (aparte del oído radio_sdr): emite el indicativo CD3LZK + su voz de pitos por RF.
// Latente en organismos sin el órgano (muestra "—"); cobra vida en A/E con ANIMA_RADIO_TX_*.
window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push({
  id:'radio_voz', tit:'Voz de radio (TX)', w:4, h:5, render:(b,r,bf)=>{
    const activo = Number(r.radiotx_activo)>0.5;
    const vivo   = Number(r.radiotx_vivo)>0.5;
    const emit   = Number(r.radiotx_emitiendo)>0.5;
    const f  = Number(r.radiotx_freq_hz)||0;  const mhz = f ? (f/1e6).toFixed(3) : null;
    const af = Number(r.radiotx_af_hz)||0;

    // --- indicador EN EL AIRE (rojo pulsante mientras emite; ámbar listo; gris latente) ---
    let air;
    if(emit){
      air = '<div style="text-align:center;padding:9px;margin-bottom:8px;border-radius:8px;'
          + 'background:#3a0d10;border:1px solid #ff5a4a;color:#ff8c7a;font-weight:700;'
          + 'letter-spacing:.5px;box-shadow:0 0 14px #ff5a4a55">● EN EL AIRE'
          + (af? ' · '+af.toFixed(0)+' Hz':'') + '</div>';
    } else if(activo && vivo){
      air = '<div style="text-align:center;padding:9px;margin-bottom:8px;border-radius:8px;'
          + 'background:#1a2410;border:1px solid #7a8c3a;color:#c9d38a">○ listo (en silencio)</div>';
    } else if(activo){
      air = '<div style="text-align:center;padding:9px;margin-bottom:8px;border-radius:8px;'
          + 'background:#241a10;border:1px solid #8c6a3a;color:#d3b58a">brazo de hardware no disponible</div>';
    } else {
      air = '<div style="text-align:center;padding:9px;margin-bottom:8px;border-radius:8px;'
          + 'background:#12181c;border:1px solid #2a3a44;color:#6a7a84">voz de radio latente</div>';
    }

    // spark del tono emitido (historia), si el motor lo trae por columna
    const spark = (bf && bf.radiotx_af_hz) ? cjSpark(bf.radiotx_af_hz,'#ff8c6b') : '';

    b.innerHTML = air
      + '<div class="obsSub">estación</div>'
      + cjRow('indicativo','CD3LZK')
      + cjRow('frecuencia', mhz? (mhz+' MHz') : '—')
      + cjRow('tono actual', af? (af.toFixed(0)+' Hz') : '—')
      + spark
      + '<div class="obsSub">estado del órgano</div>'
      + cjRow('órgano activo', si(r.radiotx_activo))
      + cjRow('brazo (hardware)', si(r.radiotx_vivo))
      + cjRow('emitiendo', si(r.radiotx_emitiendo));
  }
});
