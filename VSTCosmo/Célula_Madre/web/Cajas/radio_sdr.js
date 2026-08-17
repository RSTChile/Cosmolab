window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push({
  id:'radio_sdr',tit:'Radio / SDR',w:6,h:10,render:(b,r,bf)=>{
    // --- espectro instantáneo (lista de potencias por bin de frecuencia) ---
    let esp=r.sdr_espectro||[];
    if(typeof esp==='string'){try{esp=JSON.parse(esp);}catch(e){esp=[];}}
    esp=(esp||[]).map(Number).filter(v=>isFinite(v));
    const f=Number(r.radio_freq_dom_hz)||Number(r.sdr_freq_centro_hz)||0;
    const mhz=f?f/1e6:0;
    const fmin=Number(r.sdr_freq_min_hz)||0, fmax=Number(r.sdr_freq_max_hz)||0;

    // --- construir el DOM UNA sola vez; el canvas del analizador PERSISTE ---
    if(!b._sp){
      b.innerHTML=
        '<div class="obsSub">espectro sintonizado — frecuencia →</div>'
       +'<canvas class="obsSpectrum" width="512" height="150" '
       +'style="width:100%;height:150px;border-radius:6px;background:#04121a;display:block"></canvas>'
       +'<div class="obsWfAxis" style="display:flex;justify-content:space-between;font-size:10px;opacity:.6;margin:2px 1px 6px">'
       +'<span class="wfLo">—</span><span class="wfMid">espectro</span><span class="wfHi">—</span></div>'
       +'<div class="obsRadioRows"></div>'
       +'<div style="display:flex;align-items:center;gap:8px;margin-top:9px">'
       +'<button class="obsRadioBtn" title="Desatasca el SDR (reset del receptor), con independencia de la conducta del organismo" '
       +'style="background:#0e9068;color:#fff;border:none;border-radius:6px;padding:6px 12px;font-size:12px;font-weight:600;cursor:pointer">'
       +'⟳ Reactivar radio</button>'
       +'<span class="obsRadioMsg" style="font-size:11px;opacity:.75"></span></div>';
      const cv=b.querySelector('.obsSpectrum');
      b._sp={cv:cv, ctx:cv.getContext('2d'), W:cv.width, H:cv.height};
      // --- botón: reactivar la escucha del SDR a demanda (independiente de lo programado) ---
      const btn=b.querySelector('.obsRadioBtn'), msg=b.querySelector('.obsRadioMsg');
      if(btn) btn.onclick=()=>{
        const t0=btn.textContent; btn.disabled=true; btn.textContent='⟳ reactivando…';
        if(msg)msg.textContent='reseteando el receptor…';
        fetch('/radio/reactivar',{method:'POST'})
          .then(r=>r.json())
          .then(j=>{ if(msg)msg.textContent=(j.acciones||['ok']).join(' · '); })
          .catch(e=>{ if(msg)msg.textContent='error: '+e; })
          .finally(()=>{ setTimeout(()=>{ btn.disabled=false; btn.textContent=t0; if(msg)msg.textContent=''; },10000); });
      };
    }
    const sp=b._sp, ctx=sp.ctx, W=sp.W, H=sp.H;

    // --- redibujar TODO el espectro en cada refresco (rápido, tipo analizador) ---
    ctx.fillStyle='#04121a'; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle='rgba(120,160,180,.10)'; ctx.lineWidth=1;           // rejilla tenue
    for(let gy=1;gy<4;gy++){const y=H*gy/4; ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    if(esp.length){
      let mx=0,pk=0; for(let i=0;i<esp.length;i++){if(esp[i]>mx){mx=esp[i];pk=i;}}   // pico = freq dominante
      const inv=mx>0?1/mx:0;
      const yOf=x=>{const idx=Math.min(esp.length-1,Math.floor(x*esp.length/W));return H-Math.max(0,Math.min(1,esp[idx]*inv))*(H-2)-1;};
      // relleno bajo la curva (degradado)
      const grad=ctx.createLinearGradient(0,0,0,H);
      grad.addColorStop(0,'rgba(100,240,200,.55)'); grad.addColorStop(1,'rgba(100,240,200,.04)');
      ctx.beginPath(); ctx.moveTo(0,H);
      for(let x=0;x<W;x++) ctx.lineTo(x,yOf(x));
      ctx.lineTo(W,H); ctx.closePath(); ctx.fillStyle=grad; ctx.fill();
      // línea del espectro
      ctx.beginPath();
      for(let x=0;x<W;x++){const y=yOf(x); if(x===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}
      ctx.strokeStyle='#64f0c8'; ctx.lineWidth=1.5; ctx.stroke();
      // marca vertical del pico (la frecuencia que domina ahora)
      const px=Math.floor(pk*W/esp.length);
      ctx.strokeStyle='rgba(255,140,107,.85)'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,H); ctx.stroke();
    } else {
      ctx.fillStyle='rgba(180,200,210,.45)'; ctx.font='12px system-ui'; ctx.textAlign='center';
      ctx.fillText('esperando espectro del SDR…',W/2,H/2);
    }

    // ejes de frecuencia bajo el canvas
    if(fmin&&fmax){
      const lo=b.querySelector('.wfLo'), hi=b.querySelector('.wfHi'), mid=b.querySelector('.wfMid');
      if(lo)lo.textContent=(fmin/1e6).toFixed(1)+' MHz';
      if(hi)hi.textContent=(fmax/1e6).toFixed(1)+' MHz';
      if(mid)mid.textContent=mhz?('▲ '+mhz.toFixed(3)+' MHz'):'espectro';
    }

    // --- métricas del órgano (el SENTIDO: estructura/novedad/saliencia, nunca potencia sola) ---
    const rows=b.querySelector('.obsRadioRows');
    if(rows)rows.innerHTML=
      cjRow('SDR vivo',si(r.sdr_vivo))+cjRow('radio viva',si(r.radio_vivo))
     +cjRow('frecuencia dominante',mhz?mhz.toFixed(3)+' MHz':'—')
     +'<div class="obsSub">sentido</div>'
     +cjRow('estructura',cjN(r.radio_estructura,3))+cjGauge(r.radio_estructura,'#6db6ff')
     +cjRow('novedad',cjN(r.radio_novedad,3))+cjGauge(r.radio_novedad,'#b58cff')
     +cjRow('saliencia',cjN(r.radio_saliencia,3))+cjGauge(r.radio_saliencia,'#ff8c6b')
     +cjRow('potencia total',cjN(r.radio_potencia_total,4))+cjGauge(Math.min(1,(Number(r.radio_potencia_total)||0)*3),'#e8b86d')
     +cjRow('bandas',cjN(r.radio_n_bandas,0));
  }
});
