window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push({
  id:'vision',tit:'Visión',w:5,h:6,render:(b,r,bf)=>{
    const now=Date.now();
    window.OBS_VISION_FRAMES=window.OBS_VISION_FRAMES||[];
    if(!b._visionInit){b._visionInit=true;b._lastShot=0;}
    if(now-b._lastShot>60000 || window.OBS_VISION_FRAMES.length===0){
      b._lastShot=now;
      window.OBS_VISION_FRAMES.push({t:now,src:'/cam/capture.jpg?ts='+now});
      window.OBS_VISION_FRAMES=window.OBS_VISION_FRAMES.slice(-12);
    }
    const frames=window.OBS_VISION_FRAMES;
    const last=frames[frames.length-1];
    const thumbs=frames.map(f=>`<img src="${f.src}" title="${new Date(f.t).toLocaleTimeString()}" onerror="this.classList.add('err')">`).join('');
    // tono medio de la retina de cámara (promedio r/g/b) — proxy de intensidad, ya que el órgano no emite 'intensidad'
    const tr=Number(r.vis_cam_tono_r), tg=Number(r.vis_cam_tono_g), tb=Number(r.vis_cam_tono_b);
    const tono=[tr,tg,tb].filter(isFinite);
    const tonoMed=tono.length?tono.reduce((a,c)=>a+c,0)/tono.length:NaN;
    b.innerHTML=`<div class="obsVisionMain">${last?`<img src="${last.src}" onerror="this.classList.add('err')">`:'<span>sin captura</span>'}</div><div class="obsThumbs">${thumbs}</div>`
      // --- el SENTIDO: lo que el ojo ATIENDE (desviación, no valor absoluto) ---
     +'<div class="obsSub">sentido</div>'
     +cjRow('saliencia',cjN(r.vis_saliencia,3))+cjGauge(r.vis_saliencia,'#ff8c6b')
     +cjRow('novedad',cjN(r.vis_novedad,3))+cjGauge(r.vis_novedad,'#b58cff')
     +cjRow('dominante',r.vis_dominante??'—')
     +cjRow('retinas',cjN(r.vis_n_retinas,0))
      // --- las MAGNITUDES de la retina de cámara ---
     +'<div class="obsSub">retina cámara</div>'
     +cjRow('tono medio',isFinite(tonoMed)?cjN(tonoMed,3):'—')+cjGauge(isFinite(tonoMed)?tonoMed:0,'#6db6ff')
     +cjRow('movimiento',cjN(r.vis_cam_movimiento,3))+cjGauge(r.vis_cam_movimiento,'#ff8c6b')
     +cjRow('contraste',cjN(r.vis_cam_contraste,3))+cjGauge(r.vis_cam_contraste,'#b58cff')
     +cjRow('novedad cámara',cjN(r.vis_cam_novedad,3))
     +cjRow('visión viva',si(r.vis_vivo));
  }
});
