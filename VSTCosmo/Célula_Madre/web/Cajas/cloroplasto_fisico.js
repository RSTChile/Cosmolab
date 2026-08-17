window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push({
  id:'cloroplasto_fisico',tit:'Cloroplasto físico',w:4,h:6,render:(b,r,bf)=>{
    const luz=Math.max(0,Math.min(1,Number(r.foto_luz_norm)||0));
    const fase=luz>=0.35?'día':(luz<=0.08?'noche':'umbral');
    const astro=fase==='día'?'sol':(fase==='noche'?'luna':'crepúsculo');
    b.innerHTML=`<div class="obsAstro ${astro}"><i></i><span>${fase}</span></div>`
     +cjRowT('foto_v_panel',cjN(r.foto_v_panel,3)+' V','Tensión del panel / fuente')+cjGauge(Math.min(1,(Number(r.foto_v_panel)||0)/5),'#e8b86d')
     +cjRowT('foto_v_lipo',cjN(r.foto_v_lipo,3)+' V','Tensión de la batería LiPo')+cjGauge(Math.min(1,(Number(r.foto_v_lipo)||0)/4.2),'#5fd38a')
     +cjRowG('foto_luz_norm',luz,'Luz normalizada que recibe')+cjGauge(luz,'#ffd166')+cjSpark(bf.foto_luz_norm,'#ffd166')
     +cjRowT('foto_adc_a0',cjN(r.foto_adc_a0,0)+' · '+cjN(r.foto_adc_a1,0),'Lectura cruda del conversor (A0 · A1)')
     +cjRowT('foto_sensor_vivo',si(r.foto_sensor_vivo),'¿El sensor responde?');
  }
});
