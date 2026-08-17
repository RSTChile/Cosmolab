window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push({
  id:'localizacion',tit:'Localización / GPS',w:5,h:8,render:(b,r,bf)=>{
    const lat=Number(r.gps_lat), lon=Number(r.gps_lon), fix=Number(r.gps_fix)>=0.5;
    const ok=fix&&isFinite(lat)&&isFinite(lon);
    const key=ok?lat.toFixed(5)+','+lon.toFixed(5):'sin_fix';
    if(b._locKey!==key){
      b._locKey=key;
      b.innerHTML=`<div class="obsmap"><div class="obsmapPin">+</div>${ok?`<iframe title="mapa GPS" loading="lazy" src="https://www.openstreetmap.org/export/embed.html?bbox=${(lon-0.01).toFixed(6)}%2C${(lat-0.01).toFixed(6)}%2C${(lon+0.01).toFixed(6)}%2C${(lat+0.01).toFixed(6)}&layer=mapnik&marker=${lat.toFixed(6)}%2C${lon.toFixed(6)}"></iframe>`:'<div class="obsmapEmpty">sin fix GPS</div>'}</div><div class="obsLocRows"></div>`;
    }
    const rows=b.querySelector('.obsLocRows');
    // deriva del reloj interno vs el reloj del cielo (PPS): +adelantado / -atrasado / 0 en sincronía
    const der=Number(r.loc_pps_deriva)||0;
    const derTxt=(der>0?'+':'')+der.toFixed(3)+(Math.abs(der)<1e-6?' (en sincronía)':(der>0?' (adelantado)':' (atrasado)'));
    if(rows)rows.innerHTML=
      // --- el SENTIDO del organismo (lo que produce el órgano de localización) ---
      '<div class="obsSub">sentido</div>'
     +cjRowT('loc_desplazamiento',cjN(r.loc_desplazamiento,1)+' m','Desplazamiento recorrido')
     +cjRowG('loc_novedad',r.loc_novedad,'Novedad del lugar')+cjGauge(r.loc_novedad,'#b58cff')
     +cjRowG('loc_confianza',r.loc_confianza,'Confianza en la localización')+cjGauge(r.loc_confianza,'#5fd38a')
     +cjRowT('loc_altitud_rel',cjN(r.loc_altitud_rel,1)+' m','Altitud relativa al punto de partida')
     +cjRowT('loc_pps_deriva',derTxt,'Deriva del reloj interno frente al reloj del cielo')
     +cjRowT('loc_vivo',si(r.loc_vivo),'¿El órgano de localización responde?')
      // --- el DATO crudo del sensor (contexto) ---
     +'<div class="obsSub">sensor</div>'
     +cjRowT('gps_fix',ok?'sí':'no','¿El GPS tiene fijación?')
     +cjRowT('gps_lat',ok?`${lat.toFixed(6)}, ${lon.toFixed(6)}`:'—','Coordenadas (latitud, longitud)')
     +cjRowT('gps_sats',cjN(r.gps_sats,0),'Satélites a la vista')
     +cjRowT('gps_hdop',cjN(r.gps_hdop,2),'Dilución horizontal de la precisión')
     +cjRowT('gps_alt',cjN(r.gps_alt,1)+' m','Altitud sobre el nivel del mar')
     +cjRowT('gps_pps_count',cjN(r.gps_pps_count,0)+' pulsos','Pulsos por segundo recibidos')
     +(ok?`<a class="obslink" target="_blank" rel="noopener" href="https://www.openstreetmap.org/?mlat=${lat.toFixed(6)}&mlon=${lon.toFixed(6)}#map=18/${lat.toFixed(6)}/${lon.toFixed(6)}">abrir mapa</a>`:'');
  }
});
