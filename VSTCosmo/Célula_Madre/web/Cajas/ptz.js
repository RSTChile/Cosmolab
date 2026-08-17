// Caja: CÁMARA PTZ — el ojo con cuello (OrganoVisionPTZ).
// Muestra hacia dónde mira la cámara física: pan (izq↔der, sigue la cabeza 3D del
// animalito vía act_orientacion_deg) y tilt (abajo↔arriba, sigue el arousal).
// Lee las columnas ptz_* de la fila. Latente en organismos sin cámara (solo E).
window.OBS_CAJAS = window.OBS_CAJAS || [];
window.OBS_CAJAS.push({
  id:'ptz', tit:'Cámara PTZ (ojo con cuello)', w:4, h:5, render:(b,r,bf)=>{
    const vivo = Number(r.ptz_vivo)>0.5;
    const activo = (r.ptz_pan !== undefined && r.ptz_pan !== null);
    const pan  = Number(r.ptz_pan)||0,  tilt  = Number(r.ptz_tilt)||0;
    const tpan = Number(r.ptz_target_pan)||0, ttilt = Number(r.ptz_target_tilt)||0;
    const pdeg = Number(r.ptz_pan_deg)||0, tdeg = Number(r.ptz_tilt_deg)||0;
    const moving = Number(r.ptz_moving)>0.5;
    const frames = Number(r.ptz_frames)||0;
    const fok = Number(r.ptz_frame_ok)>0.5;

    // posición en el panel 2D (0..100%). pan −1..1 → x ; tilt −1..1 (arriba=arriba)
    const x  = ((pan+1)/2*100).toFixed(1),  y  = ((1-(tilt+1)/2)*100).toFixed(1);
    const tx = ((tpan+1)/2*100).toFixed(1), ty = ((1-(ttilt+1)/2)*100).toFixed(1);

    let banner;
    if(!activo){
      banner = '<div style="text-align:center;padding:9px;margin-bottom:8px;border-radius:8px;'
             + 'background:#12181c;border:1px solid #2a3a44;color:#6a7a84">ojo con cuello latente</div>';
    } else if(moving){
      banner = '<div style="text-align:center;padding:9px;margin-bottom:8px;border-radius:8px;'
             + 'background:#0d2a3a;border:1px solid #4aa0ff;color:#8ac6ff;font-weight:700;'
             + 'box-shadow:0 0 14px #4aa0ff55">◉ girando la mirada</div>';
    } else if(vivo){
      banner = '<div style="text-align:center;padding:9px;margin-bottom:8px;border-radius:8px;'
             + 'background:#12241a;border:1px solid #3a8c5a;color:#8ad3a0">○ mirada quieta</div>';
    } else {
      banner = '<div style="text-align:center;padding:9px;margin-bottom:8px;border-radius:8px;'
             + 'background:#241a10;border:1px solid #8c6a3a;color:#d3b58a">cámara no alcanzable</div>';
    }

    // panel 2D de la mirada (cuadro con punto actual + objetivo tenue)
    const pad =
      '<div style="position:relative;width:100%;aspect-ratio:16/10;border-radius:8px;'
      + 'background:radial-gradient(circle at 50% 50%,#16202a,#0c1218);border:1px solid #2a3a44;'
      + 'margin-bottom:8px;overflow:hidden">'
      + '<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#2a3a4488"></div>'
      + '<div style="position:absolute;top:50%;left:0;right:0;height:1px;background:#2a3a4488"></div>'
      + '<div style="position:absolute;left:'+tx+'%;top:'+ty+'%;width:12px;height:12px;'
      + 'transform:translate(-50%,-50%);border:1px dashed #7a8c9a;border-radius:50%;opacity:.7"></div>'
      + '<div style="position:absolute;left:'+x+'%;top:'+y+'%;width:14px;height:14px;'
      + 'transform:translate(-50%,-50%);border-radius:50%;background:'+(moving?'#4aa0ff':'#8ad3a0')
      + ';box-shadow:0 0 10px '+(moving?'#4aa0ff':'#8ad3a0')+'"></div>'
      + '<div style="position:absolute;left:4px;top:2px;font-size:10px;color:#6a7a84">arriba</div>'
      + '<div style="position:absolute;left:4px;bottom:2px;font-size:10px;color:#6a7a84">abajo</div>'
      + '<div style="position:absolute;right:4px;top:50%;font-size:10px;color:#6a7a84;transform:translateY(-50%)">der</div>'
      + '<div style="position:absolute;left:4px;top:50%;font-size:10px;color:#6a7a84;transform:translateY(-50%)">izq</div>'
      + '</div>';

    // imagen que ve el ojo PTZ (frame RTSP; se recarga al subir ptz_frames)
    const img = frames>0
      ? '<img src="/ptz/capture.jpg?f='+frames+'" alt="ojo PTZ" '
        + 'style="width:100%;display:block;border-radius:8px;margin-bottom:8px;background:#0c1218" '
        + 'onerror="this.style.display=\'none\'">'
      : '';

    b.innerHTML = banner + img + pad
      + '<div class="obsSub">cabeza (siguiendo al animalito)</div>'
      + cjRow('pan (izq↔der)', pdeg.toFixed(1)+'°  →  '+(tpan*90).toFixed(0)+'°')
      + cjRow('tilt (abj↔arr)', tdeg.toFixed(1)+'°  →  '+(ttilt*45).toFixed(0)+'°')
      + cjRow('moviéndose', si(r.ptz_moving))
      + '<div class="obsSub">ojo</div>'
      + cjRow('frames vistos', frames)
      + cjRow('último frame', fok ? 'ok' : '—')
      + cjRow('control vivo', si(r.ptz_vivo));
  }
});
