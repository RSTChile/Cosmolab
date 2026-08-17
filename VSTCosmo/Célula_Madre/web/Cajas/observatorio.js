// Observatorio de organelos: motor liviano. Las cajas viven como archivos independientes en /Cajas.
(function(){
  window.OBS_CAJAS = window.OBS_CAJAS || [];
  window.cjN=(x,d=2)=>{x=Number(x);return isFinite(x)?x.toFixed(d):'—';};
  window.cjRow=(k,v)=>`<div class="obsrow"><span class="obsk">${k}</span><span class="obsv">${v}</span></div>`;
  window.cjGauge=(v,col='#5fd38a')=>{v=Math.max(0,Math.min(1,Number(v)||0));return `<div class="obsgauge"><i style="width:${(v*100).toFixed(0)}%;background:${col}"></i></div>`;};
  window.cjBip=(v,col='#6db6ff')=>{v=Math.max(-1,Math.min(1,Number(v)||0));const w=Math.abs(v)*50,left=v>=0?50:50-w;return `<div class="obsgauge"><i style="margin-left:${left}%;width:${w}%;background:${col}"></i></div>`;};
  window.cjSpark=(arr,col='#6db6ff')=>{arr=(arr||[]).map(Number).filter(isFinite).slice(-60);if(arr.length<2)return '';const mn=Math.min(...arr),mx=Math.max(...arr),rg=(mx-mn)||1;const pts=arr.map((y,i)=>`${(i/(arr.length-1)*100).toFixed(1)},${(17-(y-mn)/rg*15).toFixed(1)}`).join(' ');return `<svg viewBox="0 0 100 18" preserveAspectRatio="none" style="width:100%;height:22px;margin:1px 0 3px"><polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.3"/></svg>`;};
  window.si=(x)=>Number(x)>=0.5?'sí':'no';

  const loadScript=(src)=>new Promise((resolve,reject)=>{
    const s=document.createElement('script');
    s.src=src;
    s.onload=resolve;
    s.onerror=()=>reject(new Error(src));
    document.head.appendChild(s);
  });

  async function cargarCajas(){
    try{
      const r=await fetch('/Cajas/manifest.json',{cache:'no-cache'});
      if(!r.ok) throw new Error('HTTP '+r.status);
      const m=await r.json();
      if(!Array.isArray(m)) throw new Error('manifest no es array');
      for(const item of m){
        if(!item || !item.archivo) continue;
        try{
          await loadScript('/Cajas/'+item.archivo+'?v='+Date.now());
        }catch(err){
          console.warn('[obs] no se cargó caja', item.archivo, err);
        }
      }
    }catch(e){
      console.warn('No se pudo cargar /Cajas/manifest.json',e);
    }
  }

  function waitForGridStack(timeoutMs){
    return new Promise((resolve)=>{
      if(window.GridStack) return resolve(true);
      const t0=Date.now();
      const id=setInterval(()=>{
        if(window.GridStack){ clearInterval(id); resolve(true); }
        else if(Date.now()-t0>timeoutMs){ clearInterval(id); resolve(false); }
      },40);
    });
  }

  async function initObs(){
    await cargarCajas();
    const okGs = await waitForGridStack(4000);
    const CAJAS=window.OBS_CAJAS || [];
    const ORG=(window.ORG_ID || document.title || 'organismo').replace(/\W+/g,'_');
    const LSKEY='obs_v1_'+ORG;
    let gs=null, activas=new Map(), editando=false, paletaAbierta=false, cargando=false;

    // Una caja que se cae NO puede llevarse a las demás: cada render va aislado. (8-ago-2026,
    // al pasar todas las cajas por el glosario: si un helper no cargara, antes se apagaba
    // el tablero entero desde la primera caja que fallara.)
    function pinta(c,b,row){
      try{ c.render(b,row,buf); }
      catch(err){ console.warn('[obs] caja falló al pintar:',c.id,err); }
    }
    function renderUna(c){
      const b=document.getElementById('obsbody_'+c.id);
      if(b&&window._ultimaFila&&typeof c.render==='function') pinta(c,b,window._ultimaFila);
    }
    window.renderCajas=function(row){
      activas.forEach(c=>{
        const b=document.getElementById('obsbody_'+c.id);
        if(b&&typeof c.render==='function') pinta(c,b,row);
      });
    };
    function guardar(){
      if(cargando||!gs) return;
      try{ localStorage.setItem(LSKEY, JSON.stringify(gs.save(false))); }catch(e){}
    }
    function chequearVacio(){
      const e=document.getElementById('obsVacio');
      if(e) e.style.display=activas.size?'none':'';
    }
    function refrescarCatalogo(){
      const cat=document.getElementById('obsCatalogo');
      if(!cat) return;
      cat.innerHTML='';
      CAJAS.forEach(c=>{
        const chip=document.createElement('span');
        chip.className='obschip'+(activas.has(c.id)?' puesta':'');
        chip.textContent=c.tit;
        if(!activas.has(c.id)) chip.onclick=()=>addCaja(c.id);
        cat.appendChild(chip);
      });
    }
    function addCaja(cid,pos){
      if(activas.has(cid)) return false;
      const c=CAJAS.find(x=>x.id===cid);
      if(!c) return false;
      if(!gs) return false;
      const content=`<div class="obscaja"><h3>${c.tit}<span class="obshelp" data-help="${cid}" title="¿Qué es esto?">?</span><span class="obsdel" data-cid="${cid}" title="quitar">✕</span></h3><div class="obsbody" id="obsbody_${cid}">—</div></div>`;
      // GridStack 10: no pasar x/y undefined (rompe autoPosition y el layout queda vacío).
      const opts={
        id: cid,
        content: content,
        w: (pos && Number.isFinite(+pos.w) ? +pos.w : (c.w||4)),
        h: (pos && Number.isFinite(+pos.h) ? +pos.h : (c.h||3)),
      };
      if(pos && Number.isFinite(+pos.x) && Number.isFinite(+pos.y)){
        opts.x=+pos.x;
        opts.y=+pos.y;
      }else{
        opts.autoPosition=true;
      }
      try{
        gs.addWidget(opts);
      }catch(err){
        console.warn('[obs] addWidget falló', cid, err);
        return false;
      }
      activas.set(cid,c);
      renderUna(c);
      refrescarCatalogo();
      chequearVacio();
      guardar();
      return true;
    }
    function desplegarTodas(){
      CAJAS.forEach(c=>addCaja(c.id));
    }
    function setEdit(on){
      editando=on;
      const zona=document.getElementById('obsZona');
      if(zona) zona.classList.toggle('editando',on);
      if(gs) gs.setStatic(!on);
      const add=document.getElementById('obsAdd');
      const reset=document.getElementById('obsReset');
      const edit=document.getElementById('obsEdit');
      if(add) add.style.display=on?'':'none';
      if(reset) reset.style.display=on?'':'none';
      if(edit) edit.textContent=on?'✓ Listo':'✏️ Editar tablero';
      if(!on){
        paletaAbierta=false;
        const pal=document.getElementById('obsPaleta');
        if(pal) pal.style.display='none';
      }
    }

    if(!okGs || !window.GridStack){
      console.warn('Gridstack no disponible — no se pueden desplegar cajas de organelos');
      const e=document.getElementById('obsVacio');
      if(e){
        e.style.display='';
        e.innerHTML='No se pudo cargar el tablero (GridStack). Recarga la página o reinstala ANIMA (falta /Cajas/vendor/gridstack-all.js).';
      }
      return;
    }
    if(!CAJAS.length){
      console.warn('[obs] manifest sin cajas cargadas');
      const e=document.getElementById('obsVacio');
      if(e){
        e.style.display='';
        e.textContent='No hay cajas en el manifest. Revisa /Cajas/manifest.json';
      }
      return;
    }

    const gridEl=document.getElementById('obsGrid');
    if(!gridEl){
      console.warn('[obs] #obsGrid no existe en el DOM');
      return;
    }

    gs=GridStack.init({
      column:12,
      cellHeight:60,
      margin:6,
      float:true,
      staticGrid:true,
      handle:'.obscaja h3',
      animate:false,
    }, '#obsGrid');

    gs.on('change',guardar);

    // "?" de la caja: en fase de CAPTURA sobre document, para que corra ANTES de que
    // GridStack (que usa la h3 como handle de arrastre, con el "?" dentro) consuma el evento.
    if(!window.__obsHelpBound){
      window.__obsHelpBound=true;
      document.addEventListener('click',e=>{
        const help=e.target.closest && e.target.closest('.obshelp');
        if(!help) return;
        e.preventDefault(); e.stopPropagation();
        const cid=help.dataset.help;
        const caja=(window.OBS_CAJAS||[]).find(x=>x.id===cid);
        const prev=document.getElementById('obs-ayuda-pop');
        const same=prev && prev.dataset.cid===cid;
        if(prev) prev.remove();
        if(same) return;   // toggle: segundo click cierra
        const pop=document.createElement('div');
        pop.id='obs-ayuda-pop'; pop.dataset.cid=cid;
        pop.style.cssText='position:fixed;z-index:99999;max-width:340px;background:#0e1a28;border:1px solid #2a3a4e;'
          +'border-radius:6px;padding:10px 12px;font-size:12px;color:#cfe0f0;line-height:1.45;box-shadow:0 6px 24px rgba(0,0,0,.5)';
        const rc=help.getBoundingClientRect();
        pop.style.left=Math.max(8,Math.min(rc.left, window.innerWidth-352))+'px';
        pop.style.top=(rc.bottom+6)+'px';
        pop.innerHTML='<div style="font-weight:bold;color:#7ddefa;margin-bottom:5px">'+((caja&&caja.tit)||cid)+'</div>'
          +'<div>'+((caja&&caja.ayuda)||'Sin descripción para esta caja.')+'</div>'
          +'<div style="text-align:right;margin-top:7px"><span class="obsayuda-x" style="cursor:pointer;color:#7ddefa">cerrar ✕</span></div>';
        pop.querySelector('.obsayuda-x').onclick=()=>pop.remove();
        document.body.appendChild(pop);
      }, true);
    }

    gridEl.addEventListener('click',e=>{
      // (legacy) "?" también aquí por compatibilidad; el handler de captura ya lo maneja.
      const help=e.target.closest('.obshelp');
      if(help){
        const cid=help.dataset.help;
        const caja=(window.OBS_CAJAS||[]).find(x=>x.id===cid);
        const prev=document.getElementById('obs-ayuda-pop');
        const same=prev && prev.dataset.cid===cid;
        if(prev) prev.remove();
        if(same) return;   // toggle: segundo click cierra
        const pop=document.createElement('div');
        pop.id='obs-ayuda-pop'; pop.dataset.cid=cid;
        pop.style.cssText='position:fixed;z-index:99999;max-width:340px;background:#0e1a28;border:1px solid #2a3a4e;'
          +'border-radius:6px;padding:10px 12px;font-size:12px;color:#cfe0f0;line-height:1.45;box-shadow:0 6px 24px rgba(0,0,0,.5)';
        const rc=help.getBoundingClientRect();
        pop.style.left=Math.max(8,Math.min(rc.left, window.innerWidth-352))+'px';
        pop.style.top=(rc.bottom+6)+'px';
        pop.innerHTML='<div style="font-weight:bold;color:#7ddefa;margin-bottom:5px">'+((caja&&caja.tit)||cid)+'</div>'
          +'<div>'+((caja&&caja.ayuda)||'Sin descripción para esta caja.')+'</div>'
          +'<div style="text-align:right;margin-top:7px"><span class="obsayuda-x" style="cursor:pointer;color:#7ddefa">cerrar ✕</span></div>';
        pop.querySelector('.obsayuda-x').onclick=()=>pop.remove();
        document.body.appendChild(pop);
        return;
      }
      const d=e.target.closest('.obsdel');
      if(!d) return;
      const cid=d.dataset.cid;
      const item=d.closest('.grid-stack-item');
      if(item) gs.removeWidget(item);
      activas.delete(cid);
      refrescarCatalogo();
      chequearVacio();
      guardar();
    });

    const btnEdit=document.getElementById('obsEdit');
    const btnAdd=document.getElementById('obsAdd');
    const btnReset=document.getElementById('obsReset');
    if(btnEdit) btnEdit.onclick=()=>setEdit(!editando);
    if(btnAdd) btnAdd.onclick=()=>{
      paletaAbierta=!paletaAbierta;
      const pal=document.getElementById('obsPaleta');
      if(pal) pal.style.display=paletaAbierta?'':'none';
    };
    if(btnReset) btnReset.onclick=()=>{
      if(!confirm('¿Vaciar el tablero y borrar la disposición guardada de este organismo?')) return;
      gs.removeAll();
      activas.clear();
      try{ localStorage.removeItem(LSKEY); }catch(e){}
      desplegarTodas();
      refrescarCatalogo();
      chequearVacio();
    };

    refrescarCatalogo();
    cargando=true;
    let saved=null;
    try{ saved=JSON.parse(localStorage.getItem(LSKEY)||'null'); }catch(e){ saved=null; }

    if(Array.isArray(saved) && saved.length){
      saved.forEach(n=>{
        if(!n) return;
        const id=n.id || n.i;
        if(id) addCaja(id, n);
      });
      // Si el layout guardado era de otro perfil (p.ej. solo sensores remotos)
      // y no se restauró ninguna caja, desplegar todas las del manifest actual.
      if(!activas.size) desplegarTodas();
    }else{
      // Primera visita: desplegar todas las cajas del manifest.
      desplegarTodas();
    }
    cargando=false;
    chequearVacio();

    // Forzar recálculo de tamaños (útil si el portal ocultó/mostró el wrap).
    try{
      if(typeof gs.compact==='function') gs.compact();
      if(typeof gs.onResize==='function') gs.onResize();
      else window.dispatchEvent(new Event('resize'));
    }catch(e){}

    // Cuando el portal vuelve a la pestaña Organismo, re-layout del grid.
    window.addEventListener('anima-portal-view', (ev)=>{
      if(ev && ev.detail && ev.detail.view==='organismo'){
        try{
          if(gs && typeof gs.onResize==='function') gs.onResize();
          else window.dispatchEvent(new Event('resize'));
        }catch(e){}
      }
    });
  }

  if(document.readyState!=='loading') initObs();
  else document.addEventListener('DOMContentLoaded', initObs);
})();
