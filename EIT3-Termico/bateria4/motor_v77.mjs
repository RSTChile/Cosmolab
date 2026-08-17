// Motor Node — extracción limpia de ET3-Termico_v7.7.html (física real
// Stefan-Boltzmann). Validado bit a bit contra el script real corrido en
// sandbox (ver validacion4.md). Reproduce exactamente: pasoFisica (rama sin
// día/noche), computeDaisyworld/abioticTf/ptcResponse con la física v7.7,
// generador de semilla por parada/fase (sembrarFase/claveSemilla), y el
// barrido completo (calibración global + medición, con medirRecuperacion y
// asentarHastaEquilibrio) tal como runSweep() en el HTML.
export const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
export const lerp=(a,b,t)=>a+(b-a)*t;
const gauss=(x,m,s)=>Math.exp(-0.5*Math.pow((x-m)/Math.max(s,1e-6),2));
export const pseudoNoise=(x,y,t)=>{const s=Math.sin(x*12.9898+y*78.233+t*0.021)*43758.5453;return(s-Math.floor(s))*2-1};

// ── física real (v7.7) ──────────────────────────────────────────────────
const DAISY_S=9.17e5, DAISY_SIGMA=5.6704e-5, DAISY_K=DAISY_S/DAISY_SIGMA;
export function temperaturaPlanetariaReal(absorbido){ return Math.pow(DAISY_K*Math.max(absorbido,1e-9),0.25)-273; }
export const DAISY_QPRIME=31.336694844781125;

// ── semilla (mulberry32 + hash FNV-1a, igual que v7.4-v7.6.1) ────────────
export function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
export function claveSemilla(...partes){
  const s=partes.join('|'); let h=0x811c9dc5;
  for(let i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,0x01000193); }
  return h>>>0;
}

export const gridSize=64;
export const HCFG={win:120,bins:24,lo:0,hi:1.2,margin:0.05};

export function shannonEntropy(samples,bins,lo,hi){
  if(samples.length<8) return 0;
  const counts=new Array(bins).fill(0); const span=Math.max(1e-9,hi-lo);
  for(const v of samples){ let idx=Math.floor((v-lo)/span*bins); if(idx<0)idx=0; if(idx>=bins)idx=bins-1; counts[idx]++; }
  const n=samples.length; let H=0;
  for(const c of counts){ if(c>0){ const p=c/n; H-=p*Math.log2(p); } }
  return H;
}
export function entropyLocalAbs(samples,floorSamples,bins,margin){
  if(samples.length<8) return {H:0,lo:0,hi:0,band:0,floor:0};
  let pLo=Infinity,pHi=-Infinity; for(const v of samples){ if(v<pLo)pLo=v; if(v>pHi)pHi=v; }
  let fLo=Infinity,fHi=-Infinity; for(const v of floorSamples){ if(v<fLo)fLo=v; if(v>fHi)fHi=v; }
  const band=pHi-pLo; const floor=(floorSamples.length>=8)?(fHi-fLo):0;
  const center=(pLo+pHi)/2; let width=Math.max(band,floor)*(1+2*margin); if(!(width>0))width=1e-9;
  const lo=center-width/2, hi=center+width/2;
  return {H:shannonEntropy(samples,bins,lo,hi),lo,hi,band,floor};
}
export function entropyAtWidth(samples,bins,width){
  if(samples.length<8||!(width>0)) return 0;
  let lo=Infinity,hi=-Infinity; for(const v of samples){ if(v<lo)lo=v; if(v>hi)hi=v; }
  const c=(lo+hi)/2; return shannonEntropy(samples,bins,c-width/2,c+width/2);
}
export function entropyRel(samples,bins){
  if(samples.length<8) return 0;
  let lo=Infinity,hi=-Infinity; for(const v of samples){ if(v<lo)lo=v; if(v>hi)hi=v; }
  if(hi-lo<1e-9) return 0;
  return shannonEntropy(samples,bins,lo,hi);
}
export function varianzaYAutocorr(xs0){
  const n=xs0.length; if(n<3) return {varianza:0,autocorr1:0};
  let sx=0,sy=0,sxy=0,sxx=0;
  for(let i=0;i<n;i++){ sx+=i; sy+=xs0[i]; sxy+=i*xs0[i]; sxx+=i*i; }
  const den=n*sxx-sx*sx; const b=den!==0?(n*sxy-sx*sy)/den:0; const a=sy/n-b*sx/n;
  const xs=xs0.map((y,i)=>y-(a+b*i));
  let m=0; for(const x of xs) m+=x; m/=n;
  let s2=0; for(const x of xs) s2+=(x-m)*(x-m); s2/=n;
  if(s2<=0) return {varianza:0,autocorr1:0};
  let c=0; for(let i=1;i<n;i++) c+=(xs[i]-m)*(xs[i-1]-m);
  return {varianza:s2, autocorr1:(c/(n-1))/s2};
}

export class Motor {
  constructor(){ this.reset(); }
  reset(){
    this.field=Array.from({length:gridSize},()=>Array.from({length:gridSize},()=>24.5));
    this.aBuf=[]; this.noiseEchoBuf=[]; this._Awin=[];
    this.rngTf=mulberry32(1); this.rngEco=mulberry32(1);
    this.state={
      powerBase:0.47,powerLive:0.47,beta:0.94,sigma:6.8,noise:0.0079,band:1.105,
      luminosity:0.94,tOpt:25,ptcTc:20,ptcSharp:16,minTemp:-6,maxTemp:25,
      ptcR:1,ptcOut:1,mult:0,_A_prev:0,H_at:0,H_noise:0,H_rel:0,absBand:0,absFloor:0,
      bioticTf:24.6,abioticTf:25,bioticFootprint:0,Tf:24.6,Tc:25,Th:28,delta:3.4,
      LF:0,err:0,Lambda:0.1,deltaStruct:0,A_sys_env:0,LF_exp:0,err_exp:0,Lambda_exp:0,
      fertileScore:0,regime:'RÍGIDO',action:'dejar oscilar',omega:0,envTemp:24.6,
      externalStress:0,envDrift:0,black:0.18,white:0.14,bare:0.68,
      albedoBlack:0.25,albedoWhite:0.75,albedoBare:0.5,seed:1,tick:0,step:0,
    };
  }
  resetField(){ this.field=Array.from({length:gridSize},()=>Array.from({length:gridSize},()=>24.5)); }
  // reiniciarSilencioso() — igual que v7.4-v7.6.1, sin cambios con la física nueva
  reiniciarSilencioso(){
    const st=this.state;
    Object.assign(st,{tick:0,step:0,powerLive:st.powerBase,ptcR:1,ptcOut:1,
      Tf:24.6,Tc:25,Th:28,delta:3.4,LF:0,err:0,Lambda:0.1,deltaStruct:0,A_sys_env:0,
      LF_exp:0,err_exp:0,Lambda_exp:0,fertileScore:0,regime:'RÍGIDO',action:'dejar oscilar',
      omega:0,externalStress:0,envDrift:0,black:0.18,white:0.14,bare:0.68});
    this.aBuf=[]; this.noiseEchoBuf=[]; this._Awin=[]; st._A_prev=0;
    this.resetField();
  }
  setSeed(seed){ this.state.seed=seed>>>0; this.sembrarFase('libre',0,'libre'); }
  sembrarFase(eje,punto,fase){
    const st=this.state;
    this.rngTf=mulberry32(claveSemilla(st.seed,eje,punto,fase,'Tf'));
    this.rngEco=mulberry32(claveSemilla(st.seed,eje,punto,fase,'eco'));
  }
  abioticTf(){ return temperaturaPlanetariaReal(this.state.luminosity*(1-this.state.albedoBare)); }
  ptcResponse(temp){
    const st=this.state;
    const ratio=clamp((temp+273)/Math.max(0.1,st.ptcTc+273),0.2,3);
    st.ptcR=Math.max(0.15,Math.pow(ratio,st.ptcSharp));
    st.ptcOut=clamp(1/st.ptcR,0.05,1.2);
    return st.ptcOut;
  }
  computeDaisyworld(){
    const st=this.state;
    st.bare=clamp(1-st.black-st.white,0,1);
    const albedo=st.black*st.albedoBlack+st.white*st.albedoWhite+st.bare*st.albedoBare;
    const absorbed=st.luminosity*(1-albedo);
    const Tplanet=temperaturaPlanetariaReal(absorbed);
    const localBlack=Tplanet+(albedo-st.albedoBlack)*DAISY_QPRIME;
    const localWhite=Tplanet+(albedo-st.albedoWhite)*DAISY_QPRIME;
    const growthBlack=clamp(1-0.003265*Math.pow(st.tOpt-localBlack,2),0,1);
    const growthWhite=clamp(1-0.003265*Math.pow(st.tOpt-localWhite,2),0,1);
    const death=0.28+st.noise*10;
    const spawn=Math.max(0,st.bare);
    st.black=clamp(st.black+(st.black*(growthBlack*spawn-death))*0.08,0,0.9);
    st.white=clamp(st.white+(st.white*(growthWhite*spawn-death))*0.08,0,0.9);
    st.bare=clamp(1-st.black-st.white,0,1);
    const albedo2=st.black*st.albedoBlack+st.white*st.albedoWhite+st.bare*st.albedoBare;
    return {albedo:albedo2, targetTf:temperaturaPlanetariaReal(st.luminosity*(1-albedo2))};
  }
  computeCoupling(Tf,targetTf){ const diff=Math.abs(Tf-targetTf); return Math.max(0,1-diff/8.0); }
  computeDeltaStruct(){
    const f=this.field; let sum=0,sumSq=0,n=0;
    for(let y=0;y<f.length;y++)for(let x=0;x<f[0].length;x++){ const v=f[y][x]; sum+=v; sumSq+=v*v; n++; }
    const mean=sum/n; return Math.sqrt(Math.max(sumSq/n-mean*mean,0));
  }
  evolveField(albedo){
    const st=this.state, f=this.field;
    const next=f.map(r=>r.slice()); const cx=gridSize/2,cy=gridSize/2;
    for(let y=0;y<gridSize;y++)for(let x=0;x<gridSize;x++){
      const dx=(x-cx)/cx,dy=(y-cy)/cy,r=Math.sqrt(dx*dx+dy*dy);
      const edge=gauss(r,0.72,0.11+st.band*0.03);
      const daisyMix=st.black*(1-r)+st.white*r;
      const noise=(pseudoNoise(x,y,st.tick+st.seed*1013.9)-0.5)*st.noise*16;
      const target=st.Tf+edge*st.delta*1.8+daisyMix*1.2-albedo*1.8+noise;
      const smooth=clamp(0.05+st.sigma*0.01,0.04,0.16);
      let n=0,c=0;
      for(let oy=-1;oy<=1;oy++)for(let ox=-1;ox<=1;ox++){ const nx=x+ox,ny=y+oy; if(nx>=0&&nx<gridSize&&ny>=0&&ny<gridSize){ n+=f[ny][nx]; c++; } }
      next[y][x]=lerp(n/c,target,smooth);
    }
    this.field=next;
  }
  passiveNoiseSample(){ return this.state.powerBase+(this.rngEco()-0.5)*this.state.noise*10; }
  updateBehavioralEntropy(){
    const st=this.state;
    this.aBuf.push(st.powerLive); if(this.aBuf.length>HCFG.win) this.aBuf.shift();
    this.noiseEchoBuf.push(this.passiveNoiseSample()); if(this.noiseEchoBuf.length>HCFG.win) this.noiseEchoBuf.shift();
    const la=entropyLocalAbs(this.aBuf,this.noiseEchoBuf,HCFG.bins,HCFG.margin);
    st.H_at=la.H; st.H_rel=entropyRel(this.aBuf,HCFG.bins);
    st.H_noise=entropyAtWidth(this.noiseEchoBuf,HCFG.bins,la.hi-la.lo);
    st.absBand=la.band; st.absFloor=la.floor;
  }
  computeLFandErr(){
    const st=this.state;
    const targetPower=st.powerBase*this.ptcResponse(st.Tf);
    st.powerLive=lerp(st.powerLive,targetPower,0.08);
    const deviation=Math.abs(st.powerLive-st.powerBase);
    const inertiaPenalty=Math.exp(-deviation*4);
    st.mult=clamp(deviation*(1-inertiaPenalty),0,1);
    st.LF=st.mult;
    const dA=st.A_sys_env-st._A_prev; st.err=Math.max(0,-dA); st._A_prev=st.A_sys_env;
  }
  errRatePush(A){ this._Awin.push(A); if(this._Awin.length>12) this._Awin.shift(); }
  errRate(){
    if(this._Awin.length<3) return 0;
    let loss=0,c=0;
    for(let i=1;i<this._Awin.length;i++){ const dA=this._Awin[i]-this._Awin[i-1]; if(dA<0) loss+=-dA; c++; }
    return c?loss/c:0;
  }
  stepHeadless(){
    const st=this.state;
    const daisies=this.computeDaisyworld();
    this.computeLFandErr();
    const thermalDrive=8.2*st.powerLive+0.46*(daisies.targetTf-st.Tf);
    const damping=0.09+(1-st.beta)*0.65;
    const stochastic=(this.rngTf()-0.5)*st.noise*14;
    st.Tf=st.Tf+thermalDrive*0.12-damping*(st.Tf-st.tOpt)*0.05+stochastic;
    const edgeBias=(st.black-st.white)*4.2;
    st.Tc=lerp(st.Tc,st.Tf+edgeBias*0.18,0.12);
    st.Th=lerp(st.Th,st.Tf+0.65+st.powerLive*1.8+edgeBias*0.32,0.10);
    st.delta=st.Th-st.Tf;
    st.Lambda=(st.deltaStruct*st.LF)/Math.max(st.err,1e-6)*st.A_sys_env;
    st.Lambda_exp=st.Lambda;
    st.fertileScore=clamp(st.LF*clamp(1-Math.abs(st.delta-1.1)/1.6,0,1)*(0.4+Math.abs(st.black-st.white)+st.Lambda*0.12),0,1);
    this.evolveField(daisies.albedo);
    st.deltaStruct=this.computeDeltaStruct();
    st.A_sys_env=this.computeCoupling(st.Tf,daisies.targetTf);
    st.abioticTf=this.abioticTf();
    st.bioticFootprint=Math.abs(st.Tf-st.abioticTf);
    this.updateBehavioralEntropy();
    st.LF_exp=clamp(Math.abs(st.powerLive-st.powerBase),0,1);
    st.step+=5; st.tick+=1;
  }
  // instantanea/restaurarInstantanea/medirRecuperacion/asentarHastaEquilibrio — v7.6.1, sin cambios
  instantanea(){
    const st=this.state;
    return {black:st.black,white:st.white,bare:st.bare,Tf:st.Tf,Tc:st.Tc,Th:st.Th,delta:st.delta,
      powerLive:st.powerLive,ptcR:st.ptcR,ptcOut:st.ptcOut,campo:this.field.map(f=>f.slice())};
  }
  restaurarInstantanea(s){
    const st=this.state;
    st.black=s.black; st.white=s.white; st.bare=s.bare; st.Tf=s.Tf; st.Tc=s.Tc; st.Th=s.Th; st.delta=s.delta;
    st.powerLive=s.powerLive; st.ptcR=s.ptcR; st.ptcOut=s.ptcOut;
    this.field=s.campo.map(f=>f.slice());
  }
  medirRecuperacion(golpe,tope){
    const REPS=5, UMBRAL=0.2; const st=this.state;
    const base=this.instantanea();
    let suma=0,fallos=0; const reps=[];
    for(let r=0;r<REPS;r++){
      this.restaurarInstantanea(base);
      const bB=base.black;
      st.black=clamp(bB+golpe,0,0.9); st.white=clamp(base.white-golpe*0.5,0,0.9);
      let i=1;
      for(; i<=tope; i++){ this.stepHeadless(); if(Math.abs(st.black-bB)<=golpe*UMBRAL) break; }
      if(i>tope) fallos++;
      reps.push(Math.min(i,tope)); suma+=Math.min(i,tope);
    }
    this.restaurarInstantanea(base);
    const ord=reps.slice().sort((a,b)=>a-b);
    return {pasos:suma/REPS, convergio:fallos===0?1:0, reps, topes:fallos, mediana:ord[Math.floor(REPS/2)]};
  }
  asentarHastaEquilibrio(tope,tol){
    const st=this.state; const V=50, NECESARIAS=3;
    let prev=st.black, quietas=0, pasos=0;
    while(pasos<tope){
      for(let s=0;s<V;s++) this.stepHeadless();
      pasos+=V;
      const d=Math.abs(st.black-prev); prev=st.black;
      if(d<tol){ quietas++; if(quietas>=NECESARIAS) return {pasos,asentado:1}; } else quietas=0;
    }
    return {pasos,asentado:0};
  }
}

// ── barrido completo (igual estructura que runSweep() del HTML) ─────────
export function correrBarridoV77({ seed, modo, axis='luminosity', from, to, steps, settle, measure,
  powerBase, beta, sigma, noise, band, tOpt, ptcTc, ptcSharp }){
  // Topes ORIGINALES (20.000/20.000), iguales a los del archivo real
  // ET3-Termico_v7.7.html — la reducción a 3.000/6.000 investigada en
  // bateria3 fue específica a la física vieja (recta) y NUNCA se aplicó al
  // HTML ni se re-verificó para la física real (Stefan-Boltzmann). Usar acá
  // los valores rebajados sin volver a medir habría sido una suposición, no
  // un dato — por eso este motor, que se valida bit a bit contra el HTML,
  // usa los topes tal cual están en el archivo real.
  const GOLPE_TF=0.03, TOPE_REC=20000, TOPE_EQ=20000, TOL_EQ=GOLPE_TF*0.2/10;
  const bins=HCFG.bins, margin=HCFG.margin;
  const m=new Motor();
  m.setSeed(seed);
  Object.assign(m.state,{powerBase,beta,sigma,noise,band,tOpt,ptcTc,ptcSharp});
  if(modo!=='ninguno') m.reiniciarSilencioso();
  const rows=[];
  let gLo=Infinity,gHi=-Infinity;
  for(let k=0;k<steps;k++){
    const v=from+(to-from)*k/(steps-1);
    if(modo==='parada') m.reiniciarSilencioso();
    m.resetField(); m.aBuf=[]; m.noiseEchoBuf=[]; m.state._A_prev=0; m._Awin=[];
    m.state.luminosity=v;
    m.sembrarFase(axis,k,'calibracion');
    const calSteps=Math.min(80,Math.max(20,Math.round(settle/2)||40));
    for(let s=0;s<calSteps;s++){ m.stepHeadless(); if(s>calSteps*0.5){ if(m.state.powerLive<gLo)gLo=m.state.powerLive; if(m.state.powerLive>gHi)gHi=m.state.powerLive; } }
  }
  if(!isFinite(gLo)||!isFinite(gHi)){ gLo=0; gHi=1; }
  const gMargin=((gHi-gLo)*margin)||1e-3;
  const gLoCal=gLo-gMargin, gHiCal=gHi+gMargin;
  for(let k=0;k<steps;k++){
    const v=from+(to-from)*k/(steps-1);
    if(modo==='parada') m.reiniciarSilencioso();
    m.resetField(); m.aBuf=[]; m.noiseEchoBuf=[]; m.state._A_prev=0; m._Awin=[];
    m.state.luminosity=v;
    m.sembrarFase(axis,k,'preasentamiento');
    const eq=m.asentarHastaEquilibrio(TOPE_EQ,TOL_EQ);
    m.sembrarFase(axis,k,'recuperacion');
    const asent=m.medirRecuperacion(GOLPE_TF,TOPE_REC);
    m.sembrarFase(axis,k,'asentamiento');
    for(let s=0;s<settle;s++){ m.stepHeadless(); m.errRatePush(m.state.A_sys_env); }
    m.sembrarFase(axis,k,'medicion');
    const plS=[],neS=[]; let footSum=0,lamSum=0,aSum=0,errSum=0,c=0;
    for(let s=0;s<measure;s++){
      m.stepHeadless(); m.errRatePush(m.state.A_sys_env);
      const eR=m.errRate();
      const lam=(m.state.deltaStruct*m.state.mult)/Math.max(eR,1e-6)*m.state.A_sys_env;
      plS.push(m.state.powerLive); neS.push(m.passiveNoiseSample());
      footSum+=m.state.bioticFootprint; lamSum+=lam; aSum+=m.state.A_sys_env; errSum+=eR; c++;
    }
    const la=entropyLocalAbs(plS,neS,bins,margin);
    const H_noiseLocal=entropyAtWidth(neS,bins,la.hi-la.lo);
    const H_absGlobal=shannonEntropy(plS,bins,gLoCal,gHiCal);
    const H_noiseGlobal=entropyAtWidth(neS,bins,gHiCal-gLoCal);
    const H_rel=entropyRel(plS,bins);
    let pLo=Infinity,pHi=-Infinity,sum=0; for(const x of plS){ if(x<pLo)pLo=x; if(x>pHi)pHi=x; sum+=x; }
    const plRange=pHi-pLo, plMean=sum/plS.length;
    const distinct=new Set(plS.map(x=>x.toFixed(5))).size;
    const diag=(plRange<=Math.max(la.floor,1e-9))?'banda<=ruido':'banda>ruido';
    const vac=varianzaYAutocorr(plS);
    rows.push({axis,x:v,
      pasos_recuperacion:asent.pasos, convergio:asent.convergio, rec_mediana:asent.mediana,
      rec_topes:asent.topes, rec_reps:asent.reps, asent_pasos:eq.pasos, asent_ok:eq.asentado,
      tasa_recuperacion:asent.mediana>0?1/asent.mediana:0,
      varianza_pl:vac.varianza, autocorr1_pl:vac.autocorr1,
      H_absLocal:la.H, H_noiseLocal, H_absGlobal, H_noiseGlobal, H_rel,
      footprint:footSum/c, Lambda:lamSum/c, A_sys_env:aSum/c, err_rate:errSum/c,
      powerLive:plMean, plRange, plBand:la.band, noiseFloor:la.floor, distinct,
      mult:m.state.mult, diag, ptcSat:(m.state.ptcOut>=1.2-1e-9||m.state.ptcOut<=0.05+1e-9)?1:0,
    });
  }
  return rows;
}
