// Réplica EXACTA de la dinámica+layout de cg002_genesis.html (sin WebGL) para medir anisotropía y vacíos.
const K=8,ETA=0.05,MU=0.01,KAPPA_S=1e-6,S0=1.0,S_BAND=8.0;
const K_COUPLE=0.9,K_BASE=1.4,K_CENTER=0.015,DAMP=0.86,DT=0.18;
const N=1000, SEED=3, THETA=0.3, STEPS=260;
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const rnd=mulberry32((SEED*2654435761)>>>0);
const omega=new Int32Array(N),S=new Float64Array(N),alive=new Uint8Array(N);
const px=new Float64Array(N),py=new Float64Array(N),pz=new Float64Array(N);
const vx=new Float64Array(N),vy=new Float64Array(N),vz=new Float64Array(N);
const W=new Float64Array(N*N),dSs=new Float64Array(N),axx=new Float64Array(N),ayy=new Float64Array(N),azz=new Float64Array(N);
for(let i=0;i<N;i++){omega[i]=Math.floor(rnd()*K);S[i]=S0;alive[i]=1;px[i]=(rnd()-.5)*.05;py[i]=(rnd()-.5)*.05;pz[i]=(rnd()-.5)*.05;}
let wscaleEMA=1;const sat=s=>s/(1+s/S_BAND);const tpk=2*Math.PI/K;
for(let s=0;s<STEPS;s++){
  dSs.fill(0);axx.fill(0);ayy.fill(0);azz.fill(0);
  for(let i=0;i<N;i++)if(alive[i])S[i]*=(1-MU);
  let maxW=1e-9;const ws=wscaleEMA+1e-9;
  for(let i=0;i<N;i++){if(!alive[i])continue;const oi=omega[i],xi=px[i],yi=py[i],zi=pz[i],Si=S[i];
    for(let j=i+1;j<N;j++){if(!alive[j])continue;
      const a=tpk*(oi-omega[j]),gi=Math.cos(a+THETA),gj=Math.cos(-a+THETA),mag=Math.sqrt(sat(Si)*sat(S[j]));
      const fi=ETA*gi*mag,fj=ETA*gj*mag;dSs[i]+=fi;dSs[j]+=fj;
      const idx=i*N+j,wij=W[idx]+fi;W[idx]=wij;W[j*N+i]+=fj;const aw=Math.abs(wij);if(aw>maxW)maxW=aw;
      let dx=xi-px[j],dy=yi-py[j],dz=zi-pz[j],dist=Math.sqrt(dx*dx+dy*dy+dz*dz)+1e-4,inv=1/dist;
      const ux=dx*inv,uy=dy*inv,uz=dz*inv,w=Math.tanh(wij/ws),fc=-w*K_COUPLE,fb=K_BASE/(dist*dist),li=fc+fb,lj=fc-fb;
      axx[i]+=ux*li;ayy[i]+=uy*li;azz[i]+=uz*li;axx[j]+=ux*lj;ayy[j]+=uy*lj;azz[j]+=uz*lj;}}
  wscaleEMA=0.9*wscaleEMA+0.1*maxW;
  for(let i=0;i<N;i++){if(!alive[i])continue;S[i]+=dSs[i];if(S[i]<0)S[i]=0;if(S[i]<=KAPPA_S){alive[i]=0;continue;}
    let ax=axx[i]-K_CENTER*px[i],ay=ayy[i]-K_CENTER*py[i],az=azz[i]-K_CENTER*pz[i];
    vx[i]=(vx[i]+ax*DT)*DAMP;vy[i]=(vy[i]+ay*DT)*DAMP;vz[i]=(vz[i]+az*DT)*DAMP;
    let vm=Math.hypot(vx[i],vy[i],vz[i]);if(vm>3){const k=3/vm;vx[i]*=k;vy[i]*=k;vz[i]*=k;}
    px[i]+=vx[i]*DT;py[i]+=vy[i]*DT;pz[i]+=vz[i]*DT;}}
// ----- análisis -----
let idx=[];for(let i=0;i<N;i++)if(alive[i])idx.push(i);const n=idx.length;
let cx=0,cy=0,cz=0;for(const i of idx){cx+=px[i];cy+=py[i];cz+=pz[i];}cx/=n;cy/=n;cz/=n;
let xx=0,yy=0,zz=0,xy=0,xz=0,yz=0;for(const i of idx){const x=px[i]-cx,y=py[i]-cy,z=pz[i]-cz;xx+=x*x;yy+=y*y;zz+=z*z;xy+=x*y;xz+=x*z;yz+=y*z;}
xx/=n;yy/=n;zz/=n;xy/=n;xz/=n;yz/=n;
function eig3(a,b,c,d,e,f){const p1=d*d+e*e+f*f;if(p1<1e-12)return [a,b,c].sort((x,y)=>y-x);const q=(a+b+c)/3,p2=(a-q)**2+(b-q)**2+(c-q)**2+2*p1,p=Math.sqrt(p2/6);const detB=((a-q)/p)*(((b-q)/p)*((c-q)/p)-(f/p)**2)-(d/p)*((d/p)*((c-q)/p)-(f/p)*(e/p))+(e/p)*((d/p)*(f/p)-((b-q)/p)*(e/p));let r=Math.max(-1,Math.min(1,detB/2)),phi=Math.acos(r)/3;const e1=q+2*p*Math.cos(phi),e3=q+2*p*Math.cos(phi+2*Math.PI/3);return [e1,3*q-e1-e3,e3].sort((x,y)=>y-x);}
const ev=eig3(xx,yy,zz,xy,xz,yz);
const dimEff=(ev[0]+ev[1]+ev[2])**2/(ev[0]**2+ev[1]**2+ev[2]**2);
// vacíos: rejilla 16^3 sobre la extensión; fracción de celdas vacías (real vs Poisson uniforme)
let ext=0;for(const i of idx){ext=Math.max(ext,Math.abs(px[i]-cx),Math.abs(py[i]-cy),Math.abs(pz[i]-cz));}
const G=16,cells=new Int32Array(G*G*G);const cc=v=>Math.min(G-1,Math.max(0,Math.floor((v+ext)/(2*ext)*G)));
for(const i of idx){cells[cc(px[i]-cx)*G*G+cc(py[i]-cy)*G+cc(pz[i]-cz)]++;}
let occ=0,mx=0;for(const v of cells){if(v>0)occ++;if(v>mx)mx=v;}
const fracVacia=1-occ/(G*G*G);
// densidad media por celda ocupada y "clumping" (varianza/media de ocupación)
let mean=n/(G*G*G),s2=0;for(const v of cells)s2+=(v-mean)*(v-mean);s2/=(G*G*G);const clump=s2/mean; // =1 Poisson, >1 agrupado
// Poisson esperado: fraccion vacia = exp(-mean)
const fracVaciaPoisson=Math.exp(-mean);
console.log(`N vivos: ${n}/${N}`);
console.log(`autovalores covarianza (λ1≥λ2≥λ3): ${ev.map(x=>x.toFixed(2)).join(', ')}`);
console.log(`anisotropía λ3/λ1: ${(ev[2]/ev[0]).toFixed(3)}  (1.0 = esfera isótropa; <1 = NO simétrico)`);
console.log(`dimensión efectiva: ${dimEff.toFixed(2)}`);
console.log(`celdas vacías (vacíos/zonas oscuras): ${(fracVacia*100).toFixed(1)}%  | esperado si fuera uniforme(Poisson): ${(fracVaciaPoisson*100).toFixed(1)}%`);
console.log(`clumping (var/media ocupación): ${clump.toFixed(2)}  (1.0 = uniforme aleatorio; >1 = agrupado, hay estructura)`);
