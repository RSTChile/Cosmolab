const K=8,ETA=0.05,MU=0.01,KAPPA_S=1e-6,S0=1.0,S_BAND=8.0;
const K_COUPLE=0.9,K_BASE=1.4,K_CENTER=0.015,DAMP=0.86,DT=0.18;
function mb(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function eig3(a,b,c,d,e,f){const p1=d*d+e*e+f*f;if(p1<1e-12)return[a,b,c].sort((x,y)=>y-x);const q=(a+b+c)/3,p2=(a-q)**2+(b-q)**2+(c-q)**2+2*p1,p=Math.sqrt(p2/6);const det=((a-q)/p)*(((b-q)/p)*((c-q)/p)-(f/p)**2)-(d/p)*((d/p)*((c-q)/p)-(f/p)*(e/p))+(e/p)*((d/p)*(f/p)-((b-q)/p)*(e/p));let r=Math.max(-1,Math.min(1,det/2)),phi=Math.acos(r)/3;const e1=q+2*p*Math.cos(phi),e3=q+2*p*Math.cos(phi+2*Math.PI/3);return[e1,3*q-e1-e3,e3].sort((x,y)=>y-x);}
function stats(px,py,pz,alive,N){let idx=[];for(let i=0;i<N;i++)if(alive[i])idx.push(i);const n=idx.length;if(n<5)return{n,clump:0,aniso:1};
  let cx=0,cy=0,cz=0;for(const i of idx){cx+=px[i];cy+=py[i];cz+=pz[i];}cx/=n;cy/=n;cz/=n;
  let xx=0,yy=0,zz=0,xy=0,xz=0,yz=0,ext=0;for(const i of idx){const x=px[i]-cx,y=py[i]-cy,z=pz[i]-cz;xx+=x*x;yy+=y*y;zz+=z*z;xy+=x*y;xz+=x*z;yz+=y*z;ext=Math.max(ext,Math.abs(x),Math.abs(y),Math.abs(z));}
  xx/=n;yy/=n;zz/=n;xy/=n;xz/=n;yz/=n;const ev=eig3(xx,yy,zz,xy,xz,yz);
  const G=16,cells=new Int32Array(G*G*G),cc=v=>Math.min(G-1,Math.max(0,Math.floor((v+ext)/(2*ext+1e-9)*G)));
  for(const i of idx)cells[cc(px[i]-cx)*G*G+cc(py[i]-cy)*G+cc(pz[i]-cz)]++;
  let mean=n/(G*G*G),s2=0;for(const v of cells)s2+=(v-mean)**2;s2/=(G*G*G);
  return{n,clump:s2/mean,aniso:ev[2]/ev[0]};}
function run(N,theta,seed,steps,checkpoints){
  const rnd=mb((seed*2654435761)>>>0);
  const om=new Int32Array(N),S=new Float64Array(N),al=new Uint8Array(N),px=new Float64Array(N),py=new Float64Array(N),pz=new Float64Array(N),vx=new Float64Array(N),vy=new Float64Array(N),vz=new Float64Array(N),W=new Float64Array(N*N),dS=new Float64Array(N),ax=new Float64Array(N),ay=new Float64Array(N),az=new Float64Array(N);
  for(let i=0;i<N;i++){om[i]=Math.floor(rnd()*K);S[i]=S0;al[i]=1;px[i]=(rnd()-.5)*.05;py[i]=(rnd()-.5)*.05;pz[i]=(rnd()-.5)*.05;}
  let we=1;const sat=s=>s/(1+s/S_BAND),tpk=2*Math.PI/K;const series=[];
  for(let s=0;s<steps;s++){dS.fill(0);ax.fill(0);ay.fill(0);az.fill(0);for(let i=0;i<N;i++)if(al[i])S[i]*=(1-MU);let mw=1e-9;const ws=we+1e-9;
    for(let i=0;i<N;i++){if(!al[i])continue;const oi=om[i],xi=px[i],yi=py[i],zi=pz[i],Si=S[i];
      for(let j=i+1;j<N;j++){if(!al[j])continue;const a=tpk*(oi-om[j]),gi=Math.cos(a+theta),gj=Math.cos(-a+theta),mag=Math.sqrt(sat(Si)*sat(S[j])),fi=ETA*gi*mag,fj=ETA*gj*mag;dS[i]+=fi;dS[j]+=fj;const idx=i*N+j,wij=W[idx]+fi;W[idx]=wij;W[j*N+i]+=fj;const aw=Math.abs(wij);if(aw>mw)mw=aw;
        let dx=xi-px[j],dy=yi-py[j],dz=zi-pz[j],dist=Math.sqrt(dx*dx+dy*dy+dz*dz)+1e-4,inv=1/dist,ux=dx*inv,uy=dy*inv,uz=dz*inv,w=Math.tanh(wij/ws),fc=-w*K_COUPLE,fb=K_BASE/(dist*dist),li=fc+fb,lj=fc-fb;ax[i]+=ux*li;ay[i]+=uy*li;az[i]+=uz*li;ax[j]+=ux*lj;ay[j]+=uy*lj;az[j]+=uz*lj;}}
    we=0.9*we+0.1*mw;
    for(let i=0;i<N;i++){if(!al[i])continue;S[i]+=dS[i];if(S[i]<0)S[i]=0;if(S[i]<=KAPPA_S){al[i]=0;continue;}let aX=ax[i]-K_CENTER*px[i],aY=ay[i]-K_CENTER*py[i],aZ=az[i]-K_CENTER*pz[i];vx[i]=(vx[i]+aX*DT)*DAMP;vy[i]=(vy[i]+aY*DT)*DAMP;vz[i]=(vz[i]+aZ*DT)*DAMP;let vm=Math.hypot(vx[i],vy[i],vz[i]);if(vm>3){const k=3/vm;vx[i]*=k;vy[i]*=k;vz[i]*=k;}px[i]+=vx[i]*DT;py[i]+=vy[i]*DT;pz[i]+=vz[i]*DT;}
    if(checkpoints&&checkpoints.includes(s+1)){const st=stats(px,py,pz,al,N);series.push([s+1,st.n,+st.clump.toFixed(2),+st.aniso.toFixed(3)]);}}
  const st=stats(px,py,pz,al,N);return{final:st,series};}

console.log("=== BARRIDO A: clumping(τ) — N=1000, θ_CP=0.3, config=3 ===");
console.log("paso\tvivos\tclumping\taniso(λ3/λ1)");
const A=run(1000,0.3,3,400,[20,40,80,120,160,200,280,360,400]);
for(const r of A.series)console.log(`${r[0]}\t${r[1]}\t${r[2]}\t\t${r[3]}`);

console.log("\n=== BARRIDO B: clumping vs θ_CP — N=600, 220 pasos, promedio seeds 1-3 ===");
console.log("θ_CP\tvivos\tclumping\taniso");
for(const th of [0.0,0.1,0.3,0.5,0.8,-0.3,-0.5]){
  let cs=[],ns=[],as=[];
  for(const sd of [1,2,3]){const r=run(600,th,sd,220,null);cs.push(r.final.clump);ns.push(r.final.n);as.push(r.final.aniso);}
  const avg=a=>a.reduce((x,y)=>x+y,0)/a.length;
  console.log(`${th>=0?'+':''}${th.toFixed(1)}\t${Math.round(avg(ns))}\t${avg(cs).toFixed(2)}\t\t${avg(as).toFixed(3)}`);
}
