import numpy as np, wave
SR=44100; DUR=35.0
out=np.zeros(int(SR*DUR)+SR)
def pluck(freq, dur, amp):
    n=int(SR*dur); N=max(2,int(SR/freq))
    buf=np.random.uniform(-1,1,N); y=np.zeros(n)
    for i in range(n):
        y[i]=buf[i%N]; buf[i%N]=0.5*(buf[i%N]+buf[(i+1)%N])*0.996
    return y*amp
def drum(dur=0.25, amp=0.6):
    t=np.arange(int(SR*dur))/SR
    f=110*np.exp(-t*18)+55
    return amp*np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*14)
# D dorian jig melody (6/8), scale degrees
D=293.66; ratios=[1,9/8,6/5,4/3,3/2,5/3,16/9,2]
mel=[0,2,4, 4,2,0, 1,3,5, 5,3,1, 0,2,4, 7,6,4, 3,1,2, 0,0,0]
bass=[0,0,4,4,3,3,0,0]
t=0.0; step=0
while t<DUR-0.5:
    bpm=118+40*(t/DUR)          # accelerating: "couldn't stop"
    eighth=60/bpm/2
    deg=mel[step%len(mel)]
    s=int(t*SR); p=pluck(D*ratios[deg], 0.5, 0.35); out[s:s+len(p)]+=p[:len(out)-s]
    if step%3==0:
        b=bass[(step//3)%len(bass)]
        bp=pluck(D/2*ratios[b], 0.9, 0.3); out[s:s+len(bp)]+=bp[:len(out)-s]
        dr=drum(amp=0.5 if step%6==0 else 0.3); out[s:s+len(dr)]+=dr[:len(out)-s]
    t+=eighth; step+=1
out=out[:int(SR*DUR)]
fade=int(SR*1.5); out[-fade:]*=np.linspace(1,0,fade)
out/=np.max(np.abs(out))*1.05
with wave.open("music.wav","wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((out*32767).astype(np.int16).tobytes())
