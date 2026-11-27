exec(open('/home/claude/r1/base.py').read())
N=len(dk); P=lambda a: a.reshape(len(a),-1)
blur=lambda a,s: ndimage.gaussian_filter(a,(0,s,s))
def sizenorm(imgs, box=20):
    out=np.zeros_like(imgs)
    for i,a in enumerate(imgs):
        r=np.where(a.max(1)>0.1)[0]; c=np.where(a.max(0)>0.1)[0]
        if len(r)==0: continue
        crop=a[r[0]:r[-1]+1, c[0]:c[-1]+1]; h,w=crop.shape; s=box/max(h,w)
        z=ndimage.zoom(crop, s, order=1).clip(0,1); h2,w2=z.shape
        canvas=np.zeros((28,28),np.float32); oy=(28-h2)//2; ox=(28-w2)//2; canvas[oy:oy+h2, ox:ox+w2]=z[:28-oy,:28-ox]
        cy,cx=ndimage.center_of_mass(canvas); out[i]=ndimage.shift(canvas,(14-cy,14-cx),order=1)
    return out
def cache(name,f):
    p=f'F_{name}.npy'
    if os.path.exists(p): return np.load(p)
    F=f().astype(np.float32); np.save(p,F); return F
SF=lambda: cache('stroke', lambda: stroke_features(dk))
PF=lambda: cache('prof', lambda: profiles(dk))
base=lambda: np.hstack([P(dk), SF()*1.5, PF()])
