exec(open('/home/claude/r1/feats.py').read())
SN=cache('sn', lambda: P(sizenorm(dk)))
pool=P(dk.reshape(N,14,2,14,2).mean((2,4)))
simp=simple(P(dk))/np.r_[np.full(56,28.0),np.ones(49),[784.0]]
combo=np.hstack([base(), SN, pool, simp])
job=sys.argv[1]
if job=='0':
    run('B1_combo',combo)
    run('B3_A5_C20',np.hstack([base(), SN]),C=20)
else:
    run('B2_combo_g1.5',combo,gm=1.5)
    run('B4_sqrt_base',np.hstack([P(np.sqrt(dk)), SF()*1.5, PF()]))
