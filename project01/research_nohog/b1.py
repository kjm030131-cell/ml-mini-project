exec(open('/home/claude/r1/feats.py').read())
job=sys.argv[1]
if job=='0':
    run('A0_base',base())
    run('A2_deskew_only',P(dk))
    run('A3_blur0.7+stroke+prof',np.hstack([P(blur(dk,0.7)),SF()*1.5,PF()]))
    run('A6_base+simple',np.hstack([base(), simple(P(dk))/np.r_[np.full(56,28.0),np.ones(49),[784.0]]]))
else:
    run('A1_raw',P(im))
    run('A4_base+pool14',np.hstack([base(), P(dk.reshape(N,14,2,14,2).mean((2,4)))]))
    SN=cache('sn', lambda: P(sizenorm(dk)))
    run('A5_base+sizenorm',np.hstack([base(), SN]))
    run('A5b_sizenorm+stroke+prof',np.hstack([SN, SF()*1.5, PF()]))
