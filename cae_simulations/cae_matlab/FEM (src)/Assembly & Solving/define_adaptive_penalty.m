function pen=define_adaptive_penalty(maxpenalty, countiter)

pen=maxpenalty * min(1,1e-3*2.^countiter);