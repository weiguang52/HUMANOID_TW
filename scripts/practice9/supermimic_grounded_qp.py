"""Explicit optional interior-point backend for unchanged scaled grounded QPs."""
from types import SimpleNamespace
import numpy as np,scipy.sparse as sp

def solve_grounded_qp(P,q,A,l,u,backend='osqp'):
 if backend=='osqp':
  import osqp
  solver=osqp.OSQP();solver.setup(P=P,q=q,A=A,l=l,u=u,verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polishing=True);return solver.solve()
 assert backend=='clarabel',backend
 import clarabel
 upper=np.flatnonzero(np.isfinite(u));lower=np.flatnonzero(np.isfinite(l));M=sp.vstack([A[upper],-A[lower]],format='csc');b=np.r_[u[upper],-l[lower]];settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200;settings.tol_feas=1e-8;settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-8;res=clarabel.DefaultSolver(P,q,M,b,[clarabel.NonnegativeConeT(len(b))],settings).solve();dual=np.zeros(len(l));z=np.array(res.z);dual[upper]+=z[:len(upper)];dual[lower]-=z[len(upper):]
 return SimpleNamespace(x=np.array(res.x),y=dual,info=SimpleNamespace(status=str(res.status),status_val=1 if str(res.status)=='Solved' else -1,iter=res.iterations,prim_res=res.r_prim,dual_res=res.r_dual))
