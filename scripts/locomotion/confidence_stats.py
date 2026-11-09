"""Binomial Wilson95% intervals, conditional on a fixed evaluation policy."""
import math
def wilson(k,n):
    if n<=0 or not 0<=k<=n:
        raise ValueError("Require n>0 and0<=k<=n")
    z=1.959963984540054
    p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,c-h),min(1.,c+h)]
