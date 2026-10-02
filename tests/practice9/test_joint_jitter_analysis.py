import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/practice9'))
from analyze_s2_joint_commands import highpass

def test_band_reports_absolute_sine_amplitude():
    t=np.arange(1000)/50;valid=np.ones(len(t),dtype=bool)
    x=(10*np.sin(2*np.pi*4*t))[:,None]
    band=highpass(x,valid,50,[2,8]);rms=np.sqrt(np.nanmean(band**2))
    assert abs(rms-10/np.sqrt(2))<.15
    assert np.sqrt(np.nanmean(highpass(x,valid,50)**2))<.1

def test_reset_discontinuity_is_not_counted_as_oscillation():
    x=np.zeros((1000,1));x[500:]=100
    valid=np.ones(1000,dtype=bool);valid[499:502]=False
    result=highpass(x,valid,50,[2,8])
    assert np.nanmax(np.abs(result))<1e-8
    assert np.isnan(result[499:502]).all()
