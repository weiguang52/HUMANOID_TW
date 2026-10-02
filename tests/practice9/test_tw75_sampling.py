import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[2]/'scripts/practice9/tw75_sampling.py'
spec=importlib.util.spec_from_file_location('tw75_sampling',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_category_and_clip_mass_ignore_duration():
    rows=m.balance([dict(category='walking',frames=101),dict(category='walking',frames=1001),dict(category='standing_upper',frames=51)])
    masses=[r['weight']*(r['frames']-1) for r in rows]
    assert masses==pytest.approx([.25,.25,.5])
def test_missing_category_is_not_silently_accepted():
    with pytest.raises(ValueError):m.balance([dict(category='walking',frames=101)])
