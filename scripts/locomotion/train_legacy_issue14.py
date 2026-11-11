"""Execute the exact archived issue12/13 training source for switch-off regression."""
import subprocess
from pathlib import Path
p=Path(__file__).with_name("train.py")
source=subprocess.check_output(["git","show","5a1f8b9a96a4b5d44cd8de2bc0900332d88e07bb:scripts/locomotion/train.py"]).decode()
exec(compile(source,str(p),"exec"),{"__name__":"__main__","__file__":str(p)})
