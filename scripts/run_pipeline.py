import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main():
    for script in ['download_trips.py','download_geography.py','download_weather.py','prepare_trips.py','build_datasets.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],cwd=ROOT,check=True)
if __name__=='__main__':
    main()
