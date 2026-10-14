from pathlib import Path
import requests
ROOT = Path(__file__).resolve().parents[1]
def main():
    target = ROOT / 'data/raw/geography/manhattan_borough.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get('https://data.cityofnewyork.us/resource/gthc-hcne.json', params={'$where': "boroname='Manhattan'"}, timeout=120)
    r.raise_for_status()
    data = r.json()
    if len(data) != 1 or data[0]['boroname'] != 'Manhattan':
        raise ValueError('Unexpected borough boundary response')
    target.write_text(r.text)
    print('Saved official Manhattan borough boundary', flush=True)
if __name__ == '__main__':
    main()
