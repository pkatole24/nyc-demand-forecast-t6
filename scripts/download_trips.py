from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/citibike'
BASE = 'https://s3.amazonaws.com/tripdata'

def discover():
    RAW.mkdir(parents=True, exist_ok=True)
    objects = {}
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    for year in range(2024, 2027):
        token = None
        page = 0
        while True:
            params = {'list-type': '2', 'prefix': str(year)}
            if token:
                params['continuation-token'] = token
            r = requests.get(BASE, params=params, timeout=120)
            r.raise_for_status()
            (RAW / f'bucket_{year}_{page}.xml').write_bytes(r.content)
            tree = ET.fromstring(r.content)
            for item in tree.findall('s:Contents', ns):
                key = item.findtext('s:Key', namespaces=ns)
                match = re.fullmatch(r'(202[456]\d{2})-citibike-tripdata.zip', key)
                if match:
                    objects[match[1]] = {'key': key, 'bytes': int(item.findtext('s:Size', namespaces=ns)),
                        'etag': item.findtext('s:ETag', namespaces=ns).strip('"'),
                        'last_modified': item.findtext('s:LastModified', namespaces=ns)}
            if tree.findtext('s:IsTruncated', namespaces=ns) != 'true':
                break
            token = tree.findtext('s:NextContinuationToken', namespaces=ns)
            page += 1
    months = [f'{y}{m:02}' for y in range(2024, 2027) for m in range(1, 13) if f'{y}{m:02}' <= '202609']
    manifest = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'source': BASE,
                'months': [{ 'month': m, 'available': m in objects, **objects.get(m, {}) } for m in months]}
    (RAW / 'availability.json').write_text(json.dumps(manifest, indent=2))
    return manifest

def download_extract(item):
    month = item['month']
    target = RAW / item['key']
    receipt = RAW / f'{month}_receipt.json'
    extracted = RAW / 'extracted' / month
    if receipt.exists() and target.exists() and target.stat().st_size == item['bytes']:
        old = json.loads(receipt.read_text())
        if old.get('etag') == item['etag'] and all((extracted / f['name']).exists() and (extracted / f['name']).stat().st_size == f['bytes'] for f in old['csv_files']):
            print(f'{month}: verified cached archive/extraction', flush=True)
            return old
    if not target.exists() or target.stat().st_size != item['bytes'] or (receipt.exists() and json.loads(receipt.read_text()).get('etag') != item['etag']):
        partial = target.with_suffix('.zip.part')
        for attempt in range(8):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                with partial.open('ab') as f:
                    while offset < item['bytes']:
                        end = min(offset + 4 * 1024 * 1024, item['bytes']) - 1
                        r = requests.get(f'{BASE}/{item["key"]}', headers={'Range': f'bytes={offset}-{end}', 'If-Match': f'"{item["etag"]}"', 'Connection': 'close'}, timeout=(30, 30))
                        r.raise_for_status()
                        if r.status_code != 206 or len(r.content) != end - offset + 1 or r.headers.get('Content-Range') != f'bytes {offset}-{end}/{item["bytes"]}':
                            raise ValueError('Unexpected range response')
                        f.write(r.content)
                        f.flush()
                        offset = end + 1
                if partial.stat().st_size != item['bytes']:
                    raise ValueError('Download size does not match official listing')
                partial.replace(target)
                break
            except Exception as exc:
                print(f'{month}: retry {attempt+1}, {type(exc).__name__}: {str(exc)[:200]}', flush=True)
                if attempt == 7:
                    raise
                time.sleep(2 ** attempt)
    digest = hashlib.sha256()
    with target.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    extracted.mkdir(parents=True, exist_ok=True)
    files = []
    with zipfile.ZipFile(target) as archive:
        for member in archive.infolist():
            if not member.filename.lower().endswith('.csv') or '__MACOSX/' in member.filename:
                continue
            name = Path(member.filename).name
            if name in {x['name'] for x in files}:
                raise ValueError(f'Colliding CSV names in {target.name}')
            out = extracted / name
            if not out.exists() or out.stat().st_size != member.file_size:
                with archive.open(member) as src, out.with_suffix('.csv.part').open('wb') as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                out.with_suffix('.csv.part').replace(out)
            files.append({'name': name, 'archive_member': member.filename, 'bytes': member.file_size, 'crc32': member.CRC})
    result = {**item, 'sha256': digest.hexdigest(), 'csv_files': files}
    receipt.write_text(json.dumps(result, indent=2))
    print(f'{month}: downloaded/extracted {len(files)} CSVs ({item["bytes"] / 1e9:.2f} GB ZIP)', flush=True)
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--discover-only', action='store_true')
    args = p.parse_args()
    manifest = discover()
    items = [x for x in manifest['months'] if x['available']]
    print(f'{len(items)}/33 requested months available; ZIP total {sum(x["bytes"] for x in items)/1e9:.2f} GB', flush=True)
    if not args.discover_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(download_extract, items))

if __name__ == '__main__':
    main()
