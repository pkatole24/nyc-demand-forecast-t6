import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from .data import ARTIFACTS, DATA, TZ, write_json


def check_releases():
    """Inspect official metadata only; never download or rebuild the immutable panel."""
    now = pd.Timestamp.now(tz=TZ)
    last_complete = now.tz_localize(None).to_period('M')-1
    root_url = 'https://s3.amazonaws.com/tripdata'
    namespace = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    archives = []
    for year in range(2024, now.year+1):
        token = None
        while True:
            parameters = {'list-type': '2', 'prefix': str(year)}
            if token:
                parameters['continuation-token'] = token
            response = requests.get(root_url, params=parameters, timeout=(5, 30))
            response.raise_for_status()
            tree = ET.fromstring(response.content)
            for item in tree.findall('s:Contents', namespace):
                key = item.find('s:Key', namespace).text
                if re.fullmatch(r'\d{6}-citibike-tripdata\.zip', key):
                    month = key[:6]
                    if pd.Period(month[:4]+'-'+month[4:], freq='M') <= last_complete:
                        archives.append({'month': month, 'key': key,
                            'last_modified': item.find('s:LastModified', namespace).text,
                            'bytes': int(item.find('s:Size', namespace).text)})
            if tree.find('s:IsTruncated', namespace).text == 'false':
                break
            token = tree.find('s:NextContinuationToken', namespace).text
    import json
    captured = json.loads((DATA/'source_availability.json').read_text())
    previous = {row['month']: row for row in captured['months']}
    result = {'checked_at': str(now), 'source': root_url,
              'latest_complete_published_month': max((a['month'] for a in archives), default=None),
              'new_or_revised': [a for a in archives if a['month'] not in previous or
                  pd.Timestamp(a['last_modified']) != pd.Timestamp(previous[a['month']]['last_modified'])],
              'archives': archives, 'downloaded_archives': 0,
              'next_step': 'Request a separately audited base update, then rerun diagnostics, selection, '
                           'backtests and deployment. Keep this historical test frozen.'}
    write_json(ARTIFACTS/'release_check.json', result)
    return result
