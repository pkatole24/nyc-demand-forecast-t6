from datetime import datetime, timezone

import pandas as pd
import requests

from .data import DATA

DISCOVERY = 'https://gbfs.citibikenyc.com/gbfs/gbfs.json'


def fetch_json(url):
    response = requests.get(url, timeout=(5, 20))
    response.raise_for_status()
    return response.json()


def discover():
    document = fetch_json(DISCOVERY)
    feeds = document['data']['en']['feeds']
    urls = {feed['name']: feed['url'] for feed in feeds}
    if not {'station_information', 'station_status'} <= urls.keys():
        raise ValueError('Discovery is missing station feeds')
    for name in ['station_information', 'station_status']:
        from urllib.parse import urlparse
        parsed = urlparse(urls[name])
        if parsed.scheme != 'https' or parsed.hostname not in {'gbfs.lyft.com', 'gbfs.citibikenyc.com'}:
            raise ValueError('Unexpected host in official discovery; review before fetching')
    return urls


def normalize(info, status, crosswalk, now=None):
    now = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if now.tzinfo is None:
        raise ValueError('API inspection time requires a timezone')
    information = pd.DataFrame(info['data']['stations'])
    states = pd.DataFrame(status['data']['stations'])
    if not information.station_id.is_unique or not states.station_id.is_unique:
        raise ValueError('Duplicate GBFS UUID station keys')
    selected = information[information.short_name.isin(crosswalk.trip_station_id)].copy()
    if not selected.short_name.is_unique or len(selected) != len(crosswalk):
        raise ValueError('Selected stations missing or duplicated in live information')
    matched = crosswalk.merge(selected, left_on='trip_station_id', right_on='short_name',
                               validate='one_to_one', suffixes=('_captured', ''))
    if not matched.gbfs_station_id.eq(matched.station_id).all():
        raise ValueError('Live UUID crosswalk changed; review station identity')
    result = matched.merge(states, on='station_id', how='left', validate='one_to_one', indicator=True)
    if not result['_merge'].eq('both').all():
        raise ValueError('Selected station status missing')
    result = result.drop(columns='_merge')
    feed_time = pd.to_datetime(status['last_updated'], unit='s', utc=True)
    result['reported_at'] = pd.to_datetime(result.last_reported, unit='s', utc=True)
    result['retrieved_at'] = now
    result['feed_at'] = feed_time
    result['feed_age_seconds'] = (now-feed_time).total_seconds()
    result['station_age_seconds'] = (now-result.reported_at).dt.total_seconds()
    ttl = max(60, int(status.get('ttl', 60)))
    result['ttl'] = ttl
    result['stale_warning'] = (result.station_age_seconds > max(120, 2*ttl)) | (
                              result.feed_age_seconds > max(120, 2*ttl))
    fields = ['num_bikes_available', 'num_docks_available', 'num_bikes_disabled',
              'num_docks_disabled', 'capacity']
    invalid = result[fields].isna().any(axis=1) | result[fields].lt(0).any(axis=1)
    invalid |= result.num_bikes_available.add(result.num_docks_available).gt(result.capacity)
    if 'num_ebikes_available' in result:
        invalid |= result.num_ebikes_available.lt(0) | result.num_ebikes_available.gt(result.num_bikes_available)
    result['operational'] = result[['is_installed', 'is_renting', 'is_returning']].eq(1).all(axis=1)
    result['simulation_allowed'] = (~invalid & result.operational &
        result.station_age_seconds.between(-30, 300) & result.feed_age_seconds.between(-30, 300))
    return result


def fetch_stations(urls=None):
    urls = urls or discover()
    crosswalk = pd.read_csv(DATA/'station_crosswalk.csv',
                            dtype={'trip_station_id': str, 'gbfs_station_id': str})
    info = fetch_json(urls['station_information'])
    status = fetch_json(urls['station_status'])
    return normalize(info, status, crosswalk)
