import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bikebalance.data import ARTIFACTS, DATA, TZ, digest, load_panel
from dashboard_support import (FAMILIES, LABELS, PREDICTORS, forecast_family, model_signature,
                               station_summary, test_frame, test_metrics, aggregate_test_metrics)

st.set_page_config(page_title='BikeBalance NYC', page_icon='🚲', layout='wide')
COLORS = {'actual': '#64748b', 'seasonal_naive': '#e69f00',
          'daily_sarima': '#2563eb', 'weather_arimax': '#0d9488'}
TARGET_LABELS = {'net_flow': 'Net flow', 'pickups': 'Pickups', 'dropoffs': 'Drop-offs'}


@st.cache_data
def base_data(base_hash):
    return load_panel()


@st.cache_data
def evaluation_data(path, receipt_hash):
    receipt = json.loads((path/'run.json').read_text())
    if receipt['base'] != digest(DATA/'modeling_dataset.parquet'):
        raise ValueError('Evaluation belongs to a different base')
    if receipt['status'] != 'complete' or digest(path/'forecasts.parquet') != receipt['forecasts_sha256']:
        raise ValueError('Evaluation artifacts are incomplete or corrupt')
    return pd.read_parquet(path/'forecasts.parquet')


@st.cache_data(ttl=3600)
def discovery():
    from bikebalance.gbfs import discover
    return discover()


@st.cache_data(ttl=60)
def live_stations():
    from bikebalance.gbfs import fetch_stations
    return fetch_stations(discovery())


@st.cache_data(ttl=3600)
def model_forecast(base_hash, station, family, hour, signature):
    # Full-hour templates are reused; partial amounts are derived at actual issuance.
    return forecast_family(base_data(base_hash), station, family, pd.Timestamp(hour))


def live_forecast(station, family, now):
    hour = now.floor('h')
    frame, metadata = model_forecast(base_hash, station, family, str(hour), model_signature())
    remaining, _ = model_forecast(base_hash, station, family, str(hour+pd.Timedelta(hours=1)), model_signature())
    frame = pd.concat([frame, remaining.iloc[[-1]]]).copy()
    stop = now+pd.Timedelta(hours=24)
    duration = np.array([(min(t+pd.Timedelta(hours=1), stop)-max(t, now)).total_seconds()/3600 for t in frame.index])
    frame[['pickups', 'dropoffs', 'net_flow']] = frame[['pickups', 'dropoffs', 'net_flow']].mul(duration, axis=0)
    frame['duration_hours'] = duration
    return frame[duration > 0], metadata


def choose_models(key):
    st.markdown('Compare forecasts')
    selected = []
    for column, family in zip(st.columns(len(FAMILIES)), FAMILIES):
        if column.checkbox(LABELS[family], value=family == 'seasonal_naive', key=f'{key}_{family}'):
            selected.append(family)
    return selected


def count_label(value):
    return str(int(value)) if pd.notna(value) and np.isfinite(value) and value >= 0 else '—'


def chart(actual, curves, target):
    figure = go.Figure()
    if actual is not None:
        figure.add_trace(go.Scatter(x=actual.index, y=actual[target], name='Recorded trips',
            mode='lines+markers', marker=dict(size=3), line=dict(color=COLORS['actual'], width=1), connectgaps=False))
    for family, frame in curves.items():
        figure.add_trace(go.Scatter(x=frame.index, y=frame[target], name=LABELS[family], mode='lines',
            line=dict(color=COLORS[family], width=2), connectgaps=False))
    indexes = [f.index for f in curves.values()] + ([actual.index] if actual is not None else [])
    if indexes and len(indexes[0]):
        ticks = indexes[0][::max(1, len(indexes[0])//8)]
        figure.update_xaxes(tickvals=ticks, ticktext=ticks.tz_convert(TZ).strftime('%b %d %H:%M %Z'))
    figure.update_layout(height=310, margin=dict(l=10, r=10, t=12, b=30),
        yaxis_title='Bike change per hour' if target == 'net_flow' else 'Trips per hour',
        xaxis_title='New York time', legend=dict(orientation='h', y=1.14),
        hovermode='x unified', template='plotly_white')
    figure.update_yaxes(zeroline=False)
    if target != 'net_flow':
        figure.update_yaxes(rangemode='tozero')
    return figure


st.title('BikeBalance NYC')
st.caption('Eight Manhattan stations · bikes available now and expected pickups and returns')
base_hash = digest(DATA/'modeling_dataset.parquet')
try:
    panel = base_data(base_hash)
except (ValueError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()
stations = pd.read_csv(DATA/'selected_stations.csv', dtype={'station_id': str}).set_index('station_id')
views = ['Live station overview', 'Next 24 hours', 'Test evaluation']
tabs = st.tabs(views, key='view', on_change='rerun')
page = st.session_state['view']


@st.fragment(run_every='60s')
def executive_view():
    st.subheader('Live station overview')
    st.write('Decide where to deliver bikes and where to move bikes out to make room for returns.')
    st.caption('Use current counts to spot urgent needs and the next 24 hours of forecasts to plan ahead.')
    left, right = st.columns([3, 1])
    family = left.radio('Forecast model', FAMILIES, format_func=LABELS.get, horizontal=True)
    st.caption(PREDICTORS[family])
    if right.button('Refresh live status', width='stretch'):
        live_stations.clear()
    now = pd.Timestamp.now(tz='UTC')
    try:
        live = live_stations().set_index('trip_station_id')
        st.caption(f'Bike and dock counts refresh every minute · checked {now.tz_convert(TZ):%b %d, %I:%M %p %Z}')
    except Exception:
        live = pd.DataFrame()
        st.warning('Current bike and dock counts are unavailable. Refresh to try again. Station alerts are hidden until fresh counts are available.')
    summaries = {}
    for station in stations.index:
        values = None
        try:
            values, _ = live_forecast(station, family, now)
        except (FileNotFoundError, ValueError, KeyError):
            pass
        summaries[station] = station_summary(live.loc[station], values, now) if station in live.index else {
            'status': 'Status unavailable', 'bike_shortfall': None, 'dock_shortfall': None}
    metrics = st.columns(3)
    valid = [s for s in summaries.values() if s['status'] in
             ['Few bikes available', 'Few empty docks', 'Live availability adequate']]
    shortfalls = [s for s in summaries.values() if s['bike_shortfall'] is not None]
    metrics[0].metric('Few bikes available · now',
        f'{sum(s["status"] == "Few bikes available" for s in valid)} / {len(valid)}' if valid else 'Unavailable')
    metrics[1].metric('Stations needing extra bikes · next 24h',
        f'{sum(s["bike_shortfall"] > 0 for s in shortfalls)} / {len(shortfalls)}' if shortfalls else 'Unavailable')
    metrics[2].metric('Few empty docks · now',
        f'{sum(s["status"] == "Few empty docks" for s in valid)} / {len(valid)}' if valid else 'Unavailable')
    for offset in range(0, len(stations), 4):
        columns = st.columns(4)
        for column, station in zip(columns, stations.index[offset:offset+4]):
            summary = summaries[station]
            with column.container(border=True):
                st.markdown(f'**{stations.loc[station, "station_name"]}**')
                row = live.loc[station] if station in live.index else None
                status = summary['status']
                st.caption('Now')
                if status == 'Few bikes available':
                    st.error(status)
                elif status == 'Few empty docks':
                    st.warning(status)
                else:
                    st.caption({'Live availability adequate': 'Bikes and docks available',
                                'Stale or invalid report': 'Station update needed',
                                'Not operational': 'Station unavailable'}.get(status, status))
                if (summary['bike_shortfall'] or 0) > 0 and status != 'Few bikes available':
                    st.warning('Extra bikes needed · next 24h')
                if (summary['dock_shortfall'] or 0) > 0 and status != 'Few empty docks':
                    st.warning('Extra docks needed · next 24h')
                st.metric('Available bikes', '—' if row is None else
                          f'{count_label(row.num_bikes_available)} / {count_label(row.capacity)}')
                if row is not None:
                    st.caption(f'{count_label(row.num_docks_available)} empty docks · Capacity {count_label(row.capacity)}')
                    if np.isfinite([row.num_bikes_available, row.capacity]).all() and row.capacity > 0:
                        st.progress(min(1., max(0., float(row.num_bikes_available)/float(row.capacity))))
                bikes, docks = summary['bike_shortfall'], summary['dock_shortfall']
                st.caption('Next 24h · estimated')
                st.markdown(f'Extra bikes needed: **{bikes if bikes is not None else "Unavailable"}**')
                st.caption(f'Extra docks needed: {docks if docks is not None else "Unavailable"}')
                if status in ['Stale or invalid report', 'Not operational'] and row is not None:
                    st.caption(f'Last reported {pd.Timestamp(row.reported_at).tz_convert(TZ):%I:%M %p %Z}')
                elif row is not None and (now-pd.Timestamp(row.reported_at)).total_seconds() > 120:
                    st.caption('Last update is over 2 minutes old. Refresh to check again.')
    if family != 'seasonal_naive' and model_signature() == 'pending':
        st.info('This forecast is not ready yet. Current bike counts are still shown, but future shortage estimates are unavailable.')
    st.caption('Extra bikes needed: estimated bikes to bring in. Extra docks needed: estimated empty spaces to create, '
               'usually by moving bikes out. These are possible shortages, not confirmed missed rides.')
    st.caption('Estimates assume no bikes are moved by the operator. Actual availability can change with bike moves, '
               'repairs, closures and the order of pickups and returns.')
    with st.expander('How to read the alerts'):
        st.write('A low-bike or low-dock alert appears below 10% of usable station spaces, with a minimum alert level of two. '
                 'Usable spaces are the bikes and empty docks currently reported as available.')
        st.write(f'Trip records end {panel.timestamp.max():%b %d, %Y}. Activity after that date is estimated, '
                 'while current bike and dock counts come from the live station feed. The weather model uses '
                 'observations from two days before each forecast hour; it is hidden if the needed weather is unavailable.')


with tabs[views.index(page)]:
    if page == 'Live station overview':
        executive_view()
    else:
        if page == 'Next 24 hours':
            st.subheader('Next 24 hours')
            station = st.selectbox('Station', list(stations.index), format_func=lambda value: stations.loc[value, 'station_name'])
            st.caption('Net flow = expected returns minus pickups. Above zero means trips are expected to add bikes; below zero means fewer bikes.')
            selected = choose_models('future_models')
            st.caption('The weather model uses conditions recorded two days before each forecast hour. '
                       'If those weather records are unavailable, its forecast is hidden.')
            curves, missing = {}, []
            for family in selected:
                try:
                    curves[family], _ = live_forecast(station, family, pd.Timestamp.now(tz='UTC'))
                except (FileNotFoundError, ValueError, KeyError):
                    missing.append(LABELS[family])
            if missing:
                st.info('Forecasts not ready: '+', '.join(missing)+'. Only available forecasts are shown.')
            st.caption(f'Trip records end {panel.timestamp.max():%b %d, %Y}. Later trip activity is estimated. '
                       'For part-hours, trips are spread evenly across the hour.')
            with st.expander('How the forecasts work'):
                for family in FAMILIES:
                    st.write(f'{LABELS[family]}: {PREDICTORS[family]}')
            for target in TARGET_LABELS:
                st.markdown(f'### {TARGET_LABELS[target]}')
                if curves:
                    st.plotly_chart(chart(None, curves, target), width='stretch', key='future_'+target)
                else:
                    st.caption('Select an available model to display its forecast.')
        else:
            st.subheader('Test evaluation')
            st.markdown('**Test period: January 1–September 30, 2026**')
            st.caption('Each day, we predicted the next day hour by hour, then compared with recorded trips. '
                       'Models were chosen using 2025 data before checking these results. Net flow means returns minus pickups.')
            try:
                pointer = json.loads((ARTIFACTS/'latest_run.json').read_text())
                path = ARTIFACTS/pointer['path']
                forecasts = evaluation_data(path, digest(path/'run.json'))
                weather_pointer = ARTIFACTS/'latest_arimax_run.json'
                if weather_pointer.exists():
                    weather_path = ARTIFACTS/json.loads(weather_pointer.read_text())['path']
                    weather_forecasts = evaluation_data(weather_path, digest(weather_path/'run.json'))
                    forecasts = pd.concat([forecasts[forecasts.family.isin(FAMILIES)], weather_forecasts], ignore_index=True)
                with st.expander('All-station test metrics', expanded=False):
                    aggregate_regime = st.selectbox('Trip data used for this summary', ['daily_updated', 'publication_lag'],
                        format_func=lambda v: 'Trips available through yesterday' if v == 'daily_updated' else 'Only published monthly trip files',
                        key='aggregate_regime')
                    aggregate = aggregate_test_metrics(panel, forecasts, aggregate_regime)
                    st.caption('Across all eight stations for the full test period. Each model is checked against the same recorded hours. '
                               'Changing the station or charts below does not change this summary.')
                    st.caption('MAE is the average hourly miss in bikes. RMSE gives extra weight to large misses. Lower is better for both.')
                    for column, family in zip(st.columns(len(FAMILIES)), FAMILIES):
                        with column.container(border=True):
                            st.markdown(f'**{LABELS[family]}**')
                            for target in TARGET_LABELS:
                                values = aggregate[target].get(family)
                                st.markdown(f'**{TARGET_LABELS[target]}**')
                                if values and values['mae'] is not None:
                                    st.markdown(f'MAE **{values["mae"]:.2f}** · RMSE **{values["rmse"]:.2f}**')
                                    st.caption(f'{values["scored_rows"]:,} hourly comparisons')
                                else:
                                    st.caption('Results not ready yet')
                st.divider()
                st.markdown('### Station detail')
                station = st.selectbox('Station', list(stations.index), format_func=lambda value: stations.loc[value, 'station_name'])
                selected = choose_models('test_models')
                regime = st.selectbox('Trip data available to the model', ['daily_updated', 'publication_lag'],
                    format_func=lambda v: 'Trips available through yesterday' if v == 'daily_updated' else 'Only published monthly trip files',
                    key='station_regime')
                st.caption('Through yesterday: assumes daily trip records are available. Monthly files: uses only trips '
                           'published by that date and estimates the missing days, as our live forecasts must do.')
                controls = st.columns(2)
                window = controls[0].selectbox('Chart shows', ['1 week', '2 weeks', '4 weeks', 'Full test period'])
                end = controls[1].date_input('Chart end date', date(2026, 9, 30), min_value=date(2026, 1, 1), max_value=date(2026, 9, 30))
                actual, all_curves = test_frame(panel, forecasts, station, regime)
                scores = {target: test_metrics(actual, all_curves, target) for target in TARGET_LABELS}
                start = date(2026, 1, 1) if window == 'Full test period' else max(date(2026, 1, 1),
                    end-timedelta(days={'1 week': 6, '2 weeks': 13, '4 weeks': 27}[window]))
                lower = pd.Timestamp(start, tz=TZ).tz_convert('UTC')
                upper = (pd.Timestamp(end, tz=TZ)+pd.DateOffset(days=1)).tz_convert('UTC')
                visible = (actual.index >= lower) & (actual.index < upper)
                curves = {k: frame.loc[visible] for k, frame in all_curves.items() if k in selected}
                missing = [LABELS[k] for k in selected if k not in all_curves]
                if missing:
                    st.info('Results not ready yet: '+', '.join(missing)+'.')
                st.caption('Charts show your chosen dates. Scores cover the full test period for this station, using the same '
                           'recorded hours for every model. MAE is the average hourly miss; RMSE gives more weight to large misses. Lower is better.')
                for target in TARGET_LABELS:
                    st.markdown(f'### {TARGET_LABELS[target]}')
                    columns = st.columns(3)
                    for column, family in zip(columns, FAMILIES):
                        values = scores[target].get(family)
                        with column.container(border=True):
                            st.markdown(f'**{LABELS[family]}**')
                            if values and values['mae'] is not None:
                                st.markdown(f'MAE **{values["mae"]:.2f}** · RMSE **{values["rmse"]:.2f}**')
                                st.caption(f'{values["scored_rows"]:,} hourly comparisons')
                            else:
                                st.caption('Results not ready yet')
                    st.plotly_chart(chart(actual.loc[visible], curves, target), width='stretch', key='test_'+target)
                st.markdown('**How the models make predictions**')
                for family in FAMILIES:
                    st.caption(f'{LABELS[family]}: {PREDICTORS[family]}')
                with st.expander('What to keep in mind'):
                    st.write('Models never use the weather or trip counts from the future hours they are predicting. '
                             'The weather model uses older observations, not future weather forecasts.')
                    st.write('Monthly-file tests use file update dates to estimate when trips became available. '
                             'Those dates may reflect later corrections, so the replay may give models less recent history than was actually available.')
                    st.write('These eight stations were selected using coverage across the full dataset, including the test period. '
                             'Results may not represent every Citi Bike station. Recorded trips do not include customers who could not find a bike or dock.')
            except (ValueError, FileNotFoundError, KeyError) as exc:
                st.warning('Saved test results could not be loaded or verified. Please reload the dashboard or ask the project team to check the saved results.')
