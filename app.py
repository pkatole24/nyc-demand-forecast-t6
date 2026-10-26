import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard_research import MODEL_COLORS, load_results, story_view, research_view

from bikebalance.data import ARTIFACTS, DATA, TZ, digest, load_panel
from dashboard_support import (FAMILIES, LIVE_FAMILIES, LABELS, PREDICTORS, forecast_family, model_signature,
                               station_summary, test_frame, test_metrics, aggregate_test_metrics)

st.set_page_config(page_title='BikeBalance NYC', page_icon='🚲', layout='wide')
COLORS = {'actual': '#4b5563', **MODEL_COLORS}
TARGET_LABELS = {'net_flow': 'Net flow', 'pickups': 'Pickups', 'dropoffs': 'Returns'}


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


def default_live_family(now):
    """Weather-forecast SARIMAX when its forecast inputs can be retrieved now, otherwise the calendar SARIMAX."""
    try:
        model_forecast(base_hash, stations.index[0], 'sarimax_weather_forecast', str(now.floor('h')), model_signature())
        return 'sarimax_weather_forecast'
    except (FileNotFoundError, ValueError, KeyError):
        return 'sarimax_calendar'


def choose_models(key, families, defaults):
    st.markdown('Compare forecasts')
    selected = []
    for column, family in zip(st.columns(len(families)), families):
        if column.checkbox(LABELS[family], value=family in defaults, key=f'{key}_{family}'):
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
    figure.update_yaxes(zeroline=target == 'net_flow', zerolinecolor='#6d7e8c')
    if target != 'net_flow':
        figure.update_yaxes(rangemode='tozero')
    return figure


st.title('BikeBalance NYC')
st.caption('Eight Manhattan stations · understand the rhythm, compare the evidence, explore the outlook')
base_hash = digest(DATA/'modeling_dataset.parquet')
try:
    panel = base_data(base_hash)
except (ValueError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()
stations = pd.read_csv(DATA/'selected_stations.csv', dtype={'station_id': str}).set_index('station_id')
STORY = DATA.parent/'story'
views = ['General overview', 'Live station outlook', 'Forecast explorer', 'Research & validation']
if '_open_forecast' in st.session_state:
    st.session_state['forecast_station'] = st.session_state.pop('_open_forecast')
    st.session_state['view'] = 'Forecast explorer'
tabs = st.tabs(views, key='view', on_change='rerun')
page = st.session_state['view']


def observation_context(now):
    cutoff = panel.timestamp.max()
    days = (now-cutoff).total_seconds()/86400
    st.caption(f'Trip history through {cutoff:%b %d, %Y, %I:%M %p %Z} · '
               f'{days:.1f} days since the last trip observation. Later activity is forecast, not observed.')


def weather_context():
    try:
        receipt = json.loads((DATA.parent/'live/weather/receipt.json').read_text())
        end = pd.Timestamp(receipt['stop_utc_exclusive'])-pd.Timedelta(hours=1)
        st.caption(f'Weather cache grid through {end.tz_convert(TZ):%b %d, %I:%M %p %Z}. '
                   'Individual observations may be missing. Weather ARIMAX uses 48-hour lags and explicit missing flags; '
                   'forecast loading verifies the cache and required date range.')
    except (OSError, ValueError, KeyError):
        st.caption('Incremental weather cache unavailable. Weather ARIMAX is shown only if its required inputs can be verified.')


@st.fragment(run_every='60s')
def executive_view():
    st.subheader('Live station outlook')
    st.write('Compare current bike and dock availability, then explore the next 24 hours under a forecast scenario.')
    left, right = st.columns([3, 1])
    family = left.radio('Available for live forecasts', LIVE_FAMILIES, format_func=LABELS.get, horizontal=True, key='live_family',
                        index=LIVE_FAMILIES.index(default_live_family(pd.Timestamp.now(tz='UTC'))))
    if right.button('Refresh live status', width='stretch'):
        live_stations.clear()
    now = pd.Timestamp.now(tz='UTC')
    observation_context(now)
    weather_context()
    try:
        live = live_stations().set_index('trip_station_id')
        st.caption(f'Live counts checked {now.tz_convert(TZ):%b %d, %I:%M:%S %p %Z} · refresh every minute. '
                   f'Scenario model: {LABELS[family]}.')
    except Exception:
        live = pd.DataFrame()
        st.warning('Current station counts are unavailable. Observed alerts and modeled gaps remain unavailable until reports pass validation.')
    summaries, records, failures = {}, [], []
    valid_status = ['Few bikes available', 'Few empty docks', 'Live availability adequate']
    for station in stations.index:
        values = None
        try:
            values, _ = live_forecast(station, family, now)
        except (FileNotFoundError, ValueError, KeyError):
            failures.append(station)
        row = live.loc[station] if station in live.index else None
        summary = station_summary(row, values, now) if row is not None else {
            'status': 'Status unavailable', 'bike_shortfall': None, 'dock_shortfall': None}
        summaries[station] = summary
        status = summary['status']
        age = max(0., (now-pd.Timestamp(row.reported_at)).total_seconds()/60) if row is not None else None
        records.append({'Station': stations.loc[station, 'station_name'],
            'Bikes now': float(row.num_bikes_available) if row is not None and status in valid_status else None,
            'Empty docks now': float(row.num_docks_available) if row is not None and status in valid_status else None,
            'Observed status': status.replace('Live availability adequate', 'Bikes and docks available'),
            'Modeled bike gap': summary['bike_shortfall'], 'Modeled dock gap': summary['dock_shortfall'],
            'Report age (min)': age, '_station': station,
            '_priority': 0 if status in valid_status[:2] else 1 if status in valid_status else 2,
            '_gap': max(summary['bike_shortfall'] or 0, summary['dock_shortfall'] or 0)})
    valid = [s for s in summaries.values() if s['status'] in valid_status]
    st.markdown('### Observed conditions now')
    metrics = st.columns(3)
    metrics[0].metric('Stations with few bikes', f'{sum(s["status"] == valid_status[0] for s in valid)} / {len(valid)}' if valid else 'Unavailable')
    metrics[1].metric('Stations with few empty docks', f'{sum(s["status"] == valid_status[1] for s in valid)} / {len(valid)}' if valid else 'Unavailable')
    metrics[2].metric('Reports passing checks', f'{len(valid)} / {len(stations)}')
    st.markdown('### Station attention list')
    st.caption('Observed alerts first, then largest modeled gap. Select a row for station detail. This is an attention list, not a truck-route plan.')
    table = pd.DataFrame(records).sort_values(['_priority', '_gap', 'Station'], ascending=[True, False, True]).reset_index(drop=True)
    visible = table.drop(columns=['_station', '_priority', '_gap'])
    event = st.dataframe(visible, hide_index=True, width='stretch', on_select='rerun', selection_mode='single-row',
        key='station_attention', column_config={name: st.column_config.NumberColumn(format='%.0f') for name in
        ['Bikes now', 'Empty docks now', 'Modeled bike gap', 'Modeled dock gap']})
    modeled = [s for s in summaries.values() if s['bike_shortfall'] is not None]
    st.caption(f'Modeled scenarios available for {len(modeled)} / {len(stations)} stations. '
               'Blank values are unavailable, not zero. Bike gap is the maximum net-flow deficit against current bikes; '
               'dock gap is the maximum excess against current usable spaces. Counts are rounded up.')
    st.info('Scenarios assume no operator redistribution and uniform activity within partial hours. They do not measure '
            'missed rides or establish an optimal delivery quantity. Pickup/return ordering, closures and maintenance can change availability.')
    if failures:
        st.caption(f'{LABELS[family]} forecast unavailable at {len(failures)} stations; their scenario gaps remain blank.')
    if event.selection.rows:
        station = table.iloc[event.selection.rows[0]]['_station']
        st.markdown(f'**{stations.loc[station, "station_name"]}**')
        if station in live.index:
            row = live.loc[station]
            st.caption(f'Reported usable spaces: {count_label(row.num_bikes_available+row.num_docks_available)} · '
                       f'Nominal capacity: {count_label(row.capacity)}. Disabled or unavailable equipment can explain the difference.')
        if st.button('Explore this station’s forecast'):
            st.session_state['_open_forecast'] = station
            st.rerun(scope='app')
    with st.expander('Model and alert definitions'):
        st.write(PREDICTORS[family])
        st.write('Low-bike and low-dock alerts use 10% of currently usable spaces, with a minimum threshold of two. '
                 'Old, invalid and nonoperational reports cannot initialize a scenario. Report age above two minutes warrants attention; '
                 'the existing validation blocks reports older than five minutes or outside feed/count checks.')
        st.write('Both SARIMAX models run live with parameters estimated through the last published trips; the weather-forecast '
                 'version also reads Open-Meteo forecasts issued two days ahead, covering the gap since those trips and the next 24 hours. '
                 'Poisson + weather forecast scored best in the 2026 test but is not live: it needs the previous day\'s trips, which the '
                 'monthly trip archive does not provide. Compare all studies in Research & validation.')


def forecast_view():
    st.subheader('Forecast explorer')
    station = st.selectbox('Station', list(stations.index), format_func=lambda value: stations.loc[value, 'station_name'], key='forecast_station')
    now = pd.Timestamp.now(tz='UTC')
    stop = now+pd.Timedelta(hours=24)
    st.caption(f'Issued {now.tz_convert(TZ):%b %d, %I:%M %p %Z} · through {stop.tz_convert(TZ):%b %d, %I:%M %p %Z}')
    observation_context(now)
    weather_context()
    family = st.radio('Activity outlook model', LIVE_FAMILIES, format_func=LABELS.get, horizontal=True, key='outlook_family',
                      index=LIVE_FAMILIES.index(default_live_family(now)))
    curves = {}
    try:
        curves[family], _ = live_forecast(station, family, now)
    except (FileNotFoundError, ValueError, KeyError):
        st.info('This forecast is not ready or its inputs are unavailable. Choose another model; no substitute is displayed.')
    st.caption('Available for live forecasts: last-week baseline, daily SARIMA, weather ARIMAX, SARIMAX + weekly calendar and '
               'SARIMAX + calendar + weather forecast. The Poisson models have historical evaluations only. Part-hour trip amounts are '
               'prorated uniformly.')
    if curves:
        values = curves[family]
        metrics = st.columns(3)
        metrics[0].metric('Expected pickups · next 24h', f'{values.pickups.sum():,.1f}')
        metrics[1].metric('Expected returns · next 24h', f'{values.dropoffs.sum():,.1f}')
        metrics[2].metric('Expected net change', f'{values.net_flow.sum():+,.1f}')
        st.markdown('### Expected departures and returns')
        figure = go.Figure()
        local = values.index.tz_convert(TZ)
        for target, title, dash in [('pickups', 'Pickups', 'solid'), ('dropoffs', 'Returns', 'dash')]:
            figure.add_trace(go.Scatter(x=local, y=values[target], mode='lines+markers', name=title,
                line=dict(color=COLORS[family], dash=dash, width=2), marker=dict(size=4), connectgaps=False))
        figure.update_layout(height=330, template='plotly_white', margin=dict(l=10, r=10, t=15, b=20),
            hovermode='x unified', yaxis_title='Expected trips in displayed bin', legend=dict(orientation='h', y=1.12))
        figure.update_xaxes(tickvals=local[::4], ticktext=local[::4].strftime('%b %d %H:%M %Z'))
        figure.update_yaxes(rangemode='tozero')
        st.plotly_chart(figure, width='stretch', key='activity_outlook')
        st.markdown('### Net change from trips')
        figure = go.Figure()
        figure.add_trace(go.Bar(x=local, y=values.net_flow,
            marker_color=np.where(values.net_flow >= 0, '#16877d', '#b66548'),
            hovertemplate='%{x}<br>Net change %{y:+.1f} bikes<extra></extra>'))
        figure.add_hline(y=0, line_color='#6d7e8c')
        figure.update_layout(height=230, template='plotly_white', margin=dict(l=10, r=10, t=10, b=20), yaxis_title='Returns − pickups')
        figure.update_xaxes(tickvals=local[::4], ticktext=local[::4].strftime('%b %d %H:%M %Z'))
        st.plotly_chart(figure, width='stretch', key='net_outlook')
        st.caption('Positive means trips add bikes; negative means trips remove bikes. This is a flow forecast, not station inventory. '
                   'First and last bins may cover part of an hour; other bins cover one hour.')
        full = values[values.duration_hours >= 1-1e-9]
        if not full.empty:
            low, high = full.net_flow.idxmin(), full.net_flow.idxmax()
            drain = (f'{full.loc[low, "net_flow"]:+.1f} bikes at {low.tz_convert(TZ):%I:%M %p %Z}'
                     if full.loc[low, 'net_flow'] < 0 else 'none in the full-hour bins')
            buildup = (f'{full.loc[high, "net_flow"]:+.1f} bikes at {high.tz_convert(TZ):%I:%M %p %Z}'
                       if full.loc[high, 'net_flow'] > 0 else 'none in the full-hour bins')
            st.write(f'**Largest hourly drain:** {drain}. **Largest hourly build-up:** {buildup}.')
    with st.expander('Compare deployed models'):
        target = st.selectbox('Forecast target', list(TARGET_LABELS), format_func=TARGET_LABELS.get, key='compare_target')
        selected = choose_models('future_models', LIVE_FAMILIES, {default_live_family(now), 'seasonal_naive'})
        comparisons, missing = {}, []
        for model in selected:
            try:
                comparisons[model], _ = live_forecast(station, model, now)
            except (FileNotFoundError, ValueError, KeyError):
                missing.append(LABELS[model])
        if missing:
            st.info('Forecasts not ready: '+', '.join(missing)+'. Only verified available forecasts are shown.')
        if comparisons:
            figure = chart(None, comparisons, target)
            figure.update_yaxes(title='Bike change in displayed bin' if target == 'net_flow' else 'Trips in displayed bin')
            st.plotly_chart(figure, width='stretch', key='future_comparison')
        st.caption('Differences between model lines are not prediction intervals.')
    with st.expander('How the forecasts work'):
        for model in LIVE_FAMILIES:
            st.write(f'**{LABELS[model]}:** {PREDICTORS[model]}')


def historical_view():
    st.markdown('### Historical replay: last-week baseline, SARIMA and ARIMAX')
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
        with st.expander('All-station metrics, including net flow and publication lag', expanded=True):
            aggregate_regime = st.selectbox('Trip data used for this summary', ['daily_updated', 'publication_lag'],
                format_func=lambda v: 'Trips available through yesterday' if v == 'daily_updated' else 'Only published monthly trip files',
                key='aggregate_regime')
            aggregate = aggregate_test_metrics(panel, forecasts, aggregate_regime)
            st.caption('January–September 2026 · Eight stations. Changing the station or charts below does not change this summary.')
            st.caption('MAE measures the average hourly error. RMSE gives more weight to large errors. Lower is better for both.')
            targets = ['pickups', 'dropoffs', 'net_flow']
            columns = pd.MultiIndex.from_tuples(
                [(TARGET_LABELS[target], metric.upper()) for target in targets for metric in ['mae', 'rmse']])
            rows = []
            names = []
            counts = {target: set() for target in targets}
            for family in FAMILIES:
                names.append(LABELS[family])
                row = []
                for target in targets:
                    values = aggregate[target].get(family)
                    row.extend([values[metric] if values else None for metric in ['mae', 'rmse']])
                    if values and values['mae'] is not None:
                        counts[target].add(values['scored_rows'])
                rows.append(row)
            saved_results = None
            if aggregate_regime == 'daily_updated':
                try:
                    saved_results = load_results('test')
                except (OSError, ValueError, KeyError):
                    st.caption('Additional research scores are unavailable. Verified deployed-model scores remain below.')
            for model, label in [('sarimax_calendar', 'SARIMAX + weekly calendar'),
                                 ('sarimax_weather_forecast', 'SARIMAX + calendar + weather forecast'),
                                 ('poisson', 'Poisson regression'), ('poisson_weather_forecast', 'Poisson + weather forecast')]:
                names.append(label)
                row = []
                for target in targets:
                    if saved_results is None:
                        row.extend(['Not evaluated' if aggregate_regime == 'publication_lag' else 'Unavailable'] * 2)
                    elif target == 'net_flow':
                        row.extend([None, None])
                    else:
                        available = saved_results[saved_results.model.eq(model) & saved_results.target.eq(target)]
                        if available.empty:
                            row.extend([None, None])
                        else:
                            values = available.iloc[0]
                            row.extend([values.mae, values.rmse])
                            counts[target].add(int(values.rows))
                rows.append(row)
            comparison = pd.DataFrame(rows, index=pd.Index(names, name='Model'), columns=columns)

            def highlight_best(values):
                numeric = pd.to_numeric(values, errors='coerce')
                return ['font-weight: bold; background-color: rgba(42, 120, 214, 0.15)'
                        if pd.notna(value) and value == numeric.min() else '' for value in numeric]

            styles = comparison.apply(highlight_best)
            display = comparison.map(lambda value: '—' if pd.isna(value) else
                                     value if isinstance(value, str) else f'{value:.2f}')
            st.table(display.style.apply(lambda _: styles, axis=None).set_properties(**{'text-align': 'right'}))
            coverage = []
            for target in targets:
                values = sorted(counts[target])
                amount = f'{values[0]:,}' if len(values) == 1 else '–'.join(f'{value:,}' for value in values)
                coverage.append(f'{TARGET_LABELS[target]}: {amount or "0"}')
            st.caption('Hourly comparisons per reported metric — '+ '; '.join(coverage)+'. Equal reported counts alone do not verify identical keys across independent saved summaries.')
            st.caption('Bold, shaded values mark the lowest error in each column. “—” means results unavailable; '
                       '“Not evaluated” means the model was not evaluated under this regime.')
        st.divider()
        st.markdown('### Station detail')
        station = st.selectbox('Station', list(stations.index), key='test_station', format_func=lambda value: stations.loc[value, 'station_name'])
        selected = choose_models('test_models', FAMILIES, {'seasonal_naive'})
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

with tabs[views.index(page)]:
    if page == views[0]:
        story_view(panel, stations)
    elif page == views[1]:
        executive_view()
    elif page == views[2]:
        forecast_view()
    else:
        research_view()
        st.divider()
        historical_view()
