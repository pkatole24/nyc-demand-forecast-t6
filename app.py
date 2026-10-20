import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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
views = ['General overview', 'Live station overview', 'Next 24 hours', 'Test evaluation']
STORY = DATA.parent/'story'
# Story palette: one accent per highlighted series, the rest recede to grey.
ACCENT, ACCENT_2, MUTED, INK_2 = '#2a78d6', '#eb6834', '#c3c2b7', '#52514e'


@st.cache_data
def weekday_drift(base_hash):
    """Average bikes gained (+) or lost (-) since midnight on a weekday, by station and hour."""
    frame = base_data(base_hash)
    weekdays = frame[(frame.weekday < 5) & ~frame.holiday & frame.partition.ne('test')]
    hourly = weekdays.assign(net_flow=weekdays.net_flow.astype(float)).groupby(['station_id', 'hour']).net_flow.mean().unstack()
    return hourly.cumsum(axis=1)


def drift_chart(drift, highlight):
    figure = go.Figure()
    hours = list(drift.columns)
    for station, values in drift.iterrows():
        color = highlight.get(station)
        name = stations.loc[station, 'station_name']
        figure.add_trace(go.Scatter(x=hours, y=values.to_numpy(), mode='lines', name=name,
            line=dict(color=color or MUTED, width=3 if color else 1.5), showlegend=False,
            hovertemplate=f'{name}<br>By end of %{{x}}:00: %{{y:+.0f}} bikes<extra></extra>'))
        if color:
            figure.add_annotation(x=hours[-1], y=values.iloc[-1], text=name, showarrow=False, xanchor='left',
                                  xshift=6, font=dict(color=color, size=12))
    figure.add_hline(y=0, line=dict(color=INK_2, width=1))
    figure.update_layout(height=360, margin=dict(l=10, r=170, t=10, b=40), template='plotly_white',
        xaxis=dict(title='Hour of the day (New York time)', tickvals=list(range(0, 24, 3)),
                   ticktext=[f'{h}:00' for h in range(0, 24, 3)]),
        yaxis=dict(title='Bikes gained (+) or lost (−) since midnight'))
    return figure


def leaderboard_chart(board):
    board = board.sort_values('pickups_rmse', ascending=False)
    best = board.loc[board.pickups_rmse.idxmin(), 'model']
    naive = board.set_index('model').loc['seasonal_naive']
    figure = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                           subplot_titles=('Pickups', 'Drop-offs'))
    for column, target in enumerate(['pickups', 'dropoffs'], start=1):
        colors = [ACCENT if m == best else MUTED for m in board.model]
        figure.add_trace(go.Bar(x=board[f'{target}_rmse'], y=board.label, orientation='h', marker_color=colors,
            text=[f'{v:.2f}' for v in board[f'{target}_rmse']], textposition='outside', showlegend=False,
            hovertemplate='%{y}: RMSE %{x:.2f} trips/hour<extra></extra>'), row=1, col=column)
        figure.add_vline(x=naive[f'{target}_rmse'], line=dict(color=INK_2, dash='dot', width=1), opacity=0.4,
                         layer='below', row=1, col=column)
    figure.update_xaxes(title_text='RMSE (trips per hour, lower is better)', range=[0, board[['pickups_rmse', 'dropoffs_rmse']].max().max()*1.18])
    figure.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=40), template='plotly_white')
    return figure


def story_view():
    st.subheader('Can we predict where bikes will run out, a day ahead?')
    st.write('Citi Bike stations empty out and fill up in predictable waves. When a station has no bikes, riders walk away; '
             'when it has no empty docks, riders cannot return their bike. Operators move bikes by truck to prevent both, '
             'and they need to know **where and when** the shortages will happen. BikeBalance NYC forecasts hourly '
             'pickups and returns at eight Manhattan stations for the next day, and turns those forecasts into '
             'estimated rebalancing needs.')

    st.markdown('#### 1 · The problem: stations drain and fill on a daily rhythm')
    drift = weekday_drift(base_hash)
    low, high = drift.min(axis=1), drift.max(axis=1)
    drain, fill = low.idxmin(), high.idxmax()
    capacity = pd.read_csv(DATA/'station_crosswalk.csv', dtype={'trip_station_id': str}).set_index('trip_station_id').current_capacity
    swing = (high-low)
    columns = st.columns(3)
    columns[0].metric(f'Biggest daily drain · {stations.loc[drain, "station_name"]}', f'{low[drain]:.0f} bikes',
                      help='Average weekday, 2024–2025: cumulative pickups minus returns since midnight, at its lowest point.')
    columns[1].metric(f'Biggest daily build-up · {stations.loc[fill, "station_name"]}', f'+{high[fill]:.0f} bikes')
    columns[2].metric('Stations whose daily swing exceeds a third of their docks',
                      f'{int((swing/capacity.reindex(swing.index) > 1/3).sum())} of {len(swing)}',
                      help='Swing = highest minus lowest cumulative balance on an average weekday; docks = current station capacity.')
    st.plotly_chart(drift_chart(drift, {drain: ACCENT_2, fill: ACCENT}), width='stretch', key='story_drift')
    st.caption('Average weekday (excluding holidays), 2024–2025. Each line is one station: the running total of returns minus '
               'pickups since midnight. Office-area stations fill in the morning and empty in the evening; residential and '
               'waterfront stations do the opposite. Without trucks, these swings are what leave docks empty or full.')

    st.markdown('#### 2 · What we built')
    steps = st.columns(4)
    steps[0].markdown('**Data**  \n192,760 station-hours of trips (Jan 2024–Sep 2026), airport weather, and four tested '
                      'city datasets: traffic collisions, subway ridership, street events, congestion pricing.')
    steps[1].markdown('**Models**  \nA last-week baseline, SARIMA and ARIMAX, and a Poisson count '
                      'regression that pools all stations.')
    steps[2].markdown('**Validation**  \nFit on 2024, choose on 2025, test once on Jan–Sep 2026. Every forecast is made at '
                      'midnight using only data available then.')
    steps[3].markdown('**Dashboard**  \nLive bike and dock counts, the next 24 hours of forecasts, and estimated bikes or '
                      'docks needed per station.')

    st.markdown('#### 3 · What we found')
    try:
        board = pd.read_csv(STORY/'validation_leaderboard.csv')
        experiments = pd.read_csv(STORY/'predictor_experiments.csv')
    except FileNotFoundError:
        st.warning('Validation summary files are missing; run scripts/build_story_tables.py.')
        return
    by_model = board.set_index('model')
    naive, poisson = by_model.loc['seasonal_naive'], by_model.loc['baseline']
    gain = 1-poisson.pickups_rmse/naive.pickups_rmse
    st.markdown(f'**Repeating last week is a strong benchmark, and a pooled count model beats it by {gain:.0%}.** '
                f'On the same {int(naive.pickups_rows):,} station-hours of 2025, the Poisson regression cut pickup RMSE from '
                f'{naive.pickups_rmse:.2f} to {poisson.pickups_rmse:.2f} trips per hour (Diebold–Mariano p < 0.001). '
                f'SARIMAX with weekly calendar terms came second ({by_model.loc["sarimax_calendar", "pickups_rmse"]:.2f}).')
    st.plotly_chart(leaderboard_chart(board), width='stretch', key='story_leaderboard')
    st.caption('2025 validation year, day-ahead forecasts with trips known through the previous day; every model scored on '
               'identical station-hours. Dotted line: last-week baseline. Models marked SARIMA/ARIMAX are fit one station at a time.')

    st.markdown(f'**Extra city data did not help.** We tested {len(experiments)} external sources on top of the Poisson model. '
                f'The best change in RMSE was {experiments[["pickups_rmse_change_pct", "dropoffs_rmse_change_pct"]].min().min():+.2f}%, '
                f'and none was statistically significant (smallest p = {experiments.min_dm_p.min():.2f}). Recent demand already '
                'carries most of what these sources know.')
    st.dataframe(experiments.assign(**{
        'Pickups RMSE change': experiments.pickups_rmse_change_pct.map('{:+.2f}%'.format),
        'Drop-offs RMSE change': experiments.dropoffs_rmse_change_pct.map('{:+.2f}%'.format),
        'Significance (DM p)': experiments.min_dm_p.map('{:.2f}'.format)})[
        ['label', 'Pickups RMSE change', 'Drop-offs RMSE change', 'Significance (DM p)']].rename(columns={'label': 'Data source'}),
        hide_index=True, width='stretch')

    try:
        test = pd.read_csv(STORY/'test_leaderboard.csv')
        dm = pd.read_csv(STORY/'test_dm_squared.csv')
    except FileNotFoundError:
        st.caption('Test summary files are missing; run scripts/evaluate_test.py.')
    else:
        wide = test.pivot_table(index=['model', 'label'], columns='target', values=['rmse', 'mae', 'rows']).reset_index()
        wide.columns = [f'{target}_{metric}' if target else metric for metric, target in wide.columns]
        rmse = wide.set_index('model')
        p = dm[dm.target.eq('pickups')].set_index(['model', 'vs']).p_value
        test_gain = 1-rmse.loc['poisson', 'pickups_rmse']/rmse.loc['seasonal_naive', 'pickups_rmse']
        team_ns = [rmse.loc[m, 'label'] for m in ['daily_sarima', 'weather_arimax'] if p[(m, 'seasonal_naive')] >= .05]
        st.markdown(f'**The result holds on the untouched 2026 test.** Models were frozen on 2025, then scored once on '
                    f'{int(rmse.loc["poisson", "pickups_rows"]):,} station-hours of January–September 2026. The Poisson regression '
                    f'cut pickup RMSE by {test_gain:.0%} against last week ({rmse.loc["seasonal_naive", "pickups_rmse"]:.2f} → '
                    f'{rmse.loc["poisson", "pickups_rmse"]:.2f}, p < 0.001). SARIMAX with weekly calendar terms was statistically tied '
                    f'with it (p = {p[("sarimax_calendar", "poisson")]:.2f})'
                    + (f', while {" and ".join(team_ns)} did not significantly beat the last-week baseline.' if team_ns else '.'))
        st.plotly_chart(leaderboard_chart(wide), width='stretch', key='story_test_leaderboard')
        st.caption('Test period Jan 1–Sep 30, 2026, trips known through the previous day (the Test evaluation tab\'s default); '
                   'every model scored on identical station-hours. Model choices were recorded in data/story/frozen_choices.json '
                   'before scoring. Diebold–Mariano tests use squared errors averaged per day.')

    st.markdown('#### 4 · What it means')
    st.markdown(
        '- **Demand is mostly rhythm.** Time of day, day of week and the last few weeks of demand explain most of what '
        'any model can predict; that is why a last-week copy is hard to beat and why weather or city events add little.\n'
        '- **Pool what stations share.** One model across all stations, built for counts, beat per-station SARIMA and ARIMAX '
        'by about a quarter, in 2025 and again on the untouched 2026 test. Among the course models, SARIMAX works once '
        'weekly calendar terms are added; it nearly matches the Poisson model.\n'
        '- **Forecasts become decisions.** The Live tab converts the next 24 hours of expected pickups and returns into bikes '
        'or docks needed per station, so a crew can plan truck routes before the morning drain.\n'
        '- **Limits.** Trips only show rides that happened, not riders who found no bike; the eight stations were chosen '
        'with hindsight; weather comes from one airport. Estimated needs are scenarios, not tested operations.')

    st.markdown('#### How to use this site')
    st.markdown(
        '1. **Live station overview** — current bikes and docks, and which stations will need bikes or empty docks in the next '
        '24 hours. Pick the forecast model at the top.\n'
        '2. **Next 24 hours** — choose a station and compare models\' hourly forecasts of pickups, returns and net flow.\n'
        '3. **Test evaluation** — replay January–September 2026: actual trips against each model\'s forecasts, by station and '
        'week, with accuracy scores.')
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
    if page == 'General overview':
        story_view()
    elif page == 'Live station overview':
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
                    saved_results = pd.read_csv(STORY/'test_leaderboard.csv') if aggregate_regime == 'daily_updated' else None
                    for model, label in [('sarimax_calendar', 'SARIMAX + weekly calendar'), ('poisson', 'Poisson regression')]:
                        names.append(label)
                        row = []
                        for target in targets:
                            if saved_results is None:
                                row.extend(['Not evaluated', 'Not evaluated'])
                            elif target == 'net_flow':
                                row.extend([None, None])
                            else:
                                values = saved_results[saved_results.model.eq(model) & saved_results.target.eq(target)].iloc[0]
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
                    st.caption('Hourly comparisons per reported metric — '+ '; '.join(coverage)+'.')
                    st.caption('Bold, shaded values mark the lowest error in each column. “—” means results unavailable; '
                               '“Not evaluated” means the model was not evaluated under this regime.')
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
