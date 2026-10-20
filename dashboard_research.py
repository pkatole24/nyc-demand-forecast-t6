import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from bikebalance.data import DATA


STORY = DATA.parent / 'story'
SOURCE = 'https://github.com/pkatole24/nyc-demand-forecast-t6/blob/master/'
MODEL_COLORS = {'seasonal_naive': '#b7791f', 'daily_sarima': '#4169b1',
                'weather_arimax': '#16877d', 'sarimax_calendar': '#7654a3',
                'poisson': '#176195', 'baseline': '#176195',
                'same_hour_mean_28d': '#7a8088', 'sarima': '#b66548'}


def load_results(period):
    if period == 'test':
        frame = pd.read_csv(STORY / 'test_leaderboard.csv')
    elif period == 'validation':
        wide = pd.read_csv(STORY / 'validation_leaderboard.csv')
        frame = pd.concat([wide[['model', 'label']].assign(
            target=target, mae=wide[f'{target}_mae'], rmse=wide[f'{target}_rmse'],
            rows=wide[f'{target}_rows']) for target in ('pickups', 'dropoffs')], ignore_index=True)
        frame['model'] = frame.model.replace({'baseline': 'poisson'})
    else:
        raise ValueError('Unknown research period')
    if frame.duplicated(['model', 'target']).any():
        raise ValueError('Research results contain duplicate model/target rows')
    if not np.isfinite(frame[['mae', 'rmse', 'rows']].to_numpy(float)).all() or (frame[['mae', 'rmse']] < 0).any().any():
        raise ValueError('Research results contain invalid scores')
    if (frame.rows <= 0).any() or (frame.rows % 1 != 0).any():
        raise ValueError('Research results need positive integer scored counts')
    if frame.groupby('target').rows.nunique().gt(1).any():
        raise ValueError('Research models do not have equal reported support within each target')
    return frame.assign(period=period, regime='daily_updated', horizon='Next local day')


def comparison_chart(frame, metric='rmse'):
    order = frame.groupby(['model', 'label'])[metric].mean().sort_values(ascending=False).index
    figure = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=.08,
                           subplot_titles=('Pickups', 'Returns'))
    maximum = frame[metric].max() * 1.35
    for column, target in enumerate(('pickups', 'dropoffs'), 1):
        values = frame[frame.target.eq(target)].set_index(['model', 'label']).reindex(order)
        figure.add_trace(go.Scatter(x=values[metric], y=order.get_level_values('label'),
            mode='markers+text', text=values[metric].map('{:.2f}'.format), textposition='middle right', cliponaxis=False,
            marker=dict(size=11, color=[MODEL_COLORS.get(model, '#7a8088') for model, _ in order]),
            showlegend=False, hovertemplate='%{y}<br>'+metric.upper()+': %{x:.2f} trips/hour<extra></extra>'), row=1, col=column)
        baseline = values.loc[('seasonal_naive', 'Last-week baseline'), metric]
        figure.add_vline(x=baseline, line_dash='dot', line_color='#8a929b', row=1, col=column)
    figure.update_layout(height=max(310, len(order)*40+100), margin=dict(l=10, r=25, t=40, b=20),
                         template='plotly_white', font=dict(color='#243b4b'), paper_bgcolor='rgba(0,0,0,0)')
    figure.update_xaxes(title_text=f'{metric.upper()} (trips/h)', range=[0, maximum], gridcolor='#e8edf1')
    return figure


def result_context(frame):
    counts = frame.groupby('target').rows.first()
    period = 'Jan–Sep 2026 test' if frame.period.iloc[0] == 'test' else '2025 validation'
    st.caption(f'{period} · Next-day forecasts assuming trips through the prior day are available. '
               f'Shared reported support: {int(counts["pickups"]):,} pickup and {int(counts["dropoffs"]):,} return station-hours. '
               'Lower error is better. Dotted reference: last-week baseline. Scores are saved results, not live forecast accuracy.')


def source_link(path, label='Source'):
    st.markdown(f'[{label}]({SOURCE}{path})')


def weekday_drift(panel):
    """Average bikes gained (+) or lost (-) since midnight on a weekday, by station and hour."""
    frame = panel
    weekdays = frame[(frame.weekday < 5) & ~frame.holiday & frame.partition.ne('test')]
    hourly = weekdays.assign(net_flow=weekdays.net_flow.astype(float)).groupby(['station_id', 'hour']).net_flow.mean().unstack()
    return hourly.cumsum(axis=1)


def drift_chart(drift, highlight, stations):
    figure = go.Figure()
    hours = list(drift.columns)
    for station, values in drift.iterrows():
        color = highlight.get(station)
        name = stations.loc[station, 'station_name']
        figure.add_trace(go.Scatter(x=hours, y=values.to_numpy(), mode='lines', name=name,
            line=dict(color=color or "#c3c2b7", width=3 if color else 1.5), showlegend=False,
            hovertemplate=f'{name}<br>By end of %{{x}}:00: %{{y:+.0f}} bikes<extra></extra>'))
        if color:
            figure.add_annotation(x=hours[-1], y=values.iloc[-1], text=name, showarrow=False, xanchor='left',
                                  xshift=6, font=dict(color=color, size=12))
    figure.add_hline(y=0, line=dict(color="#52514e", width=1))
    figure.update_layout(height=360, margin=dict(l=10, r=170, t=10, b=40), template='plotly_white',
        xaxis=dict(title='Hour of the day (New York time)', tickvals=list(range(0, 24, 3)),
                   ticktext=[f'{h}:00' for h in range(0, 24, 3)]),
        yaxis=dict(title='Bikes gained (+) or lost (−) since midnight'))
    return figure


def story_view(panel, stations):
    st.subheader('Can we predict where bikes will run out, a day ahead?')
    st.write('Citi Bike stations empty out and fill up in predictable waves. When a station has no bikes, riders walk away; '
             'when it has no empty docks, riders cannot return their bike. Operators move bikes by truck to prevent both, '
             'and they need to know **where and when** the shortages will happen. BikeBalance NYC forecasts hourly '
             'pickups and returns at eight Manhattan stations for the next day, and turns those forecasts into '
             'estimated rebalancing needs.')

    st.markdown('#### 1 · The problem: stations drain and fill on a daily rhythm')
    drift = weekday_drift(panel)
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
    st.plotly_chart(drift_chart(drift, {drain: "#eb6834", fill: "#2a78d6"}, stations), width='stretch', key='story_drift')
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
    except (OSError, ValueError, KeyError):
        st.warning('Validation summary files are missing; run scripts/build_story_tables.py.')
        return
    by_model = board.set_index('model')
    naive, poisson = by_model.loc['seasonal_naive'], by_model.loc['baseline']
    gain = 1-poisson.pickups_rmse/naive.pickups_rmse
    st.markdown(f'**Repeating last week is a strong benchmark, and a pooled count model beats it by {gain:.0%}.** '
                f'On the same {int(naive.pickups_rows):,} station-hours of 2025, the Poisson regression cut pickup RMSE from '
                f'{naive.pickups_rmse:.2f} to {poisson.pickups_rmse:.2f} trips per hour (Diebold–Mariano p < 0.001). '
                f'SARIMAX with weekly calendar terms came second ({by_model.loc["sarimax_calendar", "pickups_rmse"]:.2f}).')
    st.plotly_chart(comparison_chart(load_results('validation')), width='stretch', key='story_leaderboard')
    st.caption('2025 validation year, day-ahead forecasts with trips known through the previous day; every model scored on '
               'identical station-hours. Dotted line: last-week baseline. Models marked SARIMA/ARIMAX are fit one station at a time.')

    st.markdown(f'**Extra city data did not help.** We tested {len(experiments)} external sources on top of the Poisson model. '
                f'The best change in RMSE was {experiments[["pickups_rmse_change_pct", "dropoffs_rmse_change_pct"]].min().min():+.2f}%, '
                f'and none was statistically significant (smallest p = {experiments.min_dm_p.min():.2f}). Recent demand already '
                'carries most of what these sources know.')
    with st.expander('City-data RMSE changes and significance', expanded=False):
        st.dataframe(experiments.assign(**{
            'Pickups RMSE change': experiments.pickups_rmse_change_pct.map('{:+.2f}%'.format),
            'Drop-offs RMSE change': experiments.dropoffs_rmse_change_pct.map('{:+.2f}%'.format),
            'Significance (DM p)': experiments.min_dm_p.map('{:.2f}'.format)})[
            ['label', 'Pickups RMSE change', 'Drop-offs RMSE change', 'Significance (DM p)']].rename(columns={'label': 'Data source'}),
            hide_index=True, width='stretch')

    try:
        test = pd.read_csv(STORY/'test_leaderboard.csv')
        dm = pd.read_csv(STORY/'test_dm_squared.csv')
    except (OSError, ValueError, KeyError):
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
        st.plotly_chart(comparison_chart(load_results('test')), width='stretch', key='story_test_leaderboard')
        st.caption('Test period Jan 1–Sep 30, 2026, trips known through the previous day (the Research & validation tab\'s default); '
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
        '1. **Live station outlook** — current bikes and docks, and which stations will need bikes or empty docks in the next '
        '24 hours. Pick the forecast model at the top.\n'
        '2. **Forecast explorer** — choose a station and compare models\' hourly forecasts of pickups, returns and net flow.\n'
        '3. **Research & validation** — replay January–September 2026: actual trips against each model\'s forecasts, by station and '
        'week, with accuracy scores.')
def research_view():
    st.subheader('Research & validation')
    st.write('Each study answers a different question. Compare models within the same horizon, information regime and scored support.')
    period = st.radio('Evidence period', ['test', 'validation'], horizontal=True,
                      format_func=lambda x: '2026 held-out test' if x == 'test' else '2025 model selection', key='research_period')
    try:
        frame = load_results(period)
        metric = st.radio('Error measure', ['rmse', 'mae'], horizontal=True, format_func=str.upper, key='research_metric')
        st.plotly_chart(comparison_chart(frame, metric), width='stretch', key='research_comparison')
        result_context(frame)
        st.caption('MAE is the average absolute hourly error. RMSE puts more weight on large errors. Neither measures missed rides or balancing savings.')
        with st.expander('Exact scores and model specifications'):
            st.dataframe(frame[['label', 'target', 'mae', 'rmse', 'rows']].rename(columns={
                'label': 'Model', 'target': 'Target', 'mae': 'MAE', 'rmse': 'RMSE', 'rows': 'Scored station-hours'}),
                hide_index=True, width='stretch')
            source_link(f'data/story/{period}_leaderboard.csv', 'Saved comparison')
            if period == 'validation':
                specs = pd.read_csv(STORY/'validation_leaderboard.csv')
                st.dataframe(specs[['label', 'specification']], hide_index=True, width='stretch')
            else:
                source_link('data/story/frozen_choices.json', 'Recorded model choices')
                source_link('data/story/test_dm_squared.csv', 'Daily squared-loss comparisons')
        st.caption('2024 initial fitting → 2025 selection → Jan–Sep 2026 test. Evaluation updates use only prior observations '
                   'within each study’s information regime. Test scores do not certify live deployment.')
    except (OSError, ValueError, KeyError):
        st.warning('This saved comparison is unavailable or invalid. The historical replay remains available below.')
