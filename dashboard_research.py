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
                'same_hour_mean_28d': '#7a8088', 'sarima': '#b66548',
                'poisson_weather_forecast': '#0b3a5e', 'sarimax_weather_forecast': '#4b2c7f', 'chronos': '#c2410c'}


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


HOUR_LABELS = {'seasonal_naive': 'Last-week baseline', 'chronos': 'Chronos-Bolt (AI, zero-shot)',
               'sarimax_weather_forecast': 'SARIMAX + calendar + weather forecast',
               'poisson_weather_forecast': 'Poisson + weather forecast'}


def at_a_glance():
    """Three headline numbers from the frozen 2026 test."""
    try:
        test = pd.read_csv(STORY / 'test_leaderboard.csv')
    except OSError:
        return
    rmse = test[test.target.eq('pickups')].set_index('model').rmse
    naive = rmse['seasonal_naive']
    columns = st.columns(3)
    columns[0].metric('Best model · error vs copying last week', f'{rmse["poisson_weather_forecast"]/naive-1:.0%}',
                      help='Pickup RMSE, Poisson + weather forecast, January–September 2026 (never used for model choice).')
    columns[1].metric('Best live model · error vs copying last week', f'{rmse["sarimax_weather_forecast"]/naive-1:.0%}',
                      help='SARIMAX + calendar + weather forecast, the default model in the live tabs.')
    columns[2].metric('Weather forecasts · extra error reduction', f'{rmse["poisson_weather_forecast"]/rmse["poisson"]-1:.0%}',
                      help='Poisson with vs without Open-Meteo forecasts issued two days ahead. City data added nothing.')


def error_by_hour_section():
    try:
        frame = pd.read_csv(STORY / 'error_by_hour.csv')
    except OSError:
        return
    pickups = frame[frame.target.eq('pickups')].pivot(index='hour', columns='model', values='mae')
    peak = pickups.seasonal_naive.idxmax()
    st.markdown(f'**Where the forecasts still miss: the rush hours.** At night every model is nearly exact. The misses pile up '
                f'at the commuter peaks, exactly when stations empty and fill. At {peak % 12 or 12} {"pm" if peak >= 12 else "am"} '
                f'on weekdays, copying last week misses by {pickups.seasonal_naive[peak]:.1f} pickups per hour on average; '
                f'Poisson + weather forecast by {pickups.poisson_weather_forecast[peak]:.1f}.')
    figure = go.Figure()
    for model, label in HOUR_LABELS.items():
        figure.add_trace(go.Scatter(x=pickups.index, y=pickups[model], mode='lines', name=label,
            line=dict(color=MODEL_COLORS[model], width=3 if model != 'seasonal_naive' else 2,
                      dash='dot' if model == 'seasonal_naive' else 'solid'),
            hovertemplate=label + '<br>%{x}:00 · %{y:.1f} pickups/hour<extra></extra>'))
    figure.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=40), template='plotly_white',
                         legend=dict(orientation='h', y=1.12), paper_bgcolor='rgba(0,0,0,0)',
                         xaxis=dict(title='Hour of the day (New York time)', tickvals=list(range(0, 24, 3)),
                                    ticktext=[f'{h}:00' for h in range(0, 24, 3)]),
                         yaxis=dict(title='Average miss (pickups per hour)', rangemode='tozero', gridcolor='#e8edf1'))
    st.plotly_chart(figure, width='stretch', key='story_error_by_hour')
    st.caption('Weekday pickups, January–September 2026 test, mean absolute error by hour; identical station-hours for every model.')


def p_text(p):
    return 'p < 0.001' if p < .001 else f'p = {p:.2f}' if p >= .01 else f'p = {p:.3f}'


def ai_comparison(rmse, dm):
    """AI-generated (zero-shot foundation model) forecasts against the conventional models, 2026 test."""
    if 'chronos' not in rmse.index:
        return
    p = dm[dm.target.eq('pickups')].set_index(['model', 'vs']).p_value
    ai, best = rmse.loc['chronos', 'pickups_rmse'], rmse.loc['poisson_weather_forecast', 'pickups_rmse']
    naive, sarimax = rmse.loc['seasonal_naive', 'pickups_rmse'], rmse.loc['sarimax_calendar', 'pickups_rmse']
    p_best, p_sarimax = p[('chronos', 'poisson_weather_forecast')], p[('chronos', 'sarimax_calendar')]
    p_returns = dm[dm.target.eq('dropoffs')].set_index(['model', 'vs']).p_value[('chronos', 'poisson_weather_forecast')]
    returns_ai = rmse.loc['chronos', 'dropoffs_rmse'] < rmse.loc['poisson_weather_forecast', 'dropoffs_rmse']
    versus_best = (f'beat our best model, Poisson + weather forecast ({best:.2f}, {p_text(p_best)})' if ai < best and p_best < .05
                   else f'was statistically tied with our best model, Poisson + weather forecast ({best:.2f}, {p_text(p_best)})' if p_best >= .05
                   else f'did not beat our best model, Poisson + weather forecast ({best:.2f}, {p_text(p_best)})')
    versus_sarimax = ('better than' if ai < sarimax else 'worse than') + f' SARIMAX + weekly calendar ({sarimax:.2f}, {p_text(p_sarimax)})'
    st.markdown(f'**AI vs conventional models.** A pretrained AI forecasting model, Chronos-Bolt, used zero-shot with no training '
                f'on our data and no weather inputs, reached a 2026 pickup RMSE of {ai:.2f}: {1-ai/naive:.0%} better than copying '
                f'last week, {versus_sarimax}. It {versus_best}; for returns, '
                f'{"it was better" if returns_ai else "Poisson + weather forecast was better"} '
                f'({"significant" if p_returns < .05 else "not significant"}, {p_text(p_returns)}). '
                "We didn't include it in the live dashboard because it would need PyTorch.")


def ai_takeaway():
    try:
        test = pd.read_csv(STORY / 'test_leaderboard.csv')
    except OSError:
        return ''
    rmse = test[test.target.eq('pickups')].set_index('model').rmse
    if 'chronos' not in rmse.index:
        return ''
    gap = rmse['chronos'] / rmse['poisson_weather_forecast'] - 1
    verdict = (f'Chronos-Bolt beat our best tuned model by {-gap:.0%}' if gap < 0
               else f'Chronos-Bolt came within {gap:.0%} of our best tuned model')
    return ('- **AI is a strong starting point, not a replacement for domain structure.** Without any training on Citi Bike data, '
            f'{verdict}, which needed calendar terms, recent demand and weather forecasts to get there.\n')


def top_models_section():
    st.markdown('**What makes the top three models the best.** All three capture the same three things: each station\'s '
                'daily shape, the weekly rhythm (weekdays versus weekends and holidays), and how busy the station has been '
                'recently. The models that missed one of these, such as SARIMA with only a daily cycle or ARIMAX whose memory '
                'fades within hours, did not beat copying last week. Each of the three gets there differently:')
    st.markdown(
        '- **Poisson + weather forecast.** One count model shared by all eight stations. Each station keeps its own pattern '
        'for every hour of the day, while weekday, month, weekend and holiday effects are learned from all stations together, '
        'from eight times more data. It anchors each forecast on recent demand (the same hour yesterday, last week and over '
        'the last 28 days), never predicts negative trips, and adds the weather forecast: a forecast of rain in an hour means '
        'roughly 28% fewer trips.\n'
        '- **Chronos-Bolt (AI, zero-shot).** A model pretrained on a very large collection of time series. Reading the last '
        'twelve weeks of a station\'s history, it recognizes the daily and weekly repetition and the current level on its own, '
        'with no training on Citi Bike data. **Weather cannot be included in this model.** Chronos-Bolt is a univariate '
        'model: it was pretrained to read one series at a time, the history of the value being forecast, and it has no input '
        'for extra variables such as temperature or rain. Adding weather would mean switching to a different model built to '
        'accept outside variables (Amazon\'s newer Chronos-2 does) and validating it again, which was beyond this study. '
        'Without weather it cannot anticipate rainy days, which is where Poisson + weather forecast keeps its edge.\n'
        '- **SARIMAX + calendar + weather forecast.** Its seasonal terms work like a smoothed average of the same hour over '
        'recent days, its calendar terms correct for weekdays, weekends and holidays, and the weather forecast adds the same '
        'rain and temperature signal. As a state-space model it can keep forecasting across days with no new trip data, which '
        'makes it the best model that can run live.')
    try:
        test = pd.read_csv(STORY / 'test_leaderboard.csv')
        rmse = test[test.target.eq('pickups')].set_index('model').rmse
        gap = f'{rmse["sarimax_weather_forecast"] / rmse["poisson_weather_forecast"] - 1:.1%}'
    except (OSError, KeyError):
        gap = 'a few percent'
    st.info('**Why the Poisson model is not in the live dashboard.** Citi Bike publishes its trip data once a month, so the '
            'live site only has trips up to the end of the last published month. The Poisson model needs much more frequent '
            'data: each forecast uses the previous day\'s and the previous week\'s trips at every station, and those inputs do '
            'not exist yet for today. SARIMAX can carry its forecast forward from the last published hour, so it runs live '
            'instead. With daily trip data, Poisson + weather forecast would be the model to deploy.\n\n'
            '**The best model on paper is not always the best model in practice.** A test score assumes every input is '
            'available on time; a business can only run a model on the data it actually receives, when it receives it. Here '
            f'the most accurate model depends on daily trip data the operator\'s public feed does not provide, so the practical '
            f'choice is the model that works with monthly data and costs only {gap} more error on pickups. Companies often face '
            'the same trade-off: they deploy a slightly less accurate model because the data for the best one arrives too late, '
            'is too expensive to collect, or is not available at all. Getting fresher data can be worth more than a better model.')


def story_view(panel, stations):
    st.subheader('Can we predict where bikes will run out, a day ahead?')
    st.write('Citi Bike stations empty out and fill up in predictable waves. When a station has no bikes, riders walk away; '
             'when it has no empty docks, riders cannot return their bike. Operators move bikes by truck to prevent both, '
             'and they need to know **where and when** the shortages will happen. BikeBalance NYC forecasts hourly '
             'pickups and returns at eight Manhattan stations for the next day, and turns those forecasts into '
             'estimated rebalancing needs.')
    at_a_glance()

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

    st.markdown('#### 2 · How we built and tested it')
    st.markdown('**Data.** 192,760 station-hours of Citi Bike trips (January 2024–September 2026) at eight Manhattan stations; '
                'observed LaGuardia weather; Open-Meteo weather forecasts issued two days ahead; four city datasets (traffic '
                'collisions, subway ridership, permitted street events, congestion pricing); and the live Citi Bike station feed.')
    st.markdown('**Models, from simple to complex.** Each one tests one idea about what drives demand.')
    st.markdown(
        '| Model | How it forecasts | Why we tried it |\n| --- | --- | --- |\n'
        '| Last-week baseline | Same hour, one week earlier | The bar every model must clear |\n'
        '| Same-hour 28-day average | Average of that hour over the last four weeks | A smoother benchmark |\n'
        '| SARIMA | Per station, with 24-hour seasonal differencing | Course model for the daily cycle |\n'
        '| ARIMAX + past weather | ARIMA + calendar + observed weather from 48 hours earlier | Does recent weather help? |\n'
        '| SARIMAX + weekly calendar | SARIMA + weekday and weekend/holiday terms | Adds the weekly cycle SARIMA misses |\n'
        '| Poisson regression | One count model for all stations: each station\'s hourly pattern + recent demand | Pools stations; built for counts with many zeros |\n'
        '| + weather forecast | Poisson or SARIMAX + the forecast issued two days ahead (temperature, wind, rain) | The weather actually known at midnight |\n'
        '| Chronos-Bolt (AI) | A pretrained AI foundation model, used zero-shot: it reads the last 12 weeks of a station\'s history, with no training on our data | Does a general-purpose AI model beat our tuned models? |')
    st.markdown('**Key decisions behind the validation.**')
    decisions = st.columns(2, gap='large')
    decisions[0].markdown(
        '- **Forecast at midnight for the whole next day.** Crews plan routes before the 6–7 am drain; a one-hour-ahead '
        'forecast is easier but arrives too late to move bikes.\n'
        '- **Use only what is known at midnight.** No actual future weather: it would cut error about 9%, so it appears only '
        'as a labelled ceiling. Weather forecasts are those issued two days ahead; city data is lagged by its publication delay.\n'
        '- **Missing is not zero.** Hours lost to clock changes stay missing; nothing is filled in.')
    decisions[1].markdown(
        '- **Chronological split, test once.** Fit on 2024, choose on 2025, score once on January–September 2026. Choices were '
        'recorded in data/story before any test score was computed.\n'
        '- **Same rows for every model.** Each comparison scores all models on identical station-hours.\n'
        '- **Error measures and significance.** RMSE weighs the large misses that empty a station; MAE gives the typical miss. '
        'Diebold–Mariano tests on daily losses check that a gain is not luck.')

    st.markdown('#### 3 · What we found')
    try:
        board = pd.read_csv(STORY/'validation_leaderboard.csv')
        experiments = pd.read_csv(STORY/'predictor_experiments.csv')
    except (OSError, ValueError, KeyError):
        st.warning('Validation summary files are missing; run scripts/build_story_tables.py.')
        return
    by_model = board.set_index('model')
    naive, poisson = by_model.loc['seasonal_naive'], by_model.loc['baseline']
    weather = by_model.loc['poisson_weather_forecast']
    gain = 1-poisson.pickups_rmse/naive.pickups_rmse
    weather_gain = 1-weather.pickups_rmse/naive.pickups_rmse
    st.markdown(f'**Repeating last week is a strong benchmark; calendar and recent demand beat it by {gain:.0%}, and the '
                f'weather forecast takes it to {weather_gain:.0%}.** On the same {int(naive.pickups_rows):,} station-hours of 2025, '
                f'the Poisson regression cut pickup RMSE from {naive.pickups_rmse:.2f} to {poisson.pickups_rmse:.2f} trips per hour, '
                f'and {weather.pickups_rmse:.2f} with weather forecasts (Diebold–Mariano p < 0.001). Among the course models, '
                f'SARIMAX with weekly calendar terms reached {by_model.loc["sarimax_calendar", "pickups_rmse"]:.2f}, and '
                f'{by_model.loc["sarimax_weather_forecast", "pickups_rmse"]:.2f} with weather forecasts.')
    st.plotly_chart(comparison_chart(load_results('validation')), width='stretch', key='story_leaderboard')
    st.caption('2025 validation year, day-ahead forecasts with trips known through the previous day; every model scored on '
               'identical station-hours. Dotted line: last-week baseline. Models marked SARIMA/ARIMAX are fit one station at a time.')

    st.markdown(f'**City data did not help.** We tested {len(experiments)} city sources on top of the Poisson model. '
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
    p_gain = 1-weather.pickups_rmse/poisson.pickups_rmse
    s_gain = 1-by_model.loc['sarimax_weather_forecast', 'pickups_rmse']/by_model.loc['sarimax_calendar', 'pickups_rmse']
    st.markdown(f'**Weather forecasts did help.** Adding the Open-Meteo forecast issued two days ahead (temperature, wind and '
                f'rain, so it is known at midnight) cut 2025 pickup RMSE by a further {p_gain:.1%} for Poisson and {s_gain:.1%} '
                'for SARIMAX (p < 0.01). That is about half of what perfect knowledge of the actual weather would give (about 9%). '
                'A forecast of rain in an hour means roughly 28% fewer trips.')

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
        naive_rmse = rmse.loc['seasonal_naive', 'pickups_rmse']
        test_gain = 1-rmse.loc['poisson', 'pickups_rmse']/naive_rmse
        weather_test_gain = 1-rmse.loc['poisson_weather_forecast', 'pickups_rmse']/naive_rmse
        team_ns = [rmse.loc[m, 'label'] for m in ['daily_sarima', 'weather_arimax'] if p[(m, 'seasonal_naive')] >= .05]
        st.markdown(f'**The results hold on the untouched 2026 test.** Models were frozen on 2025, then scored once on '
                    f'{int(rmse.loc["poisson", "pickups_rows"]):,} station-hours of January–September 2026. Against last week '
                    f'({naive_rmse:.2f}), Poisson + weather forecast cut pickup RMSE by {weather_test_gain:.0%} '
                    f'(to {rmse.loc["poisson_weather_forecast", "pickups_rmse"]:.2f}) and Poisson without weather by {test_gain:.0%} '
                    f'(to {rmse.loc["poisson", "pickups_rmse"]:.2f}), both p < 0.001. The weather forecast improved both Poisson and '
                    f'SARIMAX (p < 0.001), and SARIMAX + calendar + weather forecast ({rmse.loc["sarimax_weather_forecast", "pickups_rmse"]:.2f}) '
                    f'beat Poisson without weather (p = {p[("sarimax_weather_forecast", "poisson")]:.4f})'
                    + (f', while {" and ".join(team_ns)} did not significantly beat the last-week baseline.' if team_ns else '.'))
        ai_comparison(rmse, dm)
        st.plotly_chart(comparison_chart(load_results('test')), width='stretch', key='story_test_leaderboard')
        st.caption('Test period Jan 1–Sep 30, 2026, trips known through the previous day (the Research & validation tab\'s default); '
                   'every model scored on identical station-hours. Model choices were recorded in data/story/frozen_choices.json, '
                   'frozen_choices_weather_forecast.json and frozen_choices_ai.json before scoring. Diebold–Mariano tests use squared errors averaged per day.')
    top_models_section()
    error_by_hour_section()

    st.markdown('#### 4 · What it means')
    st.markdown(
        '- **Demand is mostly rhythm.** Time of day, day of week and the last few weeks of demand explain most of what '
        'any model can predict; that is why a last-week copy is hard to beat and why city data added nothing.\n'
        '- **Model the weekly rhythm.** Both the pooled Poisson model and a per-station SARIMAX with weekly calendar terms '
        'beat copying last week by about a quarter, in 2025 and again on the untouched 2026 test; models without the weekly '
        'cycle did not. Pooling helps, but the weekly rhythm matters more.\n'
        '- **Weather forecasts add the rest we could find.** The forecast available at midnight improved both Poisson and '
        'SARIMAX on the 2026 test, taking the gain over last week from about 25% to about 30%.\n'
        f'{ai_takeaway()}'
        '- **Forecasts become decisions.** The live tabs turn the next 24 hours of expected pickups and returns into bikes or '
        'docks needed per station, so a crew can plan truck routes before the morning drain. They default to SARIMAX + '
        'calendar + weather forecast, the best model that works with monthly-published trip data.\n'
        '- **Limits.** Trips only show rides that happened, not riders who found no bike; the eight stations were chosen '
        'with hindsight; weather comes from one airport. Estimated needs are scenarios, not tested operations.')

    st.markdown('#### How to use this site')
    st.markdown(
        '1. **Live station outlook** — current bikes and docks, and which stations will need bikes or empty docks in the next '
        '24 hours. Pick the forecast model at the top.\n'
        '2. **Forecast explorer** — choose a station and compare models\' hourly forecasts of pickups, returns and net flow.\n'
        '3. **Research & validation** — every model\'s 2025 and 2026 scores side by side, plus a replay of January–September '
        '2026: actual trips against the forecasts, by station and week.')
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
                source_link('data/story/frozen_choices_weather_forecast.json', 'Recorded weather-forecast model choices')
                source_link('data/story/test_dm_squared.csv', 'Daily squared-loss comparisons')
        st.caption('2024 initial fitting → 2025 selection → Jan–Sep 2026 test. Evaluation updates use only prior observations '
                   'within each study’s information regime. Test scores do not certify live deployment.')
    except (OSError, ValueError, KeyError):
        st.warning('This saved comparison is unavailable or invalid. The historical replay remains available below.')
