import numpy as np
import pandas as pd


def simulate(forecasts, starting_bikes, usable_capacity, threshold=None, redistribution=None):
    """Two event-order scenarios; rejected modeled activity is not observed unmet demand."""
    if not np.isfinite([starting_bikes, usable_capacity]).all() or not 0 <= starting_bikes <= usable_capacity:
        raise ValueError('Starting bikes must be within usable capacity')
    if usable_capacity <= 0:
        raise ValueError('Usable capacity must be positive')
    values = forecasts[['pickups', 'dropoffs']].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Simulation requires finite nonnegative directional forecasts')
    threshold = max(2, int(np.ceil(.1*usable_capacity))) if threshold is None else threshold
    if not 0 <= threshold <= usable_capacity:
        raise ValueError('Threshold is outside usable capacity')
    moves = redistribution or {}
    if any(not isinstance(k, int) or k < 0 or k >= len(forecasts) or not np.isfinite(v)
           for k, v in moves.items()):
        raise ValueError('Redistribution must have finite amounts at valid bin positions')
    rows = []
    for ordering in ['pickups_first', 'returns_first']:
        bikes = float(starting_bikes)
        unconstrained = float(starting_bikes)
        for position, (index, row) in enumerate(forecasts.iterrows()):
            requested = float(moves.get(position, 0))
            applied = np.clip(requested, -bikes, usable_capacity-bikes)
            bikes += applied
            unconstrained += requested + row.dropoffs-row.pickups
            before = bikes
            if ordering == 'pickups_first':
                pickup = min(row.pickups, bikes)
                bikes -= pickup
                middle = bikes
                dropoff = min(row.dropoffs, usable_capacity-bikes)
                bikes += dropoff
            else:
                dropoff = min(row.dropoffs, usable_capacity-bikes)
                bikes += dropoff
                middle = bikes
                pickup = min(row.pickups, bikes)
                bikes -= pickup
            low = min(before, middle, bikes) < threshold
            low_docks = usable_capacity-max(before, middle, bikes) < threshold
            rows.append({'timestamp': index, 'ordering': ordering, 'projected_bikes': bikes,
                'projected_docks': usable_capacity-bikes, 'unconstrained_bikes': unconstrained,
                'rejected_modeled_pickups': row.pickups-pickup,
                'rejected_modeled_returns': row.dropoffs-dropoff,
                'redistribution_requested': requested, 'redistribution_applied': applied,
                'low_bikes': low, 'low_docks': low_docks})
    return pd.DataFrame(rows)


def illustrative_adjustment(forecasts, starting_bikes, usable_capacity, threshold):
    balance = np.r_[0., (forecasts.dropoffs-forecasts.pickups).cumsum().to_numpy()]
    lower = float(threshold-balance.min())
    upper = float(usable_capacity-threshold-balance.max())
    return {'minimum_start_for_net_path': lower, 'maximum_start_for_net_path': upper,
            'illustrative_add': max(0., lower-starting_bikes),
            'illustrative_remove': max(0., starting_bikes-upper),
            'timed_intervention_needed': lower > upper,
            'assumption': 'Net path only; ignores intra-bin ordering, operator moves and unmet demand.'}
