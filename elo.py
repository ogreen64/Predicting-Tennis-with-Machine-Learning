import numpy as np

def add_elos(df, k=32, decay_rate=0.00025):
    df = df.copy()
    
    # Elo state trackers
    overall_elo = {}
    surface_elo = {}
    last_played_date = {}
    last_played_date_surface = {}
    
    # New feature columns
    df["A Elo"] = np.nan
    df["B Elo"] = np.nan
    df["A Surface Elo"] = np.nan
    df["B Surface Elo"] = np.nan
    df["Elo Prediction"] = np.nan           # only store A's win probability
    df["Surface Elo Prediction"] = np.nan   # only store A's surface win probability
    
    for i, row in df.iterrows():
        player_a, player_b = row["Player A"], row["Player B"]
        surface, date, a_won = row["Surface"], row["Date"], row["A Won"]
        
        # Initialize ratings
        if player_a not in overall_elo: overall_elo[player_a] = 1500
        if player_b not in overall_elo: overall_elo[player_b] = 1500
        if (player_a, surface) not in surface_elo: surface_elo[(player_a, surface)] = 1500
        if (player_b, surface) not in surface_elo: surface_elo[(player_b, surface)] = 1500
        
        # Apply decay to overall Elo
        for player in [player_a, player_b]:
            if player in last_played_date:
                days = (date - last_played_date[player]).days
                decay = np.exp(-decay_rate * days)
                overall_elo[player] = 1500 + (overall_elo[player] - 1500) * decay
        
        # Apply decay to surface Elo
        for player in [player_a, player_b]:
            if (player, surface) in last_played_date_surface:
                days = (date - last_played_date_surface[(player, surface)]).days
                decay = np.exp(-decay_rate * days)
                surface_elo[(player, surface)] = 1500 + (surface_elo[(player, surface)] - 1500) * decay
        
        # Grab pre-match Elo ratings
        elo_a, elo_b = overall_elo[player_a], overall_elo[player_b]
        surface_elo_a, surface_elo_b = surface_elo[(player_a, surface)], surface_elo[(player_b, surface)]
        
        # Predicted win probabilities (store only A’s)
        prediction = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        surface_prediction = 1 / (1 + 10 ** ((surface_elo_b - surface_elo_a) / 400))
        
        # Add pre-match features to df
        df.at[i, "A Elo"] = elo_a
        df.at[i, "B Elo"] = elo_b
        df.at[i, "A Surface Elo"] = surface_elo_a
        df.at[i, "B Surface Elo"] = surface_elo_b
        df.at[i, "Elo Prediction"] = prediction
        df.at[i, "Surface Elo Prediction"] = surface_prediction
        
        # Update ratings after the match
        actual = 1 if a_won else 0
        overall_elo[player_a] += k * (actual - prediction)
        overall_elo[player_b] += k * ((1 - actual) - (1 - prediction))
        surface_elo[(player_a, surface)] += k * (actual - surface_prediction)
        surface_elo[(player_b, surface)] += k * ((1 - actual) - surface_prediction)
        
        # Update last played dates
        last_played_date[player_a] = date
        last_played_date[player_b] = date
        last_played_date_surface[(player_a, surface)] = date
        last_played_date_surface[(player_b, surface)] = date
    
    return df