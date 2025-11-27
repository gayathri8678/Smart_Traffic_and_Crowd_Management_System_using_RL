def reward_function(current_count, previous_count=None, action=None):
    """
    Reward function for reinforcement learning in crowd management.
    Parameters:
        current_count (int): Current detected number of people.
        previous_count (int): Last frame person count (optional).
        action (int): Action taken by RL agent (optional).
                      0 = do nothing
                      1 = warning
                      2 = emergency alert
    Returns:
        float: reward score
    """

    # ---- BASE REWARD BASED ON CROWD SIZE ----
    if current_count <= 1:
        reward = 3      # Very safe zone
    elif 2 <= current_count <= 4:
        reward = 1      # Moderate and acceptable
    elif 5 <= current_count <= 7:
        reward = -2     # Risky zone
    else:
        reward = -5     # Overcrowded / emergency

    # ---- PENALIZE CROWD INCREASE ----
    if previous_count is not None and current_count > previous_count:
        reward -= 1  # crowd getting worse

    # ---- BONUS IF ACTION WAS CORRECT ----
    if action is not None:
        # If action triggered AND crowd reduces, add bonus
        if current_count < previous_count and action > 0:
            reward += 3
        
        # Penalize wrong action (example: emergency alert when only 2 people)
        if action == 2 and current_count < 5:
            reward -= 2

    return reward
