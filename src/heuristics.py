from src.predict import idx_to_coord

def get_smart_explanation(move_dict, rank_mode, significant_stones, rival_move):
    """ 
    Generates dynamic, context-aware text using Geometry, XAI Analysis, and Counterfactuals.
    
    Args:
        move_dict: Dictionary with 'move' (text), 'row', 'col', 'confidence'.
        rank_mode: String from UI ('Novice', 'Intermediate', 'Expert').
        significant_stones: List of important stones identified by XAI.
        rival_move: The predicted opponent response (coordinate string).
    """
    row, col = move_dict['row'], move_dict['col']
    coord_text = move_dict['move']
    
    # --- 1. COGNITIVE CHUNKING (Analyze XAI Data) ---
    # We look at the top 3 stones the AI 'looked at' to make this decision
    focus_stones = significant_stones[:3] 
    
    # Classify the relationship between the Move and the Focus Stones
    nearby_enemies = []
    nearby_allies = []
    
    for stone in focus_stones:
        sr, sc = stone['coords']
        dist = abs(sr - row) + abs(sc - col) # Manhattan distance
        
        # If the stone is close enough to be "Tactically Related" (Distance < 5)
        if dist < 5:
            # Note: Ideally pass player_color to check ally/enemy accurately. 
            # Assuming mixed context for now or inferring based on simple distance logic
            # For simplicity in this heuristic: simple proximity check is powerful enough.
            # We treat close stones as "Local Context".
            if stone['color'] != 'Empty':
                nearby_enemies.append(stone) # Grouping all interacting stones

    # --- 2. DETERMINE INTENT (The "Why") ---
    intent = "Global Strategy"
    detail = ""
    
    # CASE A: Contact Fight / Local Response (Very close interaction)
    if len(nearby_enemies) > 0:
        target_stone = nearby_enemies[0]
        target_coord = idx_to_coord(target_stone['coords'][0], target_stone['coords'][1])
        target_color = target_stone['color']
        
        intent = "Local Combat / Shape"
        detail = f"The AI is reacting directly to the {target_color} stone at **{target_coord}**. This move creates shape and prevents being sealed in or cut."
        
    # CASE B: Direction of Play (No nearby stones lit up significantly)
    else:
        # Fallback to Geometry Logic
        is_corner = (row < 4 or row > 14) and (col < 4 or col > 14)
        is_third_line = (row == 2 or row == 16 or col == 2 or col == 16)
        is_fourth_line = (row == 3 or row == 15 or col == 3 or col == 15)
        
        if is_corner:
            intent = "Territorial Profit"
            detail = "The AI values the corner cash here. It is the largest empty area on the board."
        elif is_third_line:
            intent = "Stability / Base"
            detail = "This extension settles your group on the side, removing the opponent's attack potential."
        elif is_fourth_line:
            intent = "Influence / Moyo"
            detail = "This move expands your framework towards the center, limiting the opponent's potential."
        else:
            intent = "Center Control"
            detail = "Jumping out to the center keeps your stones connected and affects the global flow."

    # --- 3. GENERATE TEXT BY RANK (Scaffolding) ---
    
    if "Novice" in rank_mode:
        return f"""
        **Simple Goal:** {intent}
        
        **Why:** {detail}
        
        **Warning:** If you don't play here, the opponent might play at **{rival_move}** next.
        """
        
    elif "Intermediate" in rank_mode:
        return f"""
        **Concept:** {intent}
        
        **Analysis:** {detail}
        
        **Reading:** The AI predicts the opponent's best response is **{rival_move}**. Playing **{coord_text}** now keeps the initiative (Sente).
        """
        
    else: # Expert
        # Expert mode focuses on Aji, Efficiency, and Honte
        xai_note = f"Visual attention on **{len(focus_stones)} key stones** suggests complex local aji." if len(focus_stones) > 1 else "Focus is on global direction."
        
        return f"""
        **Strategic Focus:** {intent}
        
        **Heuristic:** {detail} {xai_note}
        
        **Branching:** This move anticipates **{rival_move}** as the *honte* (proper move). Deviating would likely result in a loss of local aji or efficiency.
        """