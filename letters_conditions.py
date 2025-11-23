import math

def distance_y(point1, point2):
    return abs(point1.y - point2.y)

def distance_x(point1, point2):
    return abs(point1.x - point2.x)

def doigts_tendus(fingers_tips, fingers_dip):
    return all(f_tip.y < f_dip.y for f_tip, f_dip in zip(fingers_tips, fingers_dip))

def doigts_repliés(fingers_tips, fingers_mcp):
    return all(f_tip.y > f_mcp.y for f_tip, f_mcp in zip(fingers_tips, fingers_mcp))

def detect_letter_rules(hand_landmarks): 
    """
    Rule-based detection for French Sign Language alphabet.
    Based on: https://github.com/Razane1414/Hand-Tracking---Langue-des-signes
    """
    if not hand_landmarks:
        return None

    pouce = hand_landmarks.landmark[4]   # tip du pouce
    index = hand_landmarks.landmark[8]   # tip de l'index
    majeur = hand_landmarks.landmark[12]  # tip du majeur
    annulaire = hand_landmarks.landmark[16]   # tip de l'annulaire
    auriculaire = hand_landmarks.landmark[20]  # tip de l'auriculaire

    fingers_tips = [index, majeur, annulaire, auriculaire]
    fingers_dip = [hand_landmarks.landmark[i] for i in [7, 11, 15, 19]] 
    fingers_pip = [hand_landmarks.landmark[i] for i in [6, 10, 14, 18]]  
    fingers_mcp  = [hand_landmarks.landmark[i] for i in [5, 9, 13, 17]]

    # Lettre A
    if doigts_repliés(fingers_tips, fingers_mcp) and \
       pouce.y < hand_landmarks.landmark[2].y and \
       distance_x(pouce, fingers_tips[0]) > 0.05:
        return "A"

    # Lettre B
    if doigts_tendus(fingers_tips, fingers_dip) and \
       distance_x(pouce, hand_landmarks.landmark[5]) < 0.05 and \
       pouce.y > fingers_mcp[0].y:
        return "B"

    # LETTRE "C" 
    distance_pouce_mcp_x = distance_x(pouce, fingers_mcp[0])
    distance_pouce_mcp_y = distance_y(pouce, fingers_mcp[0])
    tips_below_dip = all(f_tip.y > f_dip.y for f_tip, f_dip in zip(fingers_tips, fingers_dip))
    tips_away_from_mcp = all(distance_x(f_tip, f_mcp) > 0.02 for f_tip, f_mcp in zip(fingers_tips, fingers_mcp))
    distance_pouce_index_y = distance_y(pouce, index)

    if (distance_pouce_mcp_x > 0.05 and
        distance_pouce_mcp_y < 0.05 and
        tips_below_dip and 
        tips_away_from_mcp and
        0.05 < distance_pouce_index_y < 0.4):
            return "C"
    
    # Lettre "D"
    index_tendu = index.y < hand_landmarks.landmark[7].y  
    autres_doigts_repliés = all(
        f_tip.y > f_pip.y
        for f_tip, f_pip in zip(fingers_tips[1:], fingers_pip[1:])
    )
    pouce_proche_majeur = distance_x(pouce, majeur) < 0.02 and distance_y(pouce, majeur) < 0.04
    if  pouce_proche_majeur and index_tendu and autres_doigts_repliés:
        return "D"
            
    # Lettre "F"
    pouce_above_index = pouce.y < index.y
    pouce_index_close = distance_x(pouce, index) < 0.05
    autres_doigts_tendus = all(
        f_tip.y < f_mcp.y for f_tip, f_mcp in zip(fingers_tips[2:], fingers_mcp[2:]) 
    )
    index_plié = index.y > hand_landmarks.landmark[6].y  # tip sous PIP
    pouce_index_contact = distance_x(pouce, index) < 0.05 and index_plié

    if pouce_above_index and pouce_index_close and autres_doigts_tendus and pouce_index_contact:
        return "F"
    
    # Lettre "L"
    index_tendu = index.y < hand_landmarks.landmark[6].y
    pouce_tendu = pouce.x > hand_landmarks.landmark[2].x # Approximation
    autres_replies = doigts_repliés(fingers_tips[1:], fingers_mcp[1:])
    if index_tendu and autres_replies and distance_x(pouce, index) > 0.1:
        return "L"

    # Lettre "V"
    index_majeur_tendus = index.y < hand_landmarks.landmark[6].y and majeur.y < hand_landmarks.landmark[10].y
    autres_replies = doigts_repliés(fingers_tips[2:], fingers_mcp[2:])
    if index_majeur_tendus and autres_replies and distance_x(index, majeur) > 0.03:
        return "V"

    return None
