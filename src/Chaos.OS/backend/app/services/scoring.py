def get_severity(score: int):

    if score >= 90:
        return "💀 ABSOLUTE CHAOS"

    if score >= 75:
        return "🚨 EXTREME CHAOS"

    if score >= 50:
        return "⚠️ MODERATE CHAOS"

    if score >= 25:
        return "😬 MILD CHAOS"

    return "😌 BARELY CHAOS"


def clamp_score(score):

    return max(0, min(100, score))