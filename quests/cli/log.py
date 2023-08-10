def logger(msg):
    print(f"[QUESTS]: {msg}")


def format_time(seconds):
    if seconds < 1:
        return f"{seconds * 1000: .2f} ms"

    return f"{seconds: .1f} s"
