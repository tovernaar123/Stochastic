import time


def a():
    time.sleep(200)
    from doesnotexsit import a
a()