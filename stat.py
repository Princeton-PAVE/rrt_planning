import pstats

with open('mainstats.txt', 'w+') as f:
    p = pstats.Stats('stats.pstats', stream=f)
    p.strip_dirs().sort_stats('cumulative').print_stats()