def priority(item, bins_remain_cap):
    priority = []
    for cap in bins_remain_cap:
        diff = abs(item - cap)
        priority.append(-diff)
    return priority
