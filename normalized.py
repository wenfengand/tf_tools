#encoding:utf-8
# Normalize an array to [0,1]
# Note: only array is supported now, dimension >= 2 is not tested yet.
def my_normalized(before_normalized):
    temp_max = max(before_normalized)
    temp_min = min(before_normalized)
    if temp_max != temp_min:
        normalized = (before_normalized - temp_min) / (temp_max - temp_min)
    else:
        normalized = before_normalized
    return normalized
