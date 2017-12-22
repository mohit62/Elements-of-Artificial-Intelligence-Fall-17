def classify_blue(img):
    sum_v = 0.0
    sum_h = 0.0
    score = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    h = {2: 1, 5: 1, 140: -1, 143: -1, 146: 1, 20: -1, 149: 1, 23: -1, 26: 1, 29: 1, 164: -1, 167: -1, 170: 1, 44: -1,
         173: 1, 47: -1, 50: 1, 53: 1, 188: -1, 191: -1, 68: -1, 71: -1, 74: 1, 77: 1, 92: -1, 95: -1, 98: 1, 101: 1,
         116: -1, 119: -1, 122: 1, 125: 1}

    v = {2: 1, 5: 1, 8: 1, 11: 1, 14: 1, 17: 1, 146: -1, 20: 1, 149: -1, 23: 1, 152: -1, 26: 1, 155: -1, 29: 1, 158: -1,
         32: 1, 161: -1, 35: 1, 164: -1, 38: 1, 167: -1, 41: 1, 170: -1, 44: 1, 173: -1, 47: 1, 176: -1, 179: -1,
         182: -1, 185: -1, 188: -1, 191: -1}

    for blue_pixel, sign in v.iteritems():
        if sign > 0:
            score[0] += img[blue_pixel]
        #score[180]-=img[blue_pixel]
        else:
            score[180] += img[blue_pixel]
#score[0] -= img[blue_pixel]
        sum_v += img[blue_pixel]

    for blue_pixel, sign in h.iteritems():
        if sign > 0:
            score[270] += img[blue_pixel]
        #score[90] -= img[blue_pixel]

        else:
            score[90] += img[blue_pixel]
        #score[270] -= img[blue_pixel]

        sum_h += img[blue_pixel]

    score[0] = score[0] / sum_v
    score[180] = score[180]/ sum_v
    score[270] = score[270] / sum_h
    score[90] = score[90] / sum_h

    return max(score, key=score.get)
