import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cProfile, pstats, io
import solver


def profile(fnc):

    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):

        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

def group_clustered_items(labels, items):
    items = np.array(items)
    unique_labels = set(labels)
    res = []
    for l in unique_labels:
        res.append(items[np.where(labels == l)[0]])
    return res


def claw_mask(img_shape, claw_size):
    half_step = np.array(img_shape)/18
    points = np.array([np.array([i, j]) for i in np.linspace(half_step[0], img_shape[0]-half_step[0], 9)
                      for j in np.linspace(half_step[1], img_shape[1]-half_step[1], 9)])
    claw = np.array(
        [[1 if i == j or i == claw_size-1-j else 0 for i in range(claw_size)] for j in range(claw_size)]).astype(np.uint8)

    mask = np.zeros(img_shape, dtype=np.uint8)
    for p in points:
        mask[int(p[0]-claw_size/2):int(p[0]+claw_size / 2),
             int(p[1]-claw_size/2):int(p[1]+claw_size / 2)] += claw

    return mask*255, points

def grab_digits(img):
    claws, mid_points = claw_mask(img.shape, 9)

    claws_on_sudoku = cv2.bitwise_or(img, claws)

    curr, all_contours, hierarchy = cv2.findContours(
        claws_on_sudoku, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    outer_contours = [cnt for i, cnt in enumerate(all_contours) if (hierarchy[0, i, 3] == -1)]

    grabed_contours = [cnt for cnt in outer_contours if any(
        [(cv2.pointPolygonTest(cnt, tuple(p), False) >= 0) for p in mid_points])]

    areas = [np.prod(cv2.boundingRect(cnt)[2:]) for cnt in grabed_contours]

    clf = KMeans(n_clusters=3)
    clf.fit(np.reshape(areas, (-1, 1)))
    centers = clf.cluster_centers_

    desired_size_label = list(range(3))
    desired_size_label.remove(np.argmin(clf.cluster_centers_))
    desired_size_label.remove(np.argmax(clf.cluster_centers_))

    groups = group_clustered_items(clf.labels_, grabed_contours)
    desired_size_contours = groups[desired_size_label[0]]


    g0 = np.zeros(img.shape, np.uint8)
    cv2.drawContours(g0, groups[0], -1, (255), -1)

    g1 = np.zeros(img.shape, np.uint8)
    cv2.drawContours(g1, groups[1], -1, (255), -1)

    g2 = np.zeros(img.shape, np.uint8)
    cv2.drawContours(g2, groups[2], -1, (255), -1)

    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, desired_size_contours, -1, (255), -1)

    box_mask = np.zeros(img.shape, np.uint8)
    boxes = np.array([cv2.boundingRect(cnt) for cnt in desired_size_contours])
    [cv2.rectangle(box_mask, (x, y), (x+w, y+h), (255), -1)
        for x, y, w, h in boxes]

    digits_only = cv2.bitwise_and(img, box_mask)

    grabed_items = np.array([img[y:y+h, x:x+w] for x, y, w, h in boxes])

    return grabed_items, desired_size_contours, boxes

def normalize_digits(dgts, res_shape=(40, 40)):
    nomalized = [np.array(np.divide(dgt, 255), dtype=np.uint8) for dgt in dgts]
    resized = [cv2.resize(dgt, res_shape) for dgt in nomalized]
    return resized

def constuct_sudoku_table(nums, locs):
    sud = np.zeros((9, 9), np.int)
    for num, loc in zip(nums, locs):
        j, i = loc
        sud[i, j] = num+1

    return sud

def cluster_sudoku_digits(dgts, boxes, img_shape):
    clf = KMeans(n_clusters=9)
    n_dgts = normalize_digits(dgts)
    clf.fit([d.flatten() for d in n_dgts])

    groups = group_clustered_items(clf.labels_, dgts)

    g_sizes = [len(g) for g in groups]

    d_shape = np.array(img_shape)/9
    locs = np.array([np.array([x, y]) for x, y, w, h in boxes])
    locs = (locs/d_shape).astype(np.int)

    sud = constuct_sudoku_table(clf.labels_, locs)

    return sud, groups

def create_sudoku_image(sud, digit_groups, shape=(400, 400)):
    def draw_digit(img, loc, digit):
        x, y = np.array(loc)*40+20
        h, w = digit.shape
        img[y:y+h, x:x+w] = digit

    result = np.zeros(shape, np.uint8)
    for i, row in enumerate(sud):
        for j, n in enumerate(row):
            if n==0:
                continue

            dgt = digit_groups[n-1][np.random.randint(0, len(digit_groups[n-1]))]
            draw_digit(result, (j, i), dgt)

    return result

def solve_sudoku(sudoku):
    #https://towardsdatascience.com/peter-norvigs-sudoku-solver-25779bb349ce
    import solver
    sudoku_str = table_to_string(sudoku)
    res = solver.solve(sudoku_str)
    if res==False:
        return sudoku
    res = solver.sudoku_to_string(res)
    sud_table = string_to_table(res)

    return sud_table

def table_to_string(unsolved_sudoku):
    res = "".join(str(n) for n in np.array(unsolved_sudoku).flatten())
    return res


def string_to_table(s):
    res = np.array(list(np.array(list(int(d) for d in s[i: i+9])) for i in np.arange(9)*9))
    return res

def extract_sudoku(img):
    dgts, cnts, locs = grab_digits(img)
    sudoku, groups = cluster_sudoku_digits(dgts, locs, img.shape)
    solved = solve_sudoku(sudoku)
    res = create_sudoku_image(solved, groups)
    return res
