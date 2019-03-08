import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
