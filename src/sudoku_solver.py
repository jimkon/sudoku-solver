import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from debug_tools import *

# RESOLUTION = (240, 320)
# RESOLUTION = (480, 640)
RESOLUTION = (720, 1280)
DIGIT_RESOLUTION = np.array([40, 40])

TITLE = "Press q to quit and a when sudoku is detected"


def init_window(dev_id=0):
    cap = cv2.VideoCapture(dev_id)
    ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[1])
    ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FPS, 60)

    return cap


def run():
    cap = init_window(0)
    last_t = time.time()
    frames = 0
    fps = 0
    # points = None

    imgs = [1, 7, 8]
    count = 13
    path = "pics/test{}.jpg".format(13)
    pic = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (RESOLUTION[::-1]))

    while(True):
        # Capture frame-by-frame
        log_message("iter_start: {}".format(1e3*time.perf_counter()))
        ret, frame = cap.read()
        # frame = pic

        if frame is None:
            print("null frame")
            exit()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # display = gray

        frames += 1
        elapsed = time.time()-last_t
        if elapsed > 1:
            fps = frames
            # print(fps)
            frames = 0
            last_t = time.time()

        bin = to_binary(gray)

        bin = denoise(bin)

        display = gray_to_rgb(bin)

        mask, contour = segment_image(bin, error_th=.25)

        # cv2.drawContours(display, contour, -1, (255, 0, 0), 3)

        if mask is not None:
            # cv2.drawContours(display, [contour], -1, (255, 0, 0), 4)
            cp, dims, rot, = cv2.minAreaRect(contour)
            cv2.circle(display, tuple(np.array(list(cp)).astype(np.int)), 3, (0, 255, 0))

            segmented_bin = apply_mask(bin, mask)

            # plot_img(segmented_bin)

            display = gray_to_rgb(segmented_bin)

            detection = detect(segmented_bin, contour, display)
            if detection[0] is not None:
                points = detection[0]

                if points is not None:
                    for i, p in enumerate(points):
                        y1, x1 = p
                        y2, x2 = points[i-1]
                        cv2.circle(display, (x1, y1), 5, (255, 0, 0), -1)

                        cv2.line(display, (x1, y1), (x2, y2), (255, 0, 0), 1)

                sudoku = cv2.resize(project(segmented_bin, points, None), (
                    9*DIGIT_RESOLUTION[0], 9*DIGIT_RESOLUTION[1]))

                sudoku_bin = sudoku
                # plot_img(sudoku_bin)
                # sudoku_bin = prerecognition_bin(sudoku)

                digits, digit_boxes, digit_mask, digit_box = fetch_digits(sudoku_bin)

                if len(digit_boxes) == 0:
                    continue

                table_locations, digit_points, dims = process_box_data(digit_boxes)

                predictions, scores = recognize_digits(digits)
                # DEBUG
                # print(predictions)
                given_sudoku_table = most_possible_sudoku(predictions, scores, table_locations)
                # DEBUG
                temp_img = table_to_image(given_sudoku_table)
                for row in digit_points:
                    for p in row:
                        y, x = p
                        h, w = dims
                        cv2.rectangle(temp_img, (x, y), (x+w, y+h), (255, 0, 0))

                temp_img = cv2.resize(temp_img, (250, 250))

                # cv2.imshow("sud", temp_img)
                # cv2.waitKey(0)
                # DEBUG
                # print(predictions)
                # t = given_sudoku_table
                # given_sudoku_table = np.zeros((9, 9), np.int)
                # for i, p in enumerate(table_locations):
                #     y, x = p[0], p[1]
                #     score = scores[i]
                #     n = predictions[i]
                #     print("{}->{} in ({},{}) with score {}".format(i, n, y, x, score))
                #     given_sudoku_table[p[0], p[1]] = predictions[i]
                #
                # cv2.imshow("sud2", table_to_image(given_sudoku_table))
                # print(np.sum(np.abs(given_sudoku_table-t)))

                # DEBUG
                # cv2.imshow("bin", sudoku_bin)
                # cv2.waitKey(0)
                ###########################
                print(digits.shape)
                temp = np.vstack(digits)*255
                temp = np.hstack([temp, np.zeros(temp.shape).astype(np.uint8)])
                print(temp.shape)
                flatten_digits = np.array([d.flatten() for d in digits])
                print(flatten_digits.shape)

                clf = KMeans(n_clusters=9)
                clf.fit(flatten_digits)
                labels = clf.labels_
                print(labels.shape)
                print(labels)
                for i, l in enumerate(labels):
                    cv2.putText(temp, str(l), (28+10, i*28+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255))
                # for row in digit_points:
                #     for p in row:
                #         y, x = np.array(p)
                #         h, w = np.array(dims)
                #         temp[y:y+h, x:x+w, 2] =

                cv2.imshow("del", temp)
                cv2.imshow('del2', digit_mask)
                cv2.waitKey(0)
                ##################
                # DISABLE
                sudoku_grid = sudoku_bin-digit_mask

                digit_set = select_digit_set(digits, predictions, scores)

                solved_sudoku_table = solve(given_sudoku_table)

                sudoku_solution_table = solved_sudoku_table-given_sudoku_table

                display_without_sudoku = apply_mask(segmented_bin, (1-mask))

                given_sudoku = draw_sudoku(given_sudoku_table, digit_set,
                                           digit_points, dims, sudoku_bin.shape)
                given_sudoku = project(sudoku_grid+given_sudoku, None,
                                       points, display_without_sudoku.shape)

                given_sudoku = display_without_sudoku+given_sudoku

                sudoku_solution = draw_sudoku(sudoku_solution_table, digit_set,
                                              digit_points, dims, sudoku_bin.shape)
                sudoku_solution = project(sudoku_grid+sudoku_solution, None,
                                          points, display_without_sudoku.shape)

                sudoku_solution = display_without_sudoku+sudoku_solution

                full_sudoku = draw_sudoku(solved_sudoku_table, digit_set,
                                          digit_points, dims, sudoku_bin.shape)
                full_sudoku = cv2.bitwise_or(full_sudoku, digit_box)

                full_sudoku = project(sudoku_grid+full_sudoku, None,
                                      points, display_without_sudoku.shape)

                full_sudoku = display_without_sudoku+full_sudoku

                display = np.stack([full_sudoku, given_sudoku, sudoku_solution], axis=2)
                display[display.shape[0]-temp_img.shape[0]:, 0:temp_img.shape[1]] = temp_img

            # Display the fps
        cv2.putText(display, "fps:{}".format(fps), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
        cv2.imshow('{}, frame shape:'.format(TITLE), display)

        c = cv2.waitKey(1) & 0xFF
        if c == ord('q'):
            break
        # if c == ord('a') and points is not None:
        #     count += 1
        #     cap.release()
        #     cv2.destroyAllWindows()

    # When everything done, release the capture

    cap.release()
    cv2.destroyAllWindows()


@log_time
def to_binary(gray):
    gray = cv2.UMat(gray)

    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

    gray = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    gray = cv2.bitwise_not(gray)
    # gray = cv2.dilate(gray, np.ones((3, 3)))
    gray = cv2.UMat.get(gray)

    return gray


img_buffer = []


@log_time
def denoise(bin, b_size=2):
    global img_buffer
    img_buffer.insert(0, bin)
    if len(img_buffer) > b_size:
        img_buffer.pop()

    res = img_buffer[0]
    for i in range(1, len(img_buffer)):
        res = cv2.bitwise_and(res, img_buffer[i])

    return res


@log_time
def segment_image(bin, error_th=.1):

    curr, contours, hierarchy = cv2.findContours(
        np.copy(bin), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = np.array(contours)

    sizes = np.array([cv2.contourArea(contour) for contour in contours])
    sorted_args = np.argsort(sizes)[::-1]
    biggest_contours = contours[sorted_args[:10]]

    dims = np.array([cv2.minAreaRect(cnt)[1] for cnt in biggest_contours])
    aspect_ratios = np.array([np.min(dims)/np.max(dims) for dims in [cv2.minAreaRect(cnt)[1]
                                                                     for cnt in biggest_contours]])
    perim_area = np.array([(cv2.arcLength(cnt, True), cv2.contourArea(cnt))
                           for cnt in biggest_contours])
    perim_area_ratio = np.array([1-np.abs(1-(area/perimeter)/((area**.5)/4))
                                 for perimeter, area in perim_area if perimeter > 0])

    errors = 1-perim_area_ratio*aspect_ratios

    filtered_contours_indexes = np.array([i for i, cnt in enumerate(
        biggest_contours) if errors[i] < error_th])

    if len(filtered_contours_indexes) == 0:
        return None, None

    # final_contour_index = filtered_contours_ind[np.argmax(cont_stats[filtered_contours_ind, 1])]
    final_contour_index = np.min(filtered_contours_indexes)

    final_contour = biggest_contours[final_contour_index]

    mask = np.zeros(bin.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [final_contour], -1, (1), -1)

    return mask, final_contour


@log_time
def detect(segmented_image, contour, display=None):

    def cytrav(arr, index, window):
        return np.array([arr[i] for i in np.mod(index+np.arange(window)-int(window/2), len(arr))])

    lines = cv2.HoughLines(segmented_image, 1, np.pi/180, 30)

    if lines is not None:
        lines = lines[:min(50, len(lines))]
    else:
        return None, 0

    rhos = np.array([line[0][0] for line in lines])
    thetas = np.array([line[0][1] for line in lines])*180/np.pi

    th_range = 30
    thetas = np.sort(thetas)

    thetas = np.hstack([thetas[np.searchsorted(thetas, 180-th_range, side='right'):]-180, thetas])
    th_count = np.array([np.subtract(t[1], t[0]) for t in [np.searchsorted(thetas, [th-th_range, th],
                                                                           side='right') for th in range(180)]])
    f_th = np.copy(th_count)
    for i, th in enumerate(th_count):
        frame = cytrav(th_count, i, 50)
        if any(th < frame):
            f_th[i] = 0

    res = []
    s = -1
    for i in range(len(f_th)):
        if f_th[i] > f_th[i-1] and s < 0:
            s = i
        if f_th[i] < f_th[i-1] and s >= 0:
            ind = np.average([s, i]).astype(np.int)
            res.append(ind)
            s = -1
    res = np.array(res)
    if len(res) < 2:
        ev = 0
    else:
        res = res[np.argsort(th_count[res])[::-1]]
        res = res[:2]*np.pi/180

        ls = np.array([np.array([l[0] for l in lines if abs(l[0][1]-th) < np.pi/6]) for th in res])
        # for l in ls[0]:
        #     y1, x1, y2, x2 = rho_theta_to_coords(l)
        #     cv2.line(display, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # for l in ls[1]:
        #     y1, x1, y2, x2 = rho_theta_to_coords(l)
        #     cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
        line_stats = np.transpose([[np.min(ths),
                                    np.max(ths),
                                    np.average(ths),
                                    len(ths)]
                                   for ths in [[line[1] for line in l] for l in ls] if len(ths) > 0])

        # print(line_stats)
        min_th, max_th, avg_th, count = line_stats
        # print(min_th, max_th, avg_th, count)
        th_range_factor = np.abs(np.pi/6-np.abs(max_th-min_th))/(np.pi/6)
        cv2.putText(display, "th_range_factor = {}".format(th_range_factor),
                    (10, display.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        verticality_factor = 1-np.abs(np.abs(avg_th[0]-avg_th[1])/(np.pi/2)-1)
        cv2.putText(display, "verticality_factor = {}".format(verticality_factor),
                    (10, display.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        count_threshold = 4
        count_factor = (count-count_threshold)/(len(lines)-2*count_threshold)
        cv2.putText(display, "count_factor = {}".format(count_factor),
                    (10, display.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        ev = np.sum(np.multiply(th_range_factor, count_factor))*verticality_factor

    # cv2.putText(display, "{}%".format(ev*100),
    #             (10, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, int(ev*255), int((1-ev)*255)))
    # cv2.drawContours(display, [contour], -1, (0, int(ev*255), int((1-ev)*255)), 2)

    value_threshold = .75
    if ev > value_threshold:
        l_inds = np.array([[np.argmin(rhos), np.argmax(rhos)]
                           for rhos in [[line[0] for line in l] for l in ls]])
        l1_min, l1_max = [rho_theta_to_coords(l) for l in ls[0][l_inds[0]]]
        l2_min, l2_max = [rho_theta_to_coords(l) for l in ls[1][l_inds[1]]]

        final_points = np.around([line_intersect(l1_min[: 2], l1_min[2:], l2_min[: 2], l2_min[2:]),
                                  line_intersect(l1_min[: 2], l1_min[2:], l2_max[: 2], l2_max[2:]),
                                  line_intersect(l1_max[: 2], l1_max[2:], l2_min[: 2], l2_min[2:]),
                                  line_intersect(l1_max[: 2], l1_max[2:], l2_max[: 2], l2_max[2:])]).astype(np.int)

        final_points = order_points(final_points)
        # final_points = final_points[np.array([1, 0, 3, 2])]

        if any(np.ravel(final_points) == np.inf):
            return None, ev

        # for i, p in enumerate(final_points):
        #     y1, x1 = p
        #     y2, x2 = final_points[i-1]
        #     cv2.circle(display, (x1, y1), 5, (255, 0, 0), -1)
        #
        #     cv2.line(display, (x1, y1), (x2, y2), (255, 0, 0), 1)

        return final_points, ev
    return None, ev


@log_time
def project(src, src_pts, dest_pts, dest_shape=None):
    if dest_shape is None:
        dest_shape = src.shape

    if src_pts is None:
        src_pts = [[y, x] for y in [0, src.shape[0]] for x in [0, src.shape[1]]]

    if dest_pts is None:
        dest_pts = [[y, x] for y in [0, src.shape[0]] for x in [0, src.shape[1]]]

    src_pts = np.float32(order_points(src_pts))
    dest_pts = np.float32(order_points(dest_pts))

    src_pts = np.flip(src_pts, 1)
    dest_pts = np.flip(dest_pts, 1)

    M = cv2.getPerspectiveTransform(src_pts, dest_pts)
    res = cv2.warpPerspective(src, M, dest_shape[::-1])

    return res


@log_time
def prerecognition_bin(gray):
    gray = cv2.UMat(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 1.5)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    #
    # gray = cv2.Canny(gray, 0, 0)
    gray = cv2.bitwise_not(gray)

    gray = cv2.UMat.get(gray)

    return gray

def extract_sudoku(gray_image):
    pass

cv2.COLOR_

@log_time
def claw_mask(img_shape, points, claw_size):

    claw = np.array(
        [[1 if i == j or i == claw_size-1-j else 0 for i in range(claw_size)] for j in range(claw_size)]).astype(np.uint8)

    mask = np.zeros(img_shape, dtype=np.uint8)
    for p in points:
        mask[int(p[0]-claw_size/2):int(p[0]+claw_size / 2),
             int(p[1]-claw_size/2):int(p[1]+claw_size / 2)] += claw

    return mask


def grab_digits(img):
    mid_points = (2*DIGIT_RESOLUTION/5+np.array([np.array([x, y]) for x in np.arange(9)
                                                 for y in np.arange(9)])*(DIGIT_RESOLUTION)).astype(np.int)

    claws = claw_mask(img.shape, mid_points, 9)*255

    claws_on_sudoku = cv2.bitwise_or(img, claws)

    curr, all_contours, hierarchy = cv2.findContours(
        claws_on_sudoku, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    outer_contours = [cnt for i, cnt in enumerate(all_contours) if (hierarchy[0, i, 3] == -1)]

    grabed_contours = [cnt for cnt in outer_contours if any(
        [(cv2.pointPolygonTest(cnt, tuple(p), False) > -1) for p in mid_points])]

    # certain_size_contours = grabed_contours
    areas = [np.prod(cv2.boundingRect(cnt)[2:]) for cnt in grabed_contours]

    certain_size_contours = np.array([cnt for i, cnt in enumerate(
        grabed_contours) if areas[i] < 10000])

    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, certain_size_contours, -1, (255), -1)

    box_mask = np.zeros(img.shape, np.uint8)
    boxes = np.array([cv2.boundingRect(cnt) for cnt in certain_size_contours])

    grabed_items = np.array([img[y:y+h, x:x+w] for x, y, w, h in boxes])

    return grabed_items, certain_size_contours


def group_clustered_indexes(labels):
    unique_labels = set(labels)
    res = []
    for l in unique_labels:
        res.append(np.where(labels == l)[0])
    return np.array(res)

@log_time
def fetch_digits(aligned_bin_image, digit_shape=(28, 28)):
    sudoku_bin = np.copy(aligned_bin_image)
    cv2.rectangle(sudoku_bin, (0, 0), (sudoku_bin.shape[0]-1, sudoku_bin.shape[1]-1), (255), 1)

    dgts, cnt = grab_digits(sudoku_bin)

    def extract_features(img, cnt):
        x, y, h, w = cv2.boundingRect(cnt)
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

        img_area = w*h
        avg_brightness = np.sum(img)/img_area
        cnt_area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = float(cnt_area)/hull_area
        aspect_ratio = float(w)/h
        perimeter = cv2.arcLength(cnt, True)
        circ_factor = float(img_area)/perimeter

        res = np.array([w, h, avg_brightness, cnt_area, hull_area, solidity, circ_factor, angle])
        return res
    features = np.array([extract_features(dgts[i], cnt[i]) for i in range(len(dgts))])
    # features = np.array([cv2.resize(dgts[i], (10, 10)).flatten() for i in range(len(dgts))])
    clf = KMeans(n_clusters=2)
    clf.fit(features)
    labels = clf.labels_
    g_indexes = group_clustered_indexes(labels)
    for i, inds in enumerate(g_indexes):
        cv2.imshow("img", aligned_bin_image)
        group = np.array([cv2.resize(d, (100, 100)) for d in dgts[inds]])
        print(group.shape)
        img = np.hstack(group)
        cv2.imshow('g{}'.format(i), img)


        clf2 = KMeans(9)
        clf2.fit([g.flatten() for g in np.array([cv2.resize(d, (10, 10)) for d in dgts[inds]])])
        labels2 = clf2.labels_
        g_indexes2 = group_clustered_indexes(labels2)
        for i2, inds2 in enumerate(g_indexes2):
            group2 = np.array([d for d in group[inds2]])
            img2 = np.hstack(group2)
            cv2.imshow('g{}{}'.format(i, i2), img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.waitKey(0)
    # print(features)
    exit()
    print(g.shape, cnt.shape)

    mid_points = (2*DIGIT_RESOLUTION/5+np.array([np.array([x, y]) for x in np.arange(9)
                                                 for y in np.arange(9)])*(DIGIT_RESOLUTION)).astype(np.int)

    claws = claw_mask(sudoku_bin.shape, mid_points, 9)*255

    claws_on_sudoku = cv2.bitwise_or(sudoku_bin, claws)

    curr, all_contours, hierarchy = cv2.findContours(
        claws_on_sudoku, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    outer_contours = [cnt for i, cnt in enumerate(all_contours) if (hierarchy[0, i, 3] == -1)]

    grabed_contours = [cnt for cnt in outer_contours if any(
        [(cv2.pointPolygonTest(cnt, tuple(p), False) > -1) for p in mid_points])]

    areas = [np.prod(cv2.boundingRect(cnt)[2:]) for cnt in grabed_contours]

    certain_size_contours = [cnt for i, cnt in enumerate(
        grabed_contours) if areas[i] > 150 and areas[i] < 400]

    mask = np.zeros(sudoku_bin.shape, np.uint8)
    cv2.drawContours(mask, certain_size_contours, -1, (255), -1)

    box_mask = np.zeros(sudoku_bin.shape, np.uint8)
    [cv2.rectangle(box_mask, (x, y), (x+w, y+h), (255), -1)
        for x, y, w, h in [cv2.boundingRect(cnt) for cnt in certain_size_contours]]

    digits_only = cv2.bitwise_and(sudoku_bin, mask)

    curr, contours, hierarchy = cv2.findContours(
        digits_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_areas = [np.prod(cv2.boundingRect(cnt)[2:]) for cnt in contours]

    final_contours = [cnt for i, cnt in enumerate(
        contours) if final_areas[i] > 100 and final_areas[i] < 400]

    digits = []
    boxes = []

    for cnt in final_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        digit = cv2.threshold(cv2.resize(
            digits_only[y:y+h, x:x+w], digit_shape), 1, 1, cv2.THRESH_BINARY)[1]

        digits.append(digit)
        boxes.append([y, x, h, w])

    digits = np.array(digits)
    boxes = np.array(boxes)

    return digits, boxes, digits_only, box_mask


@log_time
def process_box_data(boxes):
    def calculate(i, x, y):
        if i in x:
            return y[x.index(i)]
        l = np.searchsorted(x, i)-1
        r = l+1
        x1, y1 = (x[l], y[l]) if l >= 0 else (x[0]-1, y[0])
        x2, y2 = (x[r], y[r]) if r < len(x) else (x[len(x)-1]+1, y[len(x)-1])
        yi = (y2-y1)/(x2-x1)*(i-x1)+y1

        return yi

    mid_locations = np.array([[int(y+h/2), (x+w/2)] for y, x, h, w in boxes])
    table_locations = np.array(mid_locations/DIGIT_RESOLUTION).astype(np.int)

    sorted_ys = [boxes[np.where(table_locations[:, 0] == n)[0], 0] for n in range(9)]

    avg_ys, inds = np.transpose(np.around([(np.average(y), i)
                                           for i, y in enumerate(sorted_ys) if len(y) > 0]).astype(np.int))
    ys = np.array([calculate(yi, list(inds), list(avg_ys)) for yi in range(9)], np.int)

    sorted_xs = [boxes[np.where(table_locations[:, 0] == n)[0], 0] for n in range(9)]
    avg_xs, inds = np.transpose(np.around([(np.average(x), i)
                                           for i, x in enumerate(sorted_xs) if len(x) > 0]).astype(np.int))
    xs = np.array([calculate(xi, list(inds), list(avg_xs)) for xi in range(9)], np.int)

    yx_points = [[(y, x) for x in xs] for y in ys]

    dims = np.array([[h, w] for y, x, h, w in boxes])

    avg_dims = np.around(np.average(dims, axis=0)).astype(np.int)

    return table_locations, yx_points, avg_dims


@log_time
def crop_digits(img):
    from itertools import product

    points = np.multiply(np.array(list(product(np.arange(9), np.arange(9)))), DIGIT_RESOLUTION)

    res = np.array([img[p[1]: p[1]+DIGIT_RESOLUTION[1], p[0]: p[0]+DIGIT_RESOLUTION[0]]
                    for p in points])
    res = np.reshape(res, (9, 9, DIGIT_RESOLUTION[0], DIGIT_RESOLUTION[1]))

    return res


@log_time
def load_templates():
    return [cv2.threshold(cv2.imread("templates/dgt_{}.png".format(i), cv2.IMREAD_GRAYSCALE), 1, 1, cv2.THRESH_BINARY)[1]
            for i in range(10)]


@log_time
def recognize_digits(digits):

    # from direc import Model
    # model = Model()
    templates = load_templates()

    def digit_matching(dgt, templates):

        scores = [1-np.sum(cv2.bitwise_xor(tmp, dgt)) / np.prod(tmp.shape)
                  for tmp in templates]
        res = np.argmax(scores)
        # score = scores[res]
        return [res, scores]

    pred = []
    scores = []
    for dgt in digits:
        # pred.append(model.predict(dgt))
        n, score = digit_matching(dgt, templates)
        pred.append(n)
        scores.append(score)

    pred = np.array(pred)
    scores = np.array(scores)

    return pred, scores


@log_time
def select_digit_set(digits, predictions, scores):
    best_scores = np.max(scores, axis=1)
    sorted_digits_index = np.array([np.where(predictions == n)[0] for n in range(1, 10)])

    if any([len(arr) == 0 for arr in sorted_digits_index]):
        return load_templates()[1:]

    max_digit_scores = np.array([inds[np.argmax(best_scores[inds])]
                                 for inds in sorted_digits_index])

    digit_set = digits[max_digit_scores]

    return digit_set


score_historic = np.zeros((9, 9, 10))
m = np.zeros((10))
m[0] = .1
d_h = np.multiply(np.ones((9, 9, 10)), m)


@log_time
def most_possible_sudoku(predictions, scores, locs):
    if predictions.shape[0] != scores.shape[0] or scores.shape[0] != locs.shape[0]:
        print("shape error")
        return

    global score_historic
    global d_h

    dh = np.copy(d_h)

    for i in range(len(predictions)):
        y, x = locs[i]
        score = scores[i]
        n = predictions[i]
        # print("{}-->{} in ({},{}) with score {}".format(i, n, y, x, score))
        # score_historic[y][x][n] = max(score, score_historic[y][x][n])
        # if x == 2 and y == 3:
        dh[y][x] = score
        # if x == 2 and y == 3:
        # print(score_historic[y][x])
        # print(np.argmax(score_historic[y][x]))
        # print(dh)
    score_historic += dh

    res = np.argmax(score_historic, axis=2).astype(np.int)

    return res


@log_time
def solve(unsolved_sudoku):
    # https://github.com/hbldh/dlxsudoku
    from dlxsudoku import Sudoku
    from dlxsudoku.exceptions import SudokuHasMultipleSolutionsError, SudokuHasNoSolutionError, SudokuTooDifficultError

    s = table_to_string(unsolved_sudoku)
    try:
        sudoku = Sudoku(s)
        sudoku.solve()
    except SudokuHasNoSolutionError as e:
        # print(e)
        return unsolved_sudoku
        pass
    except SudokuHasMultipleSolutionsError as e:
        # print(e)
        return unsolved_sudoku
        pass
    except SudokuTooDifficultError as e:
        # print(e)
        return unsolved_sudoku
    else:
        return string_to_table(sudoku.to_oneliner())


@log_time
def draw_sudoku(sudoku, digits, points, digit_dims, output_shape):
    output = np.zeros(output_shape, dtype=np.uint8)

    h, w = digit_dims

    digits_img = [cv2.resize(dgt, (w, h))*255 for dgt in digits]

    for r, row in enumerate(sudoku):
        for c, n in enumerate(row):
            if n == 0:
                continue
            digit = digits_img[n-1]

            y, x = points[r][c]
            output[y: y+h, x: x+w] = digit

    return output
####################################################################################################################################

####################################################################################################################################
####################################################################################################################################

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

####################################################################################################################################
####################################################################################################################################

####################################################################################################################################


def isPointInsideRectangle(rect, p):
    x, y, w, h = rect
    px, py = p

    return px >= x and px <= x+w and py >= y and py <= y+h


def order_points(pts):
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    rect = np.zeros((4, 2), dtype="float32")
    s = np.sum(pts, axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmax(diff)]
    rect[2] = pts[np.argmax(s)]
    rect[3] = pts[np.argmin(diff)]
    return rect


def apply_mask(img, mask):
    if len(img.shape) > 2:
        mask = np.stack([mask]*img.shape[2], axis=2)

    return np.multiply(img, mask)


def extend_image(img, new_shape, extention_value=0):
    assert img.shape[0] <= new_shape[0] and img.shape[1] <= new_shape[1], "Error on extension shapes"

    new_img = (np.ones(new_shape)*extention_value).astype(np.uint8)
    x, y = int((new_shape[1]-img.shape[1])/2), int((new_shape[0]-img.shape[0])/2)
    w, h = img.shape[1], img.shape[0]

    new_img[y: y+h, x: x+w] = img

    return new_img


def table_to_string(unsolved_sudoku):
    res = "".join(str(n) for n in np.array(unsolved_sudoku).flatten())
    return res


def string_to_table(s):
    res = np.array(list(np.array(list(int(d) for d in s[i: i+9])) for i in np.arange(9)*9))
    return res


def table_to_image(table, color_mask=[0, 0, 0.8]):
    temp = np.zeros((9, 9, DIGIT_RESOLUTION[0], DIGIT_RESOLUTION[1], 3))
    for i, row in enumerate(table):
        for j, n in enumerate(row):
            if n == 0:
                continue
            cv2.putText(temp[j][i], "{}".format(n),
                        (int(DIGIT_RESOLUTION[0]*0.25), int(DIGIT_RESOLUTION[0]*0.75)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0))
    res = None
    for row in temp:
        t_res = np.vstack(row)
        if res is None:
            res = t_res
        else:
            res = np.hstack([res, t_res])
    # res = gray_to_rgb(res, mask=color_mask)
    h, w = res.shape[: 2]
    for i in range(1, 9):
        y = i*DIGIT_RESOLUTION[0]
        x1, y1, x2, y2 = 0, y, w-1, y
        cv2.line(res, (x1, y1), (x2, y2), (0, 255, 0))

    for j in range(1, 9):
        x = j*DIGIT_RESOLUTION[1]
        x1, y1, x2, y2 = x, 0, x, h-1
        cv2.line(res, (x1, y1), (x2, y2), (0, 255, 0))

    cv2.rectangle(res, (0, 0), (w-1, h-1), (0, 255, 255))

    return res


def rho_theta_to_coords(line, image_shape=None):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 2000*(-b))
    y1 = int(y0 + 2000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))

    return [x1, y1, x2, y2]


def nearest_neighbors(arr, values):
    res = []

    vs = values
    thresholds = list((vs[i+1]+vs[i])/2 for i in range(len(vs)-1))

    thresholds.insert(0, np.min(arr)-0.01)

    thresholds.append(np.max(arr))

    for th_i in range(1, len(thresholds)):
        indexes_1 = np.where(arr <= thresholds[th_i])[0]
        indexes_2 = np.where(arr > thresholds[th_i-1])[0]
        indexes = list(set(indexes_1).intersection(indexes_2))
        res.append(indexes)
    return res


def classify_lines_by_theta(ls):
    thetas = np.array(list(line[0][1] for line in ls))

    count, ths = np.histogram(thetas)

    th1, th2 = ths[((np.argsort(count))[:: -1])[: 2]]
    print(th1, th2)

    thetas_1, thetas_2 = nearest_neighbors(thetas, [th1, th2])

    return ls[thetas_1], ls[thetas_2]


def filter_lines_by_rho(ls, threshold=25):
    rhos = np.array(list(line[0][0] for line in ls))

    sorted_indexes = np.argsort(rhos)
    rhos = rhos[sorted_indexes]
    sorted_lines = ls[sorted_indexes]

    ct = np.array(list(np.array([abs(rhos[i]-rhos[j])
                                 for i in range(len(rhos)-1, j, -1)]) for j in range(len(rhos)-1)))

    res = []
    for i, l in enumerate(ct):
        if np.min(l) > threshold:
            res.append(i)

    res.append(len(rhos)-1)

    classified_lines = nearest_neighbors(rhos, rhos[res])

    final_lines = list(np.average(sorted_lines[inds], axis=0) for inds in classified_lines)

    return final_lines


def gray_to_rgb(binary, mask=[1, 1, 1]):
    apply_mask = np.multiply(np.reshape(np.array(mask), (3, 1, 1)), [binary]*3)
    rgb_img = np.stack(apply_mask, axis=2).astype(np.uint8)
    return rgb_img


# def rho_theta_to_coords(line, image_shape=None):
#     rho, theta = line
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 2000*(-b))
#     y1 = int(y0 + 2000*(a))
#     x2 = int(x0 - 2000*(-b))
#     y2 = int(y0 - 2000*(a))
#     return x1, y1, x2, y2


def line_intersect(A1, A2, B1, B2):
    # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
    def slope(P1, P2):
        if P2[1] - P1[1] == 0:
            if P2[0]-P1[0] >= 0:
                return np.finfo(np.float).max
            else:
                return np.finfo(np.float).min
        return(P2[0] - P1[0]) / (P2[1] - P1[1])

    def y_intercept(P1, slope):
        return P1[0] - slope * P1[1]

    m1, m2 = slope(A1, A2), slope(B1, B2)
    if m1 == m2:
        return None

    b1, b2, = y_intercept(A1, m1), y_intercept(B1, m2)

    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return [y, x]


def rho_theta_to_coords(line, image_shape=None):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 2000*(-b))
    y1 = int(y0 + 2000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))
    return y1, x1, y2, x2


if __name__ == "__main__":
    # from dlxsudoku import Sudoku
    # sud = "030467050920010006067300148301006027400850600090200400005624001203000504040030702"
    # s = string_to_table(
    #     sud)
    # print(s)
    # sudoku = Sudoku(sud)
    # sudoku.solve()
    # print(string_to_table(sudoku.to_oneliner()))
    run()
