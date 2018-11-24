import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import pytesseract

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
    points = None

    count = 0
    imgs = [1, 7, 8]
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # path = "pics/test{}.jpg".format(imgs[count])
        # frame = cv2.imread(path, cv2.IMREAD_COLOR)

        if frame is None:
            print("null frame")
            exit()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # display = gray
        display = gray_to_rgb(gray, mask=[.5, .5, .5])

        frames += 1
        elapsed = time.time()-last_t
        if elapsed > 1:
            fps = frames
            # print(fps)
            frames = 0
            last_t = time.time()

        bin = to_binary(gray)
        # display = bin
        # display[:, :, 0] = bin
        # display[:, :, 0] += (bin/2).astype(np.uint8)

        mask, contour = segment_image(bin)

        if mask is not None:
            cv2.drawContours(display, [contour], -1, (255, 0, 0), 4)
            cp, dims, rot, = cv2.minAreaRect(contour)
            cv2.circle(display, tuple(np.array(list(cp)).astype(np.int)), 3, (0, 255, 0))

        #
        # segmented_bin = apply_mask(bin, mask)
        #
        # detection = detect(bin, biggest_contour, display)
        # if detection[0] is not None:
        #     points = detection[0]

        # if points is not None:
        # display_image = gray_to_rgb(mask, mask=[0, 0, .1])
        #
        # segment_color = [0, 0, 1]
        # if points is not None:
        #     segment_color = [0, 1, 0]
        #
        # display_image += gray_to_rgb(mask, mask=[_*.2 for _ in segment_color])
        #
        # cv2.drawContours(display_image, biggest_contour, -1, [_*255 for _ in segment_color], 1)

        # for i, p in enumerate(points):
        #     y1, x1 = p
        #     y2, x2 = points[i-1]
        #     cv2.circle(display_image, (x1, y1), 5, (255, 0, 0), -1)
        #
        #     cv2.line(display_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Display the fps
        cv2.putText(display, "fps:{}".format(fps), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.imshow('{}, frame shape:'.format(TITLE), display)

        c = cv2.waitKey(1) & 0xFF
        if c == ord('q'):
            break
        if c == ord('a') and points is not None:
            count += 1
            cap.release()
            cv2.destroyAllWindows()
            sudoku_image = crop_and_resize_image(bin_img, points,
                                                 new_shape=(9*DIGIT_RESOLUTION[0], 9*DIGIT_RESOLUTION[1]))

            digits = crop_digits(sudoku_image)

            unsolved_sudoku = recognize_digits(digits)

            display_image = gray_to_rgb(sudoku_image, mask=[
                0.2, 0.2, 0.2]) + table_to_image(unsolved_sudoku)
            display_image = extend_image(display_image, (440, 600, 3))
            cv2.putText(display_image, "Press q to rerun detection, or any key to continue with solution",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.imshow("Sudoku", display_image)

            c = cv2.waitKey(0) & 0xFF
            if c == ord('q'):
                cv2.destroyAllWindows()
                cap = init_window(0)
                continue

            display_image = gray_to_rgb(sudoku_image, mask=[
                0.2, 0.2, 0.2]) + table_to_image(unsolved_sudoku)
            display_image = extend_image(display_image, (440, 600, 3))
            cv2.putText(display_image, "Solving...",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.imshow("Sudoku", display_image)

            solved_sudoku = solve(unsolved_sudoku)

            solution = solved_sudoku-unsolved_sudoku

            display_image = gray_to_rgb(sudoku_image, mask=[0.2, 0.2, 0.2]) + \
                table_to_image(solution, color_mask=[0.8, 0, 0])+table_to_image(unsolved_sudoku)
            display_image = extend_image(display_image, (440, 600, 3))
            cv2.putText(display_image, "Press any key to continue...",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.imshow("Sudoku", display_image)
            c = cv2.waitKey(0)
            cv2.destroyAllWindows()
            cap = init_window(0)

    # When everything done, release the capture

    cap.release()
    cv2.destroyAllWindows()


def to_binary(gray):
    gray = cv2.UMat(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    # gray = cv2.Canny(gray, 0, 50)

    # gray = cv2.dilate(gray, np.ones((5, 5)))

    # blur = cv2.bilateralFilter(gray, 9, 75, 75)
    #
    gray = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    #
    # gray = cv2.medianBlur(gray, 9)
    #
    # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((9, 9)))
    gray = cv2.bitwise_not(gray)
    gray = cv2.UMat.get(gray)

    return gray


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


def detect(segmented_image, contour, display=None):

    def cytrav(arr, index, window):
        return np.array([arr[i] for i in np.mod(index+np.arange(window)-int(window/2), len(arr))])

    lines = cv2.HoughLines(segmented_image, 1, np.pi/180, 30)[:50]

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
        #     y1, x1, y2, x2 = rho_theta_to_coords(l[0])
        #     cv2.line(display, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # for l in ls[1]:
        #     y1, x1, y2, x2 = rho_theta_to_coords(l[0])
        #     cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
        line_stats = np.transpose([[np.min(ths),
                                    np.max(ths),
                                    np.average(ths),
                                    len(ths)]
                                   for ths in [[line[1] for line in l] for l in ls] if len(ths) >= 0])

        # print(line_stats)
        min_th, max_th, avg_th, count = line_stats
        # print(min_th, max_th, avg_th, count)
        th_range_factor = 1-(max_th-min_th)/(np.pi/6)
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

    cv2.putText(display, "{}%".format(ev*100),
                (10, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, int(ev*255), int((1-ev)*255)))
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

        def order_points(pts):
            # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect
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


def crop_and_resize_image(img, points, new_shape=None):
    if new_shape is None:
        new_shape = img.shape

    def order_points(pts):
        # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    tl, tr, br, bl = order_points(points)
    w = new_shape[1]
    h = new_shape[0]
    a1, a2 = tl[:: -1], [0, 0]
    b1, b2 = tr[:: -1], [0, w]
    c1, c2 = br[:: -1], [h, w]
    d1, d2 = bl[:: -1], [h, 0]
    pts1 = np.float32([a1, b1, c1, d1])
    pts2 = np.float32([a2, b2, c2, d2])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    res = cv2.warpPerspective(img, M, (w, h))

    return res


def crop_digits(img):
    from itertools import product

    points = np.multiply(np.array(list(product(np.arange(9), np.arange(9)))), DIGIT_RESOLUTION)

    res = np.array([img[p[1]: p[1]+DIGIT_RESOLUTION[1], p[0]: p[0]+DIGIT_RESOLUTION[0]]
                    for p in points])
    res = np.reshape(res, (9, 9, DIGIT_RESOLUTION[0], DIGIT_RESOLUTION[1]))

    return res


def recognize_digits(digits):

    from direc import Model
    model = Model()
    for row in digits:
        for dgt in row:
            img = dgt[6:34, 6:34]
            print(model.predict(img))
            cv2.imshow("d1", img)
            cv2.waitKey(0)

    exit()

    # def filter_area(contour, img):
    #     # contour area
    #     area = cv2.contourArea(contour)
    #     if area < 40*40*0.02 or area > 40*40*0.25:
    #         return False
    #     # contour convex hull area
    #     x, y, w, h = cv2.boundingRect(contour)
    #     area = w*h
    #     if h < w or area < 40*40*0.02 or area > 40*40*0.5:
    #         return False
    #     contour_sum = img[y:y+h, x:x+w]
    #     if np.sum(contour_sum)/area < 0.2:
    #         return False
    #     return True
    #
    # img = np.hstack(np.vstack(row) for row in digits)
    # im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = [cnt for cnt in contours if filter_area(cnt, img/255)]
    #
    # img = gray_to_rgb(img)
    # for c in contours:
    #     cv2.drawContours(img, [c], 0, (0, 0, 255), 1)
    #     x, y, w, h = cv2.boundingRect(c)
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
    # cv2.imshow("d1", img)
    # cv2.waitKey(0)
    # example sudoku
    return string_to_table("030467050920010006067300148301006027400850600090200400005624001203000504040030702")
    # return np.random.randint(10, size=(9, 9))


def solve(unsolved_sudoku):
    # https://github.com/hbldh/dlxsudoku
    from dlxsudoku import Sudoku

    s = table_to_string(unsolved_sudoku)
    sudoku = Sudoku(s)
    sudoku.solve()
    return string_to_table(sudoku.to_oneliner())
############################################


def apply_mask(img, mask):
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
    temp = np.zeros((9, 9, DIGIT_RESOLUTION[0], DIGIT_RESOLUTION[1]))
    for i, row in enumerate(table):
        for j, n in enumerate(row):
            if n == 0:
                continue
            cv2.putText(temp[j][i], "{}".format(n),
                        (int(DIGIT_RESOLUTION[0]*0.4), int(DIGIT_RESOLUTION[0]*0.4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0))
    res = None
    for row in temp:
        t_res = np.vstack(row)
        if res is None:
            res = t_res
        else:
            res = np.hstack([res, t_res])

    return gray_to_rgb(res, mask=color_mask)


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

    run()
