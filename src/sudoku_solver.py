import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


# RESOLUTION = (240, 320)
# RESOLUTION = (480, 640)
RESOLUTION = (720, 1280)
DIGIT_RESOLUTION = (40, 40)

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

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            print("null frame")
            continue
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_image = gray_to_rgb(gray, mask=[0.5, 0.5, 0.5])

        frames += 1
        elapsed = time.time()-last_t
        if elapsed > 1:
            fps = frames
            frames = 0
            last_t = time.time()

        # Display the resulting frame
        mask, biggest_contour, bin_img, segmented_image = segment_image(gray, apply_segmentation=True,
                                                                        display={"img": display_image, "mask": [.5, .5, .5]})

        points = detect(bin_img, biggest_contour, display_image)
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
        cv2.putText(display_image, "fps:{}".format(fps), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.imshow('{}, frame shape:{}'.format(TITLE, gray.shape), display_image)

        c = cv2.waitKey(1) & 0xFF
        if c == ord('q'):
            break
        if c == ord('a') and points is not None:
            cap.release()
            cv2.destroyAllWindows()

            sudoku_image = crop_and_resize_image(gray, points,
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


def segment_image(img, apply_segmentation=False, display=None):

    blur = cv2.bilateralFilter(img, 9, 75, 75)

    th = 255-cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    size_th = 0.2*img.shape[0]*img.shape[1]

    curr, contours, hierarchy = cv2.findContours(
        np.copy(th), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sizes = np.array([cv2.contourArea(contour) for contour in contours])
    biggest_contour = contours[np.argmax(sizes)]

    mask = np.zeros(th.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [biggest_contour], -1, (1), -1)
    segmented_image = np.multiply(img, mask) if apply_segmentation else None
    segmented_bin_image = np.multiply(th, mask) if apply_segmentation else None
    if display is not None:
        display["img"] += gray_to_rgb(segmented_image, display["mask"])
        cv2.drawContours(display["img"], biggest_contour, -1, [255, 255, 255], 1)

    return mask, biggest_contour, segmented_bin_image, segmented_image


def detect(segmented_image, contour, display=None):

    lines = cv2.HoughLines(segmented_image, 1, np.pi/180, 30)[:50]

    rhos = np.array([line[0][0] for line in lines])
    thetas = np.array([line[0][1] for line in lines])*180/np.pi

    count, bins = np.histogram(thetas, bins=30, range=[0, 180])
    # print(count, bins)
    t = count
    t = np.hstack([t[len(t)-1], t])

    cm = np.array([np.sum(t[i-1:i+1]) for i in range(len(t))])[1:]
    cm = np.insert(cm, 0, cm[len(cm)-1], )
    print(cm)
    print(np.gradient(cm, 6))
    bins = bins[:len(bins)-1]+3
    bins = np.insert(bins, 0, -3)
    # print(bins)
    exit()
    pts = np.array([[bins[i], display.shape[0]-cm[i]] for i in range(len(cm))]).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    # print(pts)
    cv2.polylines(display, [pts], False, (0, 0, 255), thickness=2)

    c_max_ind = np.argmax(cm)
    # print([[bins[i] - bins[i+1]] for i in range(len(bins)-1)])
    # # degs *= 180/np.pi
    # ths = np.array([np.average([bins[i], bins[i+1]]) for i in range(len(bins)-1)])
    # # print(count, degs)
    # # mn, mx = degs[np.argsort(count)[:-2]]
    # arg_sort = np.argsort(count)[-2:]
    # th1, th2 = ths[arg_sort]
    # th1_min, th1_max = bins[arg_sort[0]], bins[arg_sort[0]+1]
    # th2_min, th2_max = bins[arg_sort[1]], bins[arg_sort[1]+1]
    # count = count[arg_sort]/len(lines)
    #
    # cv2.putText(display, "th1({}%) {}\n, min {} max {}".format(th1, count[0], th1_min, th1_max),
    #             (20, display.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    # cv2.putText(display, "th2({}%) {}\n, min {} max {}".format(th2, count[1], th2_min, th2_max),
    #             (20, display.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    # ev =

    # cv2.putText(display, "th1({}%) {}, th2({}%) {} diff({}%)={} ev={}".format(count[0], th1, count[1], th2,
    #                                                                           np.sum(count), abs(th1-th2), np.average(count)),
    #             (20, display.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.hist(thetas, bins=5, color="r")

    coord_lines = [rho_theta_to_coords(line[0]) for line in lines]
    for l in coord_lines:
        y1, x1, y2, x2 = l
        cv2.line(display, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # cv2.imshow("Sudoku", display)
    # plt.subplot(1, 2, 2)
    # plt.hist(thetas, bins=5, color="g")
    # plt.show()

    # cv2.imshow("Sudoku", binary_img)
    # c = cv2.waitKey(0)
    return None
    intersection_points = [line_intersect(l1[: 2], l1[2:], l2[: 2], l2[2:])
                           for l1 in coord_lines for l2 in coord_lines]
    intersection_points = np.around([p for p in intersection_points if p is not None])
    intersection_points = np.array(
        [p for p in intersection_points if cv2.pointPolygonTest(contour, (p[1], p[0]), False) >= 0])

    if len(intersection_points) < 4:
        return None

    min_x = intersection_points[np.argmin(intersection_points[:, 0])]
    max_x = intersection_points[np.argmax(intersection_points[:, 0])]
    min_y = intersection_points[np.argmin(intersection_points[:, 1])]
    max_y = intersection_points[np.argmax(intersection_points[:, 1])]

    final_points = np.around(np.array([min_x, max_x, min_y, max_y])).astype(np.int32)

    if display is not None:
        for i, p in enumerate(final_points):
            y1, x1 = p
            y2, x2 = final_points[i-1]
            cv2.circle(display, (x1, y1), 5, (255, 0, 0), -1)

            cv2.line(display, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return final_points


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

    res = np.array(list(img[p[1]: p[1]+DIGIT_RESOLUTION[1], p[0]                            : p[0]+DIGIT_RESOLUTION[0]] for p in points))
    res = np.reshape(res, (9, 9, DIGIT_RESOLUTION[0], DIGIT_RESOLUTION[1]))

    return res


def recognize_digits(digits):
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
