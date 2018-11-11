import numpy as np
import cv2
import time


# RESOLUTION = (240, 320)
# RESOLUTION = (480, 640)
RESOLUTION = (720, 1280)
DIGIT_RESOLUTION = (40, 40)


def run():
    cap = cv2.VideoCapture(0)
    ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[1])
    ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FPS, 60)
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
        frames += 1
        elapsed = time.time()-last_t
        if elapsed > 1:
            print(frames)
            fps = frames
            frames = 0
            last_t = time.time()

        # Display the resulting frame
        filtered_image = filter_image(gray)  # display processed image

        # display_image = gray_to_rgb(filtered_image)
        display_image = gray_to_rgb(gray)

        points = detect(filtered_image)
        for i, p in enumerate(points):
            c_p = [(255, 0, 0),
                   (255, 0, 255),
                   (0, 0, 255),
                   (0, 0, 255)]
            c = c_p[i]
            cv2.circle(display_image, tuple(p),  3, c, -1)
            prev_p = points[i-1 if i > 0 else len(c_p)-1]
            cv2.line(display_image, tuple(p), tuple(prev_p), (255, 0, 0), 1)

        # Display the fps
        cv2.putText(display_image, "fps:{}".format(fps), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        # display_image = np.hstack((gray, filtered_image))
        cv2.imshow('frame shape:{}'.format(gray.shape), display_image)

        c = cv2.waitKey(1) & 0xFF
        if c == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
        if c == ord('a') and points is not None:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    print(points)

    sudoku_image = crop_image(gray, points)

    sudoku_image = cv2.resize(crop_image(gray, points), (9*DIGIT_RESOLUTION[0], 9*DIGIT_RESOLUTION[1]),
                              interpolation=cv2.INTER_CUBIC)  # INTER_AREA  for shrinking
    # print("cropped and resized", sudoku_image.shape)
    # cv2.imshow("cropped and resized", gray_to_rgb(sudoku_image))
    # c = cv2.waitKey(0)

    digits = crop_digits(sudoku_image)
    print("digits", digits.shape)
    table = recognize_digits(digits)
    print("sudoku\n", table)
    table_image = table_to_image(table)
    print("table_image", table_image.shape)
    temp = gray_to_rgb(sudoku_image, mask=[0.2, 0.2, 0.2]) + table_image
    cv2.imshow("digits", temp)
    # temp = (temp for digit in digits)
    # cv2.imshow("cropped and resized", sudoku_image)
    c = cv2.waitKey(0)

    cv2.destroyAllWindows()


def filter_image(img, check_size=False):
    if check_size:
        DESIRABLE_AREA = RESOLUTION[0]*RESOLUTION[1]
        current_area = img.shape[0]*img.shape[1]
        ratio = DESIRABLE_AREA/current_area

        img = cv2.resize(img, (0, 0), fx=ratio**0.5, fy=ratio**0.5)

    blur = cv2.bilateralFilter(img, 9, 75, 75)
    th = 255-cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    blur2 = cv2.medianBlur(th, 7)

    # # kernel = np.ones((5, 5), np.uint8)
    # # opening = cv2.morphologyEx(blur2, cv2.MORPH_CLOSE, kernel)
    # #
    # # kernel = np.ones((7, 7), np.uint8)
    # # dilation = cv2.dilate(opening, kernel, iterations=1)
    #
    # res = dilation
    res = blur2

    return res


def detect(filtered_img):

    length = int(0.8*RESOLUTION[0])

    x = int((RESOLUTION[1]-length)/2)
    y = int((RESOLUTION[0]-length)/2)
    w = length
    h = length

    return [x, y], [x+w, y], [x+w, y+h], [x, y+h]


def crop_image(img, points):
    x1 = points[0][0]
    x2 = points[1][0]
    y1 = points[0][1]
    y2 = points[2][1]
    return img[y1:y2, x1:x2]


def crop_digits(img):
    from itertools import product

    points = np.multiply(np.array(list(product(np.arange(9), np.arange(9)))), DIGIT_RESOLUTION)
    # print(points)
    # for p1 in points:
    #     p2 = p1+DIGIT_RESOLUTION
    #     cv2.circle(img, tuple(p1),  3, (0, 255, 0), -1)
    #     cv2.circle(img, tuple(p2),  3, (0, 255, 255), -1)

    res = np.array(list(img[p[1]:p[1]+DIGIT_RESOLUTION[1], p[0]                            :p[0]+DIGIT_RESOLUTION[0]] for p in points))
    res = np.reshape(res, (9, 9, DIGIT_RESOLUTION[0], DIGIT_RESOLUTION[1]))
    # for i, _ in enumerate(res):
    #     cv2.putText(_, "{}".format(i), (20, 20),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return res


def recognize_digits(digits):
    return np.random.randint(10, size=(9, 9))


############################################

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


def line_intersect(A1, A2, B1, B2):
    # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
    def slope(P1, P2):
        return(P2[1] - P1[1]) / (P2[0] - P1[0])

    def y_intercept(P1, slope):
        return P1[1] - slope * P1[0]

    m1, m2 = slope(A1, A2), slope(B1, B2)
    if m1 == m2:
        print("These lines are parallel!!!")
        return None

    b1, b2, = y_intercept(A1, m1), y_intercept(B1, m2)

    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y


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

    return x1, y1, x2, y2


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

    th1, th2 = ths[((np.argsort(count))[::-1])[:2]]

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
    # apply_mask = np.array([binary*mask[0], binary*mask[1], binary*mask[2]])
    apply_mask = np.multiply(np.reshape(np.array(mask), (3, 1, 1)), [binary]*3)
    rgb_img = np.stack(apply_mask, axis=2).astype(np.uint8)
    return rgb_img


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
    return x1, y1, x2, y2


if __name__ == "__main__":

    run()
