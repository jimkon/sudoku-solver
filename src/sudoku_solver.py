import numpy as np
import cv2
import time


# RESOLUTION = (240, 320)
RESOLUTION = (480, 640)
RESOLUTION = (720, 1280)


def run():
    cap = cv2.VideoCapture(0)
    ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[1])
    ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FPS, 60)
    last_t = time.time()
    frames = 0
    fps = 0

    sudoku_loc = None

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
        l1, l2 = detection(filtered_image)
        filtered_image = binary_to_rgb(filtered_image)
        for line in l1:
            line = line[0]
            x1, y1, x2, y2 = rho_theta_to_coords(line)
            cv2.line(filtered_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for line in l2:
            line = line[0]
            x1, y1, x2, y2 = rho_theta_to_coords(line)
            cv2.line(filtered_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # Display the fps

        display_image = filtered_image
        cv2.putText(display_image, "fps:{}".format(fps), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        # display_image = np.hstack((gray, filtered_image))
        cv2.imshow('frame shape:{}'.format(gray.shape), display_image)
        # sudoku_loc = detect_sudoku(gray)
        if cv2.waitKey(1) & 0xFF == ord('q') or sudoku_loc is not None:
            break

    # When everything done, release the capture
    cap.release()
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
#     vs = list(np.sort(values))
    vs = values
    thresholds = list((vs[i+1]+vs[i])/2 for i in range(len(vs)-1))
#     thresholds.insert(0, np.min(arr)*(1+np.finfo(type(arr[0])).eps))
    thresholds.insert(0, np.min(arr)-0.01)

#     print(np.finfo(type(arr[0])).epsneg, np.min(arr),
#           np.min(arr)*(1+np.finfo(type(arr[0])).eps),
#           np.min(arr)<np.min(arr)*(1+np.finfo(type(arr[0])).eps))
    thresholds.append(np.max(arr))
#     print(thresholds)
    for th_i in range(1, len(thresholds)):
        #         print(thresholds[th_i-1], thresholds[th_i])
        indexes_1 = np.where(arr <= thresholds[th_i])[0]
        indexes_2 = np.where(arr > thresholds[th_i-1])[0]
        indexes = list(set(indexes_1).intersection(indexes_2))
        res.append(indexes)
    return res


def classify_lines_by_theta(ls):
    thetas = np.array(list(line[0][1] for line in ls))
#     print(len(thetas))
#     plt.hist(thetas)
    count, ths = np.histogram(thetas)
#     print(count)
#     print( np.argsort(count))
    th1, th2 = ths[((np.argsort(count))[::-1])[:2]]
#     print(th1, th2)

    thetas_1, thetas_2 = nearest_neighbors(thetas, [th1, th2])

    return ls[thetas_1], ls[thetas_2]


def filter_lines_by_rho(ls, threshold=25):
    rhos = np.array(list(line[0][0] for line in ls))

    sorted_indexes = np.argsort(rhos)
    rhos = rhos[sorted_indexes]
    sorted_lines = ls[sorted_indexes]
#     print(len(rhos), rhos)
    ct = np.array(list(np.array([abs(rhos[i]-rhos[j])
                                 for i in range(len(rhos)-1, j, -1)]) for j in range(len(rhos)-1)))
#     for i, _ in enumerate(ct):
#         print(i, _)
    res = []
    for i, l in enumerate(ct):
        if np.min(l) > threshold:
            res.append(i)
#         else:
#             print("delele", i)
    res.append(len(rhos)-1)
#     print("res", res)
#     print("rhos\n",rhos, "\n")
    classified_lines = nearest_neighbors(rhos, rhos[res])
#     print("classified_lines\n", classified_lines)
#     for _ in list(sorted_lines[inds] for inds in classified_lines):
#         print(_, "end")
    final_lines = list(np.average(sorted_lines[inds], axis=0) for inds in classified_lines)
#     print("test")
#     print(final_lines)
    return final_lines


def detection(img, n_first_lines=50):
    lines = cv2.HoughLines(img, 1, np.pi/180, 30)[:50]
    lines_1, lines_2 = classify_lines_by_theta(lines)
    lines_1, lines_2 = filter_lines_by_rho(lines_1), filter_lines_by_rho(lines_2)

    return lines_1, lines_2


def binary_to_rgb(binary):
    rgb_img = np.ones((binary.shape[0], binary.shape[1], 3))
    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            rgb_img[i][j] = [binary[i][j]]*3
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
