import numpy as np
import cv2
import time


# RESOLUTION = (240, 320)
RESOLUTION = (480, 640)
# RESOLUTION = (720, 1280)


def rho_theta_to_coords(line, image_shape=None):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    return x1, y1, x2, y2


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
        gray = filter_image(gray)  # display processed image
        # Display the fps
        cv2.putText(gray, "fps:{}".format(fps),
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0))
        cv2.imshow('frame shape:{}'.format(gray.shape), gray)
        # sudoku_loc = detect_sudoku(gray)
        if cv2.waitKey(1) & 0xFF == ord('q') or sudoku_loc is not None:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def detect_sudoku(img):
    f_img = filter_image(img)
    return None


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


if __name__ == "__main__":

    run()
