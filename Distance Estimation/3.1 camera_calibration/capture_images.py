import cv2 as cv
import os

Chess_Board_Dimensions = (9, 6)

n = 0  # image counter

# checks images dir is exist or not
image_path = "images"

Dir_Check = os.path.isdir(image_path)

if not Dir_Check:  # if directory does not exist, a new one is created
    os.makedirs(image_path)
    print(f'"{image_path}" Directory is created')
else:
    print(f'"{image_path}" Directory already exists.')

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv.findChessboardCorners(grayImage, boardDimension)
    if ret == True:
        corners1 = cv.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv.drawChessboardCorners(image, boardDimension, corners1, ret)

    return image, ret


cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    copyFrame = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    image, board_detected = detect_checker_board(
        frame, gray, criteria, Chess_Board_Dimensions
    )
    # print(ret)
    cv.putText(
        frame,
        f"saved_img : {n}",
        (30, 40),
        cv.FONT_HERSHEY_PLAIN,
        1.4,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    cv.imshow("frame", frame)
    # copyframe; without augmentation
    cv.imshow("copyFrame", copyFrame)

    key = cv.waitKey(1)

    if key == ord("q"):
        break
    if key == ord("s") and board_detected == True:
        # the checker board image gets stored
        cv.imwrite(f"{image_path}/image{n}.png", copyFrame)

        print(f"saved image number {n}")
        n += 1  # the image counter: incrementing
cap.release()
cv.destroyAllWindows()

print("Total saved Images:", n)
