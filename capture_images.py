import os.path
import time

import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
img_counter = 0
frame_set = []
start_time = time.time()

what = "paper"
name = "albertgreinoecker"
howoften = 1

outdir = os.path.join("/home/albert/tmp/img", what)
os.makedirs(outdir)

while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", gray)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if time.time() - start_time >= howoften:  # Check if howoften sec passed
        img_name = "{}_{}_{}.png".format(what, img_counter, name)
        cv2.imwrite(os.path.join(outdir, img_name), frame)
        print("{} written!".format(img_counter))
        start_time = time.time()
        img_counter += 1
