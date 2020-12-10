import cv2
import mesh
import sys

def main():
    mirror = False
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        img = mesh.main("img", sys.argv[1], sys.argv[2])
        cv2.imshow('web', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
