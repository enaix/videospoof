import cv2
import dlib
import numpy
import sys
import os
import operator

class Config:
    def __init__(self):
        self.PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)

        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.JAW_POINTS = list(range(0, 17))
        self.TOP_BROW = [*self.LEFT_BROW_POINTS[::-1], *self.RIGHT_BROW_POINTS[::-1]]

        # Points used to line up the images.
        self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                               self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)
        self.OVERLAY_POINTS = [self.JAW_POINTS + self.TOP_BROW]
        self.REMOVE_POINTS = [self.MOUTH_POINTS, self.LEFT_EYE_POINTS, self.RIGHT_EYE_POINTS]
        self.FEATHER_AMOUNT = 11
        self.COLOUR_CORRECT_BLUR_FRAC = 0.6
        self.FACE_LINES_CLEARANCE = 2

        self.FACE_POLYGONS = [[0, 36, 1], [1, 36, 41], [1, 41, 40], [1, 40, 28], [1, 28, 2], [2, 28, 29], [2, 29, 3], [3, 29, 31],
                    [3, 31, 4], [4, 31, 48], [4, 48, 5], [5, 48, 60], [5, 60, 59], [5, 59, 6], [6, 59, 58], [6, 58, 7],
                    [7, 58, 57], [7, 57, 8], [8, 57, 9], [9, 57, 56], [9, 56, 10], [10, 56, 55], [10, 55, 11], [11, 55, 64],
                    [11, 64, 54], [11, 54, 12], [54, 12, 35], [12, 35, 13], [13, 35, 29], [13, 29, 14], [29, 14, 28],
                    [14, 28, 15], [28, 15, 47], [15, 47, 46], [46, 15, 45], [15, 45, 16], [16, 45, 26], [45, 26, 44], [26, 44, 25],
                    [25, 44, 24], [24, 44, 23], [44, 23, 43], [23, 43, 22], [43, 22, 42], [22, 42, 27], [27, 42, 28],
                    [28, 42, 47], [40, 28, 39], [28, 39, 27], [21, 39, 27], [21, 22, 27], [38, 21, 39], [20, 38, 21],
                    [37, 20, 38], [19, 37, 20], [18, 37, 19], [17, 37, 18], [36, 17, 37], [0, 17, 36], [48, 31, 49],
                    [49, 31, 50], [31, 50, 32], [32, 50, 33], [50, 33, 51], [51, 33, 52], [33, 52, 34], [34, 52, 35],
                    [52, 35, 53], [53, 35, 54], [31, 29, 30], [35, 29, 30], [31, 30, 32], [32, 30, 33], [33, 30, 34],
                    [34, 30, 35]
                ]
        print(len(self.FACE_POLYGONS), "face mesh polygons total")

config = Config()

class PointFinder:
    def __init__(self):
        pass

    def get_landmarks(self, img):
        rects = config.detector(img, 1)
        # TODO recognize this face

        if len(rects) == 0:
            print("No faces found")
            os._exit(1)

        return numpy.matrix([[p.x, p.y] for p in config.predictor(img, rects[0]).parts()])

    def draw_convex_hull(self, img, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(img, points, color=color)

    def draw_hull(self, img, points, color):
        # points = cv2.convexHull(points)
        cv2.fillPoly(img, [points], color=color)

    def generate_mask(self, im, lm):
        img = numpy.zeros(im.shape[:2], dtype=numpy.uint8)

        for group in config.OVERLAY_POINTS:
            #self.draw_convex_hull(img, lm[group], 255)
            self.draw_hull(img, lm[group], 255)

            #img = numpy.array([img, img, img]).transpose((1, 2, 0))

            #img = cv2.GaussianBlur(img, (config.FEATHER_AMOUNT, config.FEATHER_AMOUNT), 0) # * 1.0
            #img = cv2.GaussianBlur(img, (config.FEATHER_AMOUNT, config.FEATHER_AMOUNT), 0)

        for group in config.REMOVE_POINTS:
            #print(group)
            self.draw_hull(img, lm[group], 0)

            #img = numpy.array([img, img, img]).transpose((1, 2, 0))

            #img = (cv2.GaussianBlur(img, (config.FEATHER_AMOUNT, config.FEATHER_AMOUNT), 0) > 0) * 1.0
            #img = cv2.GaussianBlur(img, (config.FEATHER_AMOUNT, config.FEATHER_AMOUNT), 0)
        
        return img


    def align(self, pt1, pt2):
        pt1 = pt1.astype(numpy.float64)
        pt2 = pt2.astype(numpy.float64)
        c1 = numpy.mean(pt1, axis=0)
        c2 = numpy.mean(pt2, axis=0)
        pt1 -= c1
        pt2 -= c2
        s1 = numpy.std(pt1)
        s2 = numpy.std(pt2)
        pt1 /= s1
        pt2 /= s2

        U, S, Vt = numpy.linalg.svd(pt1.T * pt2)
        R = (U * Vt).T

        return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

    def warp_imgs(self, base_img, M, shapes):
        res_im = numpy.zeros(shapes, dtype=base_img.dtype)
        cv2.warpAffine(base_img, M[:2], (shapes[1], shapes[0]), dst=res_im, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
        return res_im

    def generate_transform_matrix(self, pt1, pt2):
        #print(pt1, pt2)
        return cv2.getAffineTransform(pt1, pt2)

    def generate_affine(self, src, M):
        return cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))

    def generate_meshmask(self, im, pts):
        img = numpy.zeros(im.shape[:2], dtype=numpy.uint8)
        self.draw_hull(img, pts, 255)
        return img

    def transform_mesh(self, img1, img2, lm1, lm2):
        res_im = numpy.zeros(img2.shape, dtype=img2.dtype)
        triangle_mask = numpy.zeros(img2.shape[:2], dtype=numpy.uint8)
        clean_mask = numpy.zeros(img2.shape[:2], dtype=numpy.uint8)
        for poly in config.FACE_POLYGONS:
            M = self.generate_transform_matrix(numpy.float32(lm2[poly]), numpy.float32(lm1[poly]))
            #print(lm2[poly])
            im = cv2.bitwise_and(img2, img2, mask=self.generate_meshmask(img2, numpy.int32(lm2[poly])))
            im = self.generate_affine(im, M)
            new_mask = self.generate_meshmask(img2, numpy.int32(lm1[poly]))
            new_mask2 = cv2.threshold(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
            contour_mask = numpy.zeros(img2.shape[:2], dtype=numpy.uint8)
            #cv2.polylines(contour_mask, lm1[poly], True, 255, 5)
            for p in range(1, len(lm1[poly]) + 1):
                cv2.line(contour_mask, (lm1[poly][(p%3)][0,0], lm1[poly][(p%3)][0,1]), (lm1[poly][p-1][0,0], lm1[poly][p-1][0,1]), 255, config.FACE_LINES_CLEARANCE)
            xor_mask = cv2.bitwise_xor(new_mask, new_mask2)
            clean_mask = cv2.add(clean_mask, xor_mask)
            triangle_mask = cv2.add(triangle_mask, contour_mask)
            #clean_mask = cv2.subtract(new_mask2, xor_mask)
            #cv2.imshow("Sub", new_mask2)
            #cv2.imshow("Img", im)
            #cv2.imshow("Orig", new_mask)
            #cv2.imshow("Xor", cv2.bitwise_xor(new_mask, new_mask2))
            #cv2.imshow("Clean", cv2.bitwise_and(im, im, mask=clean_mask))
            #cv2.imshow("Contour", contour_mask)
            #cv2.waitKey(0)
            reversed_facemask = cv2.bitwise_not(new_mask)
            res_im = cv2.bitwise_and(res_im, res_im, mask=reversed_facemask)
            im = cv2.bitwise_and(im, im, mask=new_mask)
            res_im = cv2.add(res_im, im)
        # Generate threshold mask
        cv2.imshow("Clean", triangle_mask)
        cv2.waitKey(0)
        r_triangle_mask = cv2.bitwise_not(triangle_mask)
        gaps = cv2.bitwise_and(img1, img1, mask=triangle_mask)
        #res_im = cv2.bitwise_and(res_im, res_im, mask=cv2.bitwise_not(clean_mask))
        res_im = cv2.bitwise_and(res_im, res_im, mask=r_triangle_mask)
        #tmask = cv2.threshold(cv2.cvtColor(res_im, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
        #reversed_tmask = cv2.bitwise_not(tmask)
        #gaps = cv2.bitwise_and(img1, img1, mask=reversed_tmask)
        res_im = cv2.add(gaps, res_im)
        return res_im, triangle_mask

    def merge_imgs(self, img1, img2):
        return cv2.add(img1, img2)
        # return img1 * (1.0 - img2) + img2
        # return img1 + img2

    def correct_colours(self, im1, im2, landmarks1):
        #blur_amount = config.COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
        #                      numpy.mean(landmarks1[config.LEFT_EYE_POINTS], axis=0) -
        #                      numpy.mean(landmarks1[config.RIGHT_EYE_POINTS], axis=0))
        #blur_amount = int(blur_amount)
        #print(blur_amount)
        #if blur_amount % 2 == 0:
        #    blur_amount += 1
        blur_amount = 11
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        cv2.imshow("Orig", im1_blur)
        cv2.imshow("Mask", im2_blur)
        cv2.waitKey(0)

        im1_gray = cv2.cvtColor(im1_blur, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2_blur, cv2.COLOR_BGR2GRAY)

        res = numpy.zeros(im2.shape, dtype=im2.dtype)

        rows, cols, _ = im2.shape

        #for i in range(rows):
            #for j in range(cols):
                #print(i, j, numpy.take(im2, numpy.array([i, j], dtype=numpy.uint8), axis=0))
                #delta = im2_gray.item(i, j) - im1_gray.item(i, j)
        delta = numpy.subtract(im2_gray, im1_gray)
                #color = list(map(lambda x, y: (lambda i: 255 if i > 255 else i)(x + y), numpy.take(im2, (i, j)), [delta, delta, delta]))
        m_delta = numpy.expand_dims(delta, axis=2)

        m_delta = numpy.repeat(m_delta, 3, axis=2)

        cv2.imshow("Delta", m_delta)

        color = numpy.add(im2, m_delta)

        #res = numpy.add(im2, color)

        """
        def process_axis(x):
            for i in range(len(x)):
                if x[i] > 255:
                    x[i] = 255
                elif x[i] < 0:
                    x[i] = 0

        res = numpy.apply_along_axis(process_axis, 2, res)
        """
        #res = cv2.add(im2, color)

                #print(color)
                #color_res = numpy.array(color)
                #for c in range(len(color)):
                    #print(color_res[0])
                    #color_res[0, c] = color[c]
                #print(im2[i][j], color)
                #for l in range(res[i, j].size):
                    #print(res.item((i, j)))
                    #res.itemset((i, j, l), color[l])


        #sub = cv2.subtract(im1_gray, im2_gray)

        #cv2.imshow("Sub", sub)

        #im2 = cv2.add(im2, cv2.merge((sub, sub, sub)))

        #cv2.imshow("res", im2)
        #cv2.waitKey(0)

        # Avoid divide-by-zero errors.
        #im2_blur += 128 * 

        #return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) / im2_blur.astype(numpy.float64))
        return color

def main(*args, **kwargs):
    ptf = PointFinder()
    base_img = cv2.imread(sys.argv[1])
    original_img = cv2.imread(sys.argv[2])
    cloaked_img = cv2.imread(sys.argv[3]) 
    
    print("Generating landmarks...")
    lm1 = ptf.get_landmarks(base_img)
    lm2 = ptf.get_landmarks(original_img)
    
    img_with_landmarks = base_img.copy()
    for idx, point in enumerate(lm1):
        pos = (point[0, 0], point[0, 1])
        cv2.circle(img_with_landmarks, pos, 1, (0,255,0), -1)
    cv2.imshow("face_with_landmarks", img_with_landmarks)
    img_with_landmarks2 = cloaked_img.copy()
    for idx, point in enumerate(lm2):
        pos = (point[0, 0], point[0, 1])
        cv2.circle(img_with_landmarks2, pos, 1, (0,0,255), -1)
    cv2.imshow("face_with_landmarks2", img_with_landmarks2)

    
    print("Generating cloak mask...")
    #       mask = cv2.threshold(cv2.cvtColor(cv2.subtract(cloaked_img, original_img), cv2.COLOR_BGR2GRAY), 2, 255, cv2.THRESH_BINARY)[1]
    #       xor_img = cv2.bitwise_and(cloaked_img, cloaked_img, mask=mask)
    facemask = ptf.generate_mask(base_img, lm1)
    reversed_facemask = facemask.copy()
    cv2.bitwise_not(facemask, reversed_facemask) 
    cv2.imshow("Mask", reversed_facemask)
    #       xor_img = cv2.bitwise_xor(cloaked_img, original_img) * 255
    #       cv2.imshow("Cloaked", xor_img)
    cv2.waitKey(0)
    print("Aligning landmarks...")
    M = ptf.align(lm1[config.ALIGN_POINTS], lm2[config.ALIGN_POINTS])
    print("Warping mask...")
    warped_img = ptf.warp_imgs(cloaked_img, M, base_img.shape)
    #       facemask = ptf.warp_imgs(facemask, M, base_img.shape)
    #       reversed_facemask = ptf.warp_imgs(reversed_facemask, M, base_img.shape)
    #cv2.imshow("Warped", warped_img)
    #cv2.waitKey(0)
    #warped_img = cv2.bitwise_and(warped_img, warped_img, mask=facemask)
    print("Generating face mesh transformations...")
    mesh, cmask = ptf.transform_mesh(cv2.bitwise_and(base_img, base_img, mask=facemask), cloaked_img, lm1, lm2)
    cv2.imshow("Mesh", mesh)
    cv2.waitKey(0)
    mesh = cv2.bitwise_and(mesh, mesh, mask=facemask)
    print("Merging result...")
    reversed_facemask = cv2.bitwise_not(facemask)
    #base_img = cv2.bitwise_and(base_img, base_img, mask=reversed_facemask)
    #result = ptf.merge_imgs(base_img, mesh)
    result = ptf.correct_colours(base_img, mesh, lm1)
    result = ptf.merge_imgs(cv2.bitwise_and(result, result, mask=facemask), cv2.bitwise_and(base_img, base_img, mask=reversed_facemask))
    cv2.imwrite('output.png', result)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    return result


if __name__ == "__main__":
    main()


