import cv2
import os
import numpy as np

DIM = 128


def draw_background():
    img = np.zeros([DIM, DIM, 1], dtype=np.uint8)
    img.fill(0)
    return img


def draw_rect():
    img_outline, img_filled = draw_background(), draw_background()
    x1, x2, x3, x4 = np.random.randint(0, DIM, 4)
    cv2.rectangle(img_outline, (x1, x2), (x3, x4), (255, 255, 255), 2)
    cv2.rectangle(img_filled, (x1, x2), (x3, x4), (255, 255, 255), cv2.FILLED)
    return img_outline, img_filled


def draw_circle():
    img_outline, img_filled = draw_background(), draw_background()
    r = np.random.randint(0, DIM / 2)
    c1, c2 = np.random.randint(r, DIM - r, 2)
    cv2.circle(img_outline, (c1, c2), r, (255, 255, 255), 2)
    cv2.circle(img_filled, (c1, c2), r, (255, 255, 255), cv2.FILLED)
    return img_outline, img_filled


def draw_dot():
    img_dot = draw_background()
    r = 10
    c1, c2 = np.random.randint(r, DIM - r, 2)
    cv2.circle(img_dot, (c1, c2), r, (255, 255, 255), cv2.FILLED)
    return img_dot, img_dot


def draw_star():
    img_outline, img_filled = draw_background(), draw_background()
    x, y = np.random.randint(0, DIM, 2)
    s = np.random.randint(0, 50)

    r_18 = s * np.sin(np.radians(18))
    r_36 = s * np.sin(np.radians(36))
    r_72 = s * np.sin(np.radians(72))
    r_54 = s * np.sin(np.radians(54))
    pts = np.array([(x, y),
                    (x - r_18, y - r_72),
                    (x - r_18 - s, y - r_72),
                    (x - r_18 - s + r_54, y - r_72 - r_36),
                    (x - r_54, y - r_72 - r_36 - r_72),
                    (x, y - r_72 - r_72),
                    (x + r_54, y - r_72 - r_36 - r_72),
                    (x + r_18 + s - r_54, y - r_72 - r_36),
                    (x + r_18 + s, y - r_72),
                    (x + r_18, y - r_72)], np.int32)

    cv2.polylines(img_outline, [pts], True, (255, 255, 255), 2)
    cv2.fillPoly(img_filled, [pts], (255, 255, 255))
    return img_outline, img_filled


def draw_ellipse():
    img_outline, img_filled = draw_background(), draw_background()
    x, y = np.random.randint(1, DIM, 2)
    ax1, ax2 = np.random.randint(1, 30, 2)
    angle = np.random.randint(1, 360)
    cv2.ellipse(img_outline, (x, y), (ax1, ax2), angle, 0.0, 360.0, (255, 255, 255), 2)
    cv2.ellipse(img_filled, (x, y), (ax1, ax2), angle, 0.0, 360.0, (255, 255, 255), cv2.FILLED)
    return img_outline, img_filled


def main(num_train=100, num_test=10, seed=123):
    # num_train: num of train images for each shape
    # num_test: num of test images for each shape

    cwd = os.getcwd()
    outline_path = os.path.join(cwd, r'train/image')
    filled_path = os.path.join(cwd, r'train/label')
    test_path = os.path.join(cwd, r'test/image')
    if not os.path.exists(outline_path):
        os.makedirs(outline_path)
    if not os.path.exists(filled_path):
        os.makedirs(filled_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # generate training images
    for i in range(num_train):
        np.random.seed(i * seed)

        # draw outline and filled images
        rect_outline, rect_filled = draw_rect()
        circle_outline, circle_filled = draw_circle()
        star_outline, star_filled = draw_star()
        ellipse_outline, ellipse_filled = draw_ellipse()
        dot_outline, dot_filled = draw_dot()

        shape_names = ['rectangle', 'circle', 'star', 'ellipse', 'dot']
        outlines = [rect_outline, circle_outline, star_outline, ellipse_outline, dot_outline]
        filled = [rect_filled, circle_filled, star_filled, ellipse_filled, dot_filled]

        # write image files
        for j in range(len(shape_names)):
            f_name = '/' + shape_names[j] + '_{}.png'.format(i)
            outline_final = outline_path + f_name
            filled_final = filled_path + f_name
            cv2.imwrite(outline_final, outlines[j])
            cv2.imwrite(filled_final, filled[j])

    # generate testing images
    for i in range(num_test):
        np.random.seed(i + seed)

        # draw outline and filled images
        rect_outline, rect_filled = draw_rect()
        circle_outline, circle_filled = draw_circle()
        star_outline, star_filled = draw_star()
        ellipse_outline, ellipse_filled = draw_ellipse()
        dot_outline, dot_filled = draw_dot()

        shape_names = ['rectangle', 'circle', 'star', 'ellipse', 'dot']
        outlines = [rect_outline, circle_outline, star_outline, ellipse_outline, dot_outline]

        for j in range(len(shape_names)):
            f_name = '/' + shape_names[j] + '_{}.png'.format(i + 1)
            outline_final = test_path + f_name
            cv2.imwrite(outline_final, outlines[j])


if __name__ == "__main__":
    main()
