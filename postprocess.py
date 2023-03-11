import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

for p, path in enumerate(os.listdir("output")):
    # reads image
    img = cv2.imread(os.path.join('output', path))
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    road_color = np.array([128, 64, 128])
    road_hsv = (150, 128, 128)

    # extract mask for road
    mask = cv2.inRange(hsv_img, road_hsv, road_hsv)
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    plt.tight_layout()

    ax1.imshow(img)

    ax2.imshow(masked_img)

    # erode mask to account for extra space
    EROSION_THRESHOLD = 50
    kernel = np.ones((EROSION_THRESHOLD, EROSION_THRESHOLD), np.uint8)
    img_erosion = cv2.erode(masked_img, kernel, iterations=2)
    ax3.imshow(img_erosion)

    bot_region = np.zeros_like(img)
    ellipse_axis = 450

    W = img.shape[1]
    H = img.shape[0]

    # making ellipse for bot physical traversable region
    bot_region = cv2.ellipse(bot_region, (W // 2, H), (W // 2, ellipse_axis), 0, 180, 360, (255, 255, 255), -1)
    traversable = cv2.bitwise_and(img_erosion, bot_region)

    start_point = (W // 2, H)

    # calculate end_point (midpoint deviation strategy)
    # TODO: CLEAN UP ALGORITHM
    left_count = 0
    right_count = 0
    consider_mid = True
    spacing = 10
    GROUP_THRESHOLD = 20
    alpha = 0.5
    for i in range(0, W // 2, spacing):
        updated = np.copy(traversable)
        y = int(np.sqrt((1 - ((i ** 2) / ((W // 2) ** 2))) * (ellipse_axis ** 2)))
        # add lines on each side to show current points being processed
        cv2.arrowedLine(updated, start_point, (W // 2 - i, H - y), (0, 0, 255), 10)
        cv2.arrowedLine(updated, start_point, (W // 2 + i, H - y), (0, 0, 255), 10)
        # add line(s) to show current best direction of traversal
        if consider_mid:
            end_point = (W // 2 + (right_count - left_count) // 2 * spacing, H - y)
            cv2.arrowedLine(updated, start_point, end_point, (255, 87, 81), 10)
            lx = left_count * spacing
            ly = int(np.sqrt((1 - ((lx ** 2) / ((W // 2) ** 2))) * (ellipse_axis ** 2)))
            start_angle = np.arctan(lx / ly) * (180 / np.pi)
            rx = right_count * spacing
            ry = int(np.sqrt((1 - ((rx ** 2) / ((W // 2) ** 2))) * (ellipse_axis ** 2)))
            end_angle = np.arctan(rx / ry) * (180 / np.pi)
            overlay = updated.copy()
            cv2.ellipse(overlay, (W // 2, H), (W // 2, ellipse_axis), 0, 270 - start_angle // 2, 270 + end_angle // 2, (0, 255, 0), -1)
            updated = cv2.addWeighted(overlay, alpha, updated, 1 - alpha, 0)
        else:
            cv2.arrowedLine(updated, start_point, (W // 2 + (right_count - left_count) // 2 * spacing, H - y), (255, 87, 81), 10)
            cv2.arrowedLine(updated, start_point, (W // 2 + (right_count - left_count) // 2 * spacing, H - y), (255, 87, 81), 10)

        ax4.imshow(updated)
        fig.canvas.draw()
        fig.canvas.flush_events()
        ax4.clear()

        y = int(np.sqrt((1 - ((i ** 2) / ((W // 2) ** 2))) * (ellipse_axis ** 2)))
        THRESHOLD = 10
        y -= THRESHOLD
        if np.array_equal(traversable[H - y - 1][W // 2 - i - 1], road_color):
            left_count += 1
            if consider_mid and right_count + left_count == GROUP_THRESHOLD:
                x = (left_count + GROUP_THRESHOLD // 2) * spacing
                y = int(np.sqrt((1 - ((x ** 2) / ((W // 2) ** 2))) * (ellipse_axis ** 2)))
                end_point = (W // 2 + x, H - y)
                break
            if left_count == GROUP_THRESHOLD:
                x = -(i - (GROUP_THRESHOLD // 2) * spacing)
                y = int(np.sqrt((1 - ((x ** 2) / ((W // 2) ** 2))) * (ellipse_axis ** 2)))
                end_point = (W // 2 + x, H - y)
                break
        else:
            left_count = 0
            consider_mid = False
        if np.array_equal(traversable[H - y - 1][W // 2 + i - 1], road_color):
            right_count += 1
            if consider_mid and right_count + left_count == GROUP_THRESHOLD:
                x = (right_count - GROUP_THRESHOLD // 2) * spacing
                y = int(np.sqrt((1 - ((x ** 2) / ((W // 2) ** 2))) * (ellipse_axis ** 2)))
                end_point = (W // 2 + x, H - y)
                break
            if right_count == GROUP_THRESHOLD:
                x = i - (GROUP_THRESHOLD // 2) * spacing
                y = int(np.sqrt((1 - ((x ** 2) / ((W // 2) ** 2))) * (ellipse_axis ** 2)))
                end_point = (W // 2 + x, H - y)
                break
        else:
            right_count = 0
            consider_mid = False
    cv2.arrowedLine(traversable, start_point, end_point, (0, 255, 0), 10)
    ax4.imshow(traversable)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(5)
    # break
    if p == 15:
        break
