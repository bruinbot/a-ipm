import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

for p, path in enumerate(os.listdir("input")):
    original_image = cv2.cvtColor(cv2.imread(os.path.join('input', path)), cv2.COLOR_BGR2RGB)
    segmented_image = cv2.cvtColor(cv2.imread(os.path.join('output', path)), cv2.COLOR_BGR2RGB)
    TARGET_H, TARGET_W = 500, 500

    ################
    # OpenCV
    ################
    # Vertices coordinates in the source image
    s = np.array([[830, 598],
                [868, 568],
                [1285, 598],
                [1248, 567]], dtype=np.float32)

    # Vertices coordinates in the destination image
    t = np.array([[177, 231],
                [213, 231],
                [178, 264],
                [216, 264]], dtype=np.float32)

    # Warp the input image
    M = cv2.getPerspectiveTransform(s, t)
    original_warped = cv2.warpPerspective(original_image, M, (TARGET_W, TARGET_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)

    M = cv2.getPerspectiveTransform(s, t)
    segmented_warped = cv2.warpPerspective(segmented_image, M, (TARGET_W, TARGET_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)

    # Draw results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    cv2.line(original_warped, (200, 0), (200, TARGET_H), (255, 0, 0), 2)
    cv2.circle(original_warped, (0, TARGET_H // 2), 200, (255, 0, 0), 2)
    original_warped = np.rot90(original_warped)
    ax2.imshow(original_warped)
    ax2.set_title('Original Image IPM')
    ax3.imshow(segmented_image)
    ax3.set_title('Segmented Image')
    cv2.line(segmented_warped, (200, 0), (200, TARGET_H), (255, 0, 0), 2)
    cv2.circle(segmented_warped, (0, TARGET_H // 2), 200, (255, 0, 0), 2)
    segmented_warped = np.rot90(segmented_warped)
    ax4.imshow(segmented_warped)
    ax4.set_title('Segmented Image IPM')
    plt.tight_layout()
    plt.show()
    
    if p == 5:
        break
    