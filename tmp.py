import cv2
import numpy as np
import matplotlib.pyplot as plt

def otsu(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape[:2]
    pixel = h * w
    threshold_k = 0
    max_var = .0
    for k in range(255):
        c1 = img_gray[img_gray <= k]
        p1 = len(c1) / pixel
        if p1 == 0:
            continue
        elif p1 == 1:
            break
        MG = np.sum(img_gray) / pixel
        m = np.sum(c1) / pixel
        d = (MG*p1 - m) ** 2 / (p1 * (1 - p1))
        if d > max_var:
            max_var = d
            threshold_k = k
    offset=70
    adjusted_threshold = threshold_k - offset
    adjusted_threshold = min(max(adjusted_threshold, 0), 255)
    threshold_k = adjusted_threshold
    
    binarized_img = img_gray.copy()
    binarized_img[binarized_img <= threshold_k] = 0
    binarized_img[binarized_img > threshold_k] = 255
    
    print(f"阈值: {threshold_k}")
    return binarized_img

img0 = cv2.imread('D:\HP\Downloads\image.png')
img_gray0 = otsu(img0.copy())

# 1. 反转二值图像，使孔洞变为白色
img_binary_inverted = cv2.bitwise_not(img_gray0)
# 2. 查找轮廓
contours, hierarchy = cv2.findContours(img_binary_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 3. 选择面积最大的轮廓作为孔洞
hole_contour = max(contours, key=cv2.contourArea)
# 4. 计算距离变换
dist_transform = cv2.distanceTransform(img_binary_inverted, cv2.DIST_L2, 5)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
center = max_loc
radius = int(max_val)
circle_area = np.pi * (radius**2)

print(f"最大内接圆圆心: {center}")
print(f"最大内接圆半径: {radius} 像素")
print(f"最大内接圆面积: {circle_area:.2f} 平方像素")

img_with_circle = img0.copy()
cv2.circle(img_with_circle, center, radius, (0, 255, 0), 2)  # 绿色圆圈
cv2.circle(img_with_circle, center, 3, (0, 0, 255), -1)  # 红色圆心
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(img_gray0, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_with_circle, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.tight_layout()
plt.show()

cv2.imwrite('D:\HP\Downloads\image_with_inscribed_circle.png', img_with_circle)
cv2.imwrite('D:\HP\Downloads\image_binarized.png', img_gray0)