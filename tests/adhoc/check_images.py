import cv2
img = cv2.imread("data/real_samples/clear_reference.png")
if img is None:
    print("clear_reference.png DOES NOT EXIST")
else:
    print("clear_reference.png shape:", img.shape)
    
img2 = cv2.imread("data/real_samples/cloudy_0.png")
print("cloudy_0.png shape:", img2.shape)
