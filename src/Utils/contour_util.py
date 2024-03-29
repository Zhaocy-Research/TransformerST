import pandas as pd
import numpy as np
import cv2
def scan_contour(spots, scan_x=True, shape="hexagon"):
	"""
 	This function scans a set of spatial transcriptomic spots to create a contour boundary based on their spatial distribution. 
  	"""
	#shape="hexagon" # For 10X Vsium, shape="square" for ST data
	if scan_x:
		array_a="array_row"
		array_b="array_col"
		pixel_a="pxl_row_in_fullres"
		pixel_b="pxl_col_in_fullres"
	else:
		array_a="array_col"
		array_b="array_row"
		pixel_a="pxl_col_in_fullres"
		pixel_b="pxl_row_in_fullres"
	upper, lower={}, {}
	uniq_array_a=sorted(set(spots[array_a]))
	if shape=="hexagon":
		for i in range(len(uniq_array_a)-1):
			a1=uniq_array_a[i]
			a2=uniq_array_a[i+1]
			group=spots.loc[spots[array_a].isin([a1, a2]),:]
			lower[np.min(group[pixel_a])]=np.min(group[pixel_b])
			upper[np.min(group[pixel_a])]=np.max(group[pixel_b])
			#print(a1, lower[np.min(group[pixel_a])], upper[np.min(group[pixel_a])])
		a1=uniq_array_a[-1]
		a2=uniq_array_a[-2]
		group=spots.loc[spots[array_a].isin([a1, a2]),:]
		lower[np.min(group[pixel_a])]=np.min(group[pixel_b])
		upper[np.min(group[pixel_a])]=np.max(group[pixel_b])
		#print(a1, lower[np.min(group[pixel_a])], upper[np.min(group[pixel_a])])
	elif shape=="square":
		for i in range(len(uniq_array_a)-1):
			a1=uniq_array_a[i]
			group=spots.loc[spots[array_a]==a1,:]
			lower[np.min(group[pixel_a])]=np.min(group[pixel_b])
			upper[np.min(group[pixel_a])]=np.max(group[pixel_b])
		a1=uniq_array_a[-1]
		group=spots.loc[spots[array_a]==a1,:]
		lower[np.min(group[pixel_a])]=np.min(group[pixel_b])
		upper[np.min(group[pixel_a])]=np.max(group[pixel_b])
	else:
		print("Error, unknown shape, pls specify 'square' or 'hexagon'.")
	lower=np.array(list(lower.items())[::-1]).astype("int32")
	upper=np.array(list(upper.items())).astype("int32")
	cnt=np.concatenate((upper, lower), axis=0)
	cnt=cnt.reshape(cnt.shape[0], 1, 2)
	if scan_x:
		cnt=cnt[:, : , [1, 0]]
	return cnt

def cv2_detect_contour(img, 
	CANNY_THRESH_1 = 100,
	CANNY_THRESH_2 = 200,
	apertureSize=5,
	L2gradient = True,
	all_cnt_info=False):
	"""
 	 Utilizes OpenCV's capabilities to detect contours in an image. 
   	It first converts the image to grayscale (if necessary) and applies Canny edge detection to find edges in the image. 
  	"""
	if len(img.shape)==3:
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	elif len(img.shape)==2:
		gray=(img*((1, 255)[np.max(img)<=1])).astype(np.uint8)
	else:
		print("Image format error!")
	edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2,apertureSize = apertureSize, L2gradient = L2gradient)
	edges = cv2.dilate(edges, None)
	edges = cv2.erode(edges, None)
	cnt_info = []
	cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	for c in cnts:
		cnt_info.append((c,cv2.isContourConvex(c),cv2.contourArea(c),))
	cnt_info = sorted(cnt_info, key=lambda c: c[2], reverse=True)
	cnt=cnt_info[0][0]
	if all_cnt_info:
		return cnt_info
	else:
		return cnt


def cut_contour_boundary(cnt, x_min, x_max, y_min, y_max, enlarge):
	"""
 	Adjusts a given contour to fit within specified boundary limits. 
  	It ensures that the contour coordinates do not exceed the provided minimum and maximum x and y values. 
  	"""
	ret=cnt.copy()
	ret[:, : , 0][ret[:, : , 0]>y_max]=y_max
	ret[:, : , 0][ret[:, : , 0]<y_min]=y_min
	ret[:, : , 1][ret[:, : , 1]>x_max]=x_max
	ret[:, : , 1][ret[:, : , 1]<x_min]=x_min
	return ret

def scale_contour(cnt, scale):
	"""
 	Scales a contour by a specified factor around its centroid. 
  	It calculates the center of the contour and normalizes the contour points relative to this center.
  	"""
	M = cv2.moments(cnt)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	cnt_norm = cnt - [cx, cy]
	cnt_scaled = cnt_norm * scale
	cnt_scaled = cnt_scaled + [cx, cy]
	cnt_scaled = cnt_scaled.astype(np.int32)
	return cnt_scaled








