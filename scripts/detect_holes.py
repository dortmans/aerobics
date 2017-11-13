#!/usr/bin/env python

"""ROS OpenCV image processing node."""

# For Python2/3 compatibility
from __future__ import print_function
from __future__ import division

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import sys

__author__ = "Eric Dortmans"
__copyright__ = "Copyright 2017, Fontys"

# Define colors for drawing in images
BLACK = (0, 0, 0)
BLUE  = (255, 0, 0)
GREEN = (0, 255, 0)
RED   =  (0, 0, 255)
WHITE = (255, 255, 255)

class ImageProcessor:
    """This class processes ROS Images using OpenCV

    """

    def __init__(self):
        self.bridge = CvBridge()
        self.image_subscriber = rospy.Subscriber("image_raw", Image, self.on_image_message, queue_size=1)
        self.image_publisher = rospy.Publisher("image_processed", Image, queue_size=1)
        self.process_image_setup()

    def to_cv2(self, image_msg):
        """Convert ROS image message to OpenCV image

        """
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except CvBridgeError as e:
            print(e)
        return image

    def to_imgmsg(self, image):
        """Convert OpenCV image to ROS image message

        """
        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        except CvBridgeError as e:
            print(e)
        return image_msg

    def on_image_message(self, image_msg):
        """Process received ROS image message
        """
        self.image = self.to_cv2(image_msg)
        self.processed_image = self.image
        self.process_image()
        self.image_publisher.publish(self.to_imgmsg(self.processed_image))

    def process_image_setup(self):
        """Setup for image processing. 

        This code will run only once to setup image processing.
        """
        pass

    def process_image(self):
        """Process the image using OpenCV

        This code is run for reach image
        """

        cv2.imshow("image", self.image)

        # -------------------------------------------------

        # Detect tags
        tags = self.detect_tags()

        # Pick a tag
        tag = tags[0]

        # Calculate bird's eye view perspective transform
        H = homography_from_tag(tag)
        _, H_inv = cv2.invert(H)

        # Apply transform
        processed_image = self.apply_transform(H)

        # TODO:
        #   Find holes 
        #   Reverse transform:
        #       use H_inv
        #       or warpPerspective with flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        #

        # -------------------------------------------------

        cv2.imshow("processed_image", self.processed_image)
        cv2.waitKey(3)


    def detect_tags(self):
        """Detect tags in the image

        returns list of tags
        """
        # Apply thresholding
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.bitwise_not(bw)
        # _ ,bw = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)

        # Contour detection
        mode = cv2.RETR_TREE
        if cv2.__version__.startswith('2.'): # OpenCV2
            (contours, hierarchy) = cv2.findContours(bw.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)
        else:
            (_, contours, hierarchy) = cv2.findContours(bw.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)

        hierarchy = hierarchy[0]
        contours = np.array(contours)
        indices = np.array(range(len(contours)))

        # The contours we are looking for have a child
        has_child = hierarchy[:, 2] >= 0
        contours = contours[has_child]
        indices = indices[has_child]

        # They must have a convex quadrilateral shape
        candidates = []
        candidate_indices = []
        for contour, index in zip(contours, indices):
            # approximate the contour
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            candidate = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            # approximated contour must have four points, must be convex and must have minimal size
            if len(candidate) == 4 and cv2.isContourConvex(candidate) and area > 100:
                candidates.append(candidate)
                candidate_indices.append(index)

        # Remove candidates that contain other candidates
        parents = hierarchy[candidate_indices, 3]
        tags = []
        for candidate, index in zip(candidates, candidate_indices):
            if index not in parents:
                tags.append(candidate)
        
        return tags


    def homography_from_tag(tag):
        """Calculate homography from tag

        returns homography matrix
        """
        tag_points = tag[:, 0]

        # Calculate source points
        #   points should be ordered TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT
        pts = np.array(tag_points, dtype="float32")
        src = np.zeros((4, 2), dtype="float32")
        # the top-left point has the smallest sum whereas the
        # bottom-right has the largest sum
        s = pts.sum(axis=1)
        src[0] = pts[np.argmin(s)]
        src[2] = pts[np.argmax(s)]
        # compute the difference between the points -- the top-right
        # will have the minumum difference and the bottom-left will
        # have the maximum difference
        diff = np.diff(pts, axis=1)
        src[1] = pts[np.argmin(diff)]
        src[3] = pts[np.argmax(diff)]

        # Calculate destination points
        #   knowing that the tag is a square
        rect = cv2.minAreaRect(tag)
        size = int(max(rect[1]))
        center = rect[0]
        top_left = (center[0] - size // 2, center[1] - size // 2)
        top_right = (center[0] + size // 2, center[1] - size // 2)
        bottom_right = (center[0] + size // 2, center[1] + size // 2)
        bottom_left = (center[0] - size // 2, center[1] + size // 2)
        dst = np.array([
            top_left,
            top_right,
            bottom_right,
            bottom_left], dtype="float32")

        # Find homography transform from source to destination
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC,5.0)
        #H = cv2.getPerspectiveTransform(src, dst)
        #To compute the inverse perspective transform use cv2.getPerspectiveTransform(dst,src)
        #or use: _, H_inv = cv2.invert(H)

        return H


    def apply_transform(transform):
        """Apply perspective transform on image

        returns warped image
        """

        def translation(x, y):
            return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=float)

        H = transform
        height, width = self.image.shape[:2]

        # Calculate transformed screen corners and size
        corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        new_corners = cv2.perspectiveTransform(corners[None, :, :], H)
        new_x, new_y, new_width, new_height = cv2.boundingRect(new_corners)

        # Warp source image to destination image based on homography
        T = translation(-new_x, -new_y)
        TH = np.dot(T, H)
        warped = cv2.warpPerspective(img, TH, (new_width, new_height))

        return warped


def main(args):
    rospy.init_node('image_processor', anonymous=True)
    ip = ImageProcessor()
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
