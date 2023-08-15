# imports
import cv2 as cv
import numpy as np
# each frame of the video


class Frame():
    """Frame class for each frame of the video

    static variables:
    prev_bbox: previous bounding box of the droplet tracked to calculate the new position
    bbox: bounding box of current position of the droplet being tracked
    is_tracker_initialized: whether or not the tracker needs to be initialized
    outline: whether or not to display the outline
    """
    prev_bbox = None
    bbox = None
    is_tracker_initialized = False
    outline = True
    threshold = 0

    def __init__(self, img, threshold):
        self.image = img
        self.processed_image = self.process_image(threshold)
        self.cimg = img.copy()
        self.droplet_speed = None
        self.droplet_radius = None
        self.jet_length = None
        self.jet_width = None

    def process_image(self, threshold):
        """process the image by adding changing it to grayscale, adding blur, and thresholding the image

        returns:
        thresh1: a thresholded image after processing
        """
        ksize = (5, 5)
        max_val = 255

        self.image = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)
        img_blur = cv.blur(self.image, ksize)
        ret, thresh1 = cv.threshold(
            img_blur, threshold, max_val, cv.THRESH_BINARY_INV)
        return thresh1

    def detect_droplets_and_jet(self):
        """detects the droplets and jet and returns droplet radius, jet width and length

        finds all the contours of the image, finds the roundness and group as droplets or jet

        returns:
        self.cimg: the image with outlines drawn on it (if outline is chosen)
        radius_avg: the average radius of all the droplets in the frame
        jet_width: the width of the jet
        jet_length: the height of the jet
        """
        circles_num = 0
        radius_sum = 0
        radius_avg = 0
        jet_width = 0
        jet_length = 0

        contours, _ = cv.findContours(
            self.processed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            moment = cv.moments(cnt, True)
            if moment["m00"] != 0:
                roundness, area, center_x, center_y = self.detect_roundness(
                    moment, cnt)
                if roundness >= 0.5:
                    circles_num, radius_sum = self.droplet_rad(
                        area, center_x, center_y, circles_num, radius_sum)
                if roundness <= 0.3:
                    jet_width, jet_length = self.jet_width_length(cnt)

        if circles_num > 0:
            radius_avg = radius_sum / circles_num

        return self.cimg, radius_avg, jet_width, jet_length

    def detect_roundness(self, moment, cnt):
        """returns how round the detected contour is

        uses the equation roundness = 4*pi*area/perimeter^2

        args:
        moment: the moment array for the contour of all the moments up to order 3
        cnt: coordinates that form the contour

        return:
        roundness: integer between 0 (not round) and 1 (circle) that displays roundness
        area: the area of the contour
        center_x: x coordinate of the contour center
        center_y: y coordinate of the contour center
        """
        center_x = int(moment["m10"] / moment["m00"])
        center_y = int(moment["m01"] / moment["m00"])
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        roundness = (4 * np.pi * area)/(perimeter ** 2)
        return roundness, area, center_x, center_y

    def droplet_rad(self, area, center_x, center_y, circles_num, radius_sum):
        """determines the radius of a droplet

        uses r = sqrt(A/pi) to find the radius.
        updates radius total and total number of droplets.

        args:
        area: area of the droplet
        center_x: x coordinate of the contour center
        center_y: y coordinate of the contour center
        circles_num: total number of droplets
        radius_num: sum of the radius

        returns:
        circles_num: updated total number of droplets
        radius_num: updated sum of radius
        """
        radius = np.sqrt(area/np.pi)
        circles_num += 1
        radius_sum += radius
        cv.circle(self.cimg, (center_x, center_y), int(radius), (0, 255, 0), 3)
        return circles_num, radius_sum

    def jet_width_length(self, cnt):
        """returns the jet width and length

        find the height, the different between max and min values.
        find width by finding the difference between the means of the x values' array split in half

        args:
        cnt: the contour of the jet

        returns:
        jet_width: the width of the jet
        jet_length: the height of the jet

        """
        cnt = np.squeeze(cnt)

        max = np.amax(cnt, axis=0)
        max_y = max[1]
        min = np.amin(cnt, axis=0)
        min_y = min[1]

        x_vals = cnt[:, 0]
        x_vals = np.sort(x_vals)
        seperator = int(x_vals.size/2)
        first_half = x_vals[:seperator]
        second_half = x_vals[seperator:]
        x_val_left = int(np.mean(first_half))
        x_val_right = int(np.mean(second_half))
        cv.rectangle(self.cimg, (x_val_left, min_y),
                     (x_val_right, max_y), (255, 0, 0), 2, 1)
        jet_width = x_val_right - x_val_left
        jet_length = max_y - min_y
        return jet_width, jet_length

    def bounding_rect_droplets(self):
        """returns bounding box of a droplet so its easier to track the droplet speed

        returns:
        x: top left x coord
        y: top left y coord
        w: width of rectangle
        h: height of rectangle
        """
        w, h, x, y = 0, 0, 0, 0
        contours, hier = cv.findContours(
            self.processed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if w*h > 5000 and w*h < 11000:
                return w, h, x, y
        x, y, w, h = cv.boundingRect(contours[0])
        return x, y, w, h

    def init_tracker(self, tracker):
        """initialzes the tracker used to track droplet speed

        args:
        tracker: tracker object
        """
        w, h, rect_x, rect_y = self.bounding_rect_droplets()
        Frame.bbox = [rect_x, rect_y, w, h]
        Frame.prev_bbox = Frame.bbox
        tracker.init(self.image, Frame.bbox)

    def update_tracker(self, tracker, temp):
        """updates the tracker to find droplet in new frame

        args:
        tracker: tracker object
        temp: variable to see wether to run the tracker
        """
        if temp:
            ret, Frame.bbox = tracker.update(self.image)
            if ret:
                p1 = (int(Frame.bbox[0]), int(Frame.bbox[1]))
                p2 = (int(Frame.bbox[0] + Frame.bbox[2]),
                      int(Frame.bbox[1] + Frame.bbox[3]))
                if Frame.bbox[0] <= 0 or Frame.bbox[1] <= 0:
                    temp = False
                    Frame.is_tracker_initialized = False# imports
import cv2 as cv
import numpy as np
# each frame of the video


class Frame():
    """Frame class for each frame of the video

    static variables:
    prev_bbox: previous bounding box of the droplet tracked to calculate the new position
    bbox: bounding box of current position of the droplet being tracked
    is_tracker_initialized: whether or not the tracker needs to be initialized
    outline: whether or not to display the outline
    """
    outline = True
    threshold = 0

    def __init__(self, img, threshold):
        self.image = img
        self.processed_image = self.process_image(threshold)
        self.cimg = img.copy()
        self.droplet_speed = None
        self.droplet_radius = None
        self.jet_length = None
        self.jet_width = None

    def process_image(self, threshold):
        """process the image by adding changing it to grayscale, adding blur, and thresholding the image

        returns:
        thresh1: a thresholded image after processing
        """
        ksize = (5, 5)
        max_val = 255
        kernel = np.ones((5, 5), np.uint8)

        self.image = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)
        img_erode = cv.erode(self.image, kernel, iterations=1)

        img_blur = cv.blur(img_erode, ksize)
        ret, thresh1 = cv.threshold(
            img_blur, threshold, max_val, cv.THRESH_BINARY_INV)
        return thresh1

    def detect_droplets_and_jet(self):
        """detects the droplets and jet and returns droplet radius, jet width and length

        finds all the contours of the image, finds the roundness and group as droplets or jet

        returns:
        self.cimg: the image with outlines drawn on it (if outline is chosen)
        radius_avg: the average radius of all the droplets in the frame
        jet_width: the width of the jet
        jet_length: the height of the jet
        """
        circles_num = 0
        radius_sum = 0
        radius_avg = 0
        jet_width = 0
        jet_length = 0
        droplet_list = []

        contours, _ = cv.findContours(
            self.processed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            moment = cv.moments(cnt, True)
            if moment["m00"] != 0:
                roundness, area, center_x, center_y = self.detect_roundness(
                    moment, cnt)
                if roundness >= 0.5:
                    circles_num, radius_sum, radius = self.droplet_rad(
                        area, center_x, center_y, circles_num, radius_sum)
                    droplet_list.append([center_y, radius])

                if roundness <= 0.3:
                    jet_width, jet_length = self.jet_width_length(cnt)

        if circles_num > 0:
            radius_avg = radius_sum / circles_num

        return self.cimg, droplet_list, radius_avg, jet_width, jet_length

    def detect_roundness(self, moment, cnt):
        """returns how round the detected contour is

        uses the equation roundness = 4*pi*area/perimeter^2

        args:
        moment: the moment array for the contour of all the moments up to order 3
        cnt: coordinates that form the contour

        return:
        roundness: integer between 0 (not round) and 1 (circle) that displays roundness
        area: the area of the contour
        center_x: x coordinate of the contour center
        center_y: y coordinate of the contour center
        """
        center_x = int(moment["m10"] / moment["m00"])
        center_y = int(moment["m01"] / moment["m00"])
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        roundness = (4 * np.pi * area)/(perimeter ** 2)
        return roundness, area, center_x, center_y

    def droplet_rad(self, area, center_x, center_y, circles_num, radius_sum):
        """determines the radius of a droplet

        uses r = sqrt(A/pi) to find the radius.
        updates radius total and total number of droplets.

        args:
        area: area of the droplet
        center_x: x coordinate of the contour center
        center_y: y coordinate of the contour center
        circles_num: total number of droplets
        radius_num: sum of the radius

        returns:
        circles_num: updated total number of droplets
        radius_num: updated sum of radius
        """
        radius = np.sqrt(area/np.pi)
        circles_num += 1
        radius_sum += radius
        cv.circle(self.cimg, (center_x, center_y), int(radius), (0, 255, 0), 3)
        return circles_num, radius_sum, radius

    def jet_width_length(self, cnt):
        """returns the jet width and length

        find the height, the different between max and min values.
        find width by finding the difference between the means of the x values' array split in half

        args:
        cnt: the contour of the jet

        returns:
        jet_width: the width of the jet
        jet_length: the height of the jet

        """
        cnt = np.squeeze(cnt)

        max = np.amax(cnt, axis=0)
        max_y = max[1]
        min = np.amin(cnt, axis=0)
        min_y = min[1]

        x_vals = cnt[:, 0]
        x_vals = np.sort(x_vals)
        seperator = int(x_vals.size/2)
        first_half = x_vals[:seperator]
        second_half = x_vals[seperator:]
        x_val_left = int(np.mean(first_half))
        x_val_right = int(np.mean(second_half))
        cv.rectangle(self.cimg, (x_val_left, min_y),
                     (x_val_right, max_y), (255, 0, 0), 2, 1)
        jet_width = x_val_right - x_val_left
        jet_length = max_y - min_y
        return jet_width, jet_length

    def calculate_droplet_speed(self, lists, prev_lists):
        target_pairs = lists
        reference_pairs = prev_lists

        closest_pairs = []
        distance = 0

        for target_pair, reference_pair in zip(target_pairs, reference_pairs):
            y_intercept_difference = target_pair[0] - reference_pair[0]
            filtered_pairs = [
                pair for pair in reference_pairs if pair[0] > target_pair[0]]
            filtered_pairs_array = np.array(filtered_pairs)

            if len(filtered_pairs_array) != 0:
                distances = np.linalg.norm(
                    filtered_pairs_array - np.array(target_pair), axis=1)
                closest_index = np.argmin(distances)
                closest_pair = filtered_pairs[closest_index]
                closest_pairs.append((target_pair[0], closest_pair[0]))

        if len(closest_pairs) > 0:
            for pair in closest_pairs:
                distance += pair[1] - pair[0]
            distance /= len(closest_pairs)

        time = 1/6400
        average_speed = distance / time
        return average_speed

            return temp, Frame.bbox
        else:
            return temp, [0, 0, 0, 0]

    def calculate_droplet_speed(self):
        """calculates the droplet speed

        uses speed = distance/time

        returns:
        speed: the speed of the droplet
        """
        time = 1/6400
        distance = abs(Frame.prev_bbox[1] - Frame.bbox[1])
        speed = distance / time
        return speed
