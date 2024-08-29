import cv2
import glob
import os
import numpy as np

inputDirectory = "TestVideo_1"
outputDirectory = inputDirectory+"_output_images"
outputVideoPath = inputDirectory+"_output_video.mp4"

def sobel_algorithm(image):
    # sobel x and y filters with kernel size of 3
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

    # Apply threshold
    edges = np.zeros_like(gradient_magnitude)
    edges[(gradient_magnitude >= 100) & (gradient_magnitude <= 200)] = 255

    return edges

def region_of_interest(image):
    # Define a rectangle region of interest
    area = np.array([[(0, image.shape[0]), (0, 250), (image.shape[1], 250), (image.shape[1], image.shape[0])]])
    mask = np.zeros_like(image)

    cv2.fillPoly(mask, area, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def hough_transform(image):
    # Hough transform parameters
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi/180  # angular resolution in radians of the Hough grid
    threshold = 70  # minimum number of votes (intersections in Hough grid cell)
    minLineLength = 30  # minimum number of pixels making up a line
    maxLineGap = 20 # maximum gap in pixels between connectable line segments

    # Run Hough transform on edge detected image
    lines = cv2.HoughLinesP(image, rho, theta,threshold, minLineLength, maxLineGap)
    
    # Filter lines based on slope constraints
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
        if -1.5 <= slope <= -0.5 or 0.5 <= slope <= 1.5:
            filtered_lines.append(line)

    return filtered_lines

def get_coordinates(image, line_parameters):
    slope, intercept = line_parameters

    # Get coordinates for averaged lines
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    # Separate lines into left and right lines
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # Get averaged left and right lines
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    # Get coordinates for left and right lines
    left_line = get_coordinates(image, left_fit_average)
    right_line = get_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])

def get_intercept_point(line1, line2):
    # Get intercept point of two lines
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    # Solve for x and y
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(x_diff, y_diff)
    if div == 0:
        print('No intersection')

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return x, y

def draw_lines_and_circle(image, lines):
    line_image = np.zeros_like(image)

    # Get coordinates for lines
    A = [lines[0][0], lines[0][1]]
    B = [lines[0][2], lines[0][3]]
    C = [lines[1][0], lines[1][1]]
    D = [lines[1][2], lines[1][3]]
    coordinates = get_intercept_point((A, B), (C, D))

    # Draw lines and circle
    if lines is not None:
        cv2.line(line_image, A, (int(coordinates[0]), int(coordinates[1])), (0, 0, 255), 10)
        cv2.line(line_image, C, (int(coordinates[0]), int(coordinates[1])), (0, 0, 255), 10)
    cv2.circle(line_image, (int(coordinates[0]), int(coordinates[1])), 15, (0, 255, 255), 4)
    
    return line_image

def process_images(input_path, output_path, video_writer):
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Get a list of all BMP files in the input directory
    bmp_files = glob.glob(os.path.join(input_path, '*.bmp'))

    # Process each BMP file
    for bmp_file in bmp_files:
        # Read the image
        image = cv2.imread(bmp_file)

        # Convert to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("grayscale_image", grayscale_image)
        # cv2.waitKey(0)

        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
        # cv2.imshow("blurred_image", blurred_image)

        # Apply Sobel algorithm    
        sobel_image = sobel_algorithm(blurred_image)
        # cv2.imshow("sobel_image", sobel_image)

        # Apply region of interest
        cropped_image = region_of_interest(sobel_image)
        # cv2.imshow("cropped_image", cropped_image)

        # Apply Hough transform
        filtered_lines = hough_transform(cropped_image)

        # Get averaged lines
        averaged_lines = average_slope_intercept(image, filtered_lines)

        # Draw lines and circle
        line_image = draw_lines_and_circle(image, averaged_lines)
        # cv2.imshow("line_image", line_image)

        # Draw lines on the original image
        combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
        # cv2.imshow("combo_image", combo_image)
        
        # Write the frame to the output video
        video_writer.write(combo_image)

        file_name = os.path.splitext(os.path.basename(bmp_file))[0]

        cv2.imwrite(os.path.join(output_path, f'{file_name}_output.jpg'), combo_image)


if __name__ == "__main__":
    # Create output video 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    frame_size = (800, 600)
    video_writer = cv2.VideoWriter(outputVideoPath, fourcc, 20.0, frame_size)
    process_images(inputDirectory, outputDirectory, video_writer)

    # Release the VideoWriter object
    video_writer.release()
