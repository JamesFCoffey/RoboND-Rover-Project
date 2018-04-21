import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def obstacle_thresh(img):
    # define range of blue color in RGB
    lower= np.array([0,0,0])
    upper = np.array([110,110,110])

    # Threshold the RGB image to get only obstacles
    mask = cv2.inRange(img, lower, upper)
    return mask

def rock_thresh(img):
    # define range of blue color in RGB
    lower= np.array([130,100,0])
    upper = np.array([170,140,20])

    # Threshold the RGB image to get only rock for sampling
    mask = cv2.inRange(img, lower, upper)
    return mask

def navigable_thresh(img):
    # define range of blue color in RGB
    lower= np.array([150,150,150])
    upper = np.array([255,255,255])

    # Threshold the RGB image to get only navigable terrain
    mask = cv2.inRange(img, lower, upper)
    return mask

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped

# Define a function to convert from cartesian to polar coordinates
def to_polar_coords(xpix, ypix):
    # Calculate distance to each pixel
    dist = np.sqrt(xpix**2 + ypix**2)
    # Calculate angle using arctangent function
    angles = np.arctan2(ypix, xpix)
    return dist, angles

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # NOTE: camera image is coming to you in Rover.img

    # 1) Define source and destination points for perspective transform
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset], [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset], [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset]])

    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    obstacle_threshed = obstacle_thresh(warped)
    rock_threshed = rock_thresh(warped)
    navigable_threshed = navigable_thresh(warped)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = obstacle_threshed
    Rover.vision_image[:,:,1] = rock_threshed
    Rover.vision_image[:,:,2] = navigable_threshed

    # 5) Convert map image pixel values to rover-centric coords
    obstacle_xpix, obstacle_ypix = rover_coords(obstacle_threshed)
    rock_xpix, rock_ypix = rover_coords(rock_threshed)
    navigable_xpix, navigable_ypix = rover_coords(navigable_threshed)

    # 6) Convert rover-centric pixel values to world coordinates
    obstacle_y_world, obstacle_x_world = pix_to_world(obstacle_xpix, obstacle_ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size=200, scale=10)
    rock_y_world, rock_x_world = pix_to_world(rock_xpix, rock_ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size=200, scale=10)
    navigable_y_world, navigable_x_world = pix_to_world(navigable_xpix, navigable_ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size=200, scale=10)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    if Rover.pitch and Rover.roll < 1: # Set 1 degree thresholds near zero in roll and pitch to determine which transformed images are valid for mapping.
        Rover.worldmap[obstacle_x_world, obstacle_y_world, 0] += 1
        Rover.worldmap[rock_x_world, rock_y_world, 1] += 1
        Rover.worldmap[navigable_x_world, navigable_y_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    #Update Rover pixel distances and angles
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(navigable_xpix, navigable_ypix)

    return Rover
