**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook).
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands.
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Media References)

[image1]: ./output/warped_example.jpg
[image2]: ./output/warped_threshed.jpg
[image3]: ./output/obstacle_threshed.jpg
[image4]: ./output/rock_threshed.jpg
[image5]: ./output/navigable_threshed.jpg
[image6]: ./output/auto_ouput.jpg
[video1]: ./output/test_mapping.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

This is the write-up.

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
The function `perspect_transform()` provided in the notebook yielded

![warped_example][image1]

The other provided function `color_thresh()` yielded

![warped_threshed][image2]

To allow for color selection of obstacles and rock samples the following functions were added to the notebook.

To select for obstacles:
```
def obstacle_thresh(img):
    # define range of blue color in RGB
    lower= np.array([0,0,0])
    upper = np.array([110,110,110])

    # Threshold the RGB image to get only obstacles
    mask = cv2.inRange(img, lower, upper)
    return mask
```
![obstacle_threshed][image3]

To select for rocks:
```
def rock_thresh(img):
    # define range of blue color in RGB
    lower= np.array([130,100,0])
    upper = np.array([170,140,20])

    # Threshold the RGB image to get only rock for sampling
    mask = cv2.inRange(img, lower, upper)
    return mask
```
![rock_threshed][image4]

To select for navigable terrain:
```
def navigable_thresh(img):
    # define range of blue color in RGB
    lower= np.array([150,150,150])
    upper = np.array([255,255,255])

    # Threshold the RGB image to get only navigable terrain
    mask = cv2.inRange(img, lower, upper)
    return mask
```
![navigable_threshed][image5]

#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.

The `process_image()` was populated a follows:
```
def process_image(img):
    # Example of how to use the Databucket() object defined above
    # to print the current x, y and yaw values
    # print(data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])

    # TODO:
    # 1) Define source and destination points for perspective transform

    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      ])

    # 2) Apply perspective transform
    warped = perspect_transform(grid_img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    obstacle_threshed = obstacle_thresh(warped)
    rock_threshed = rock_thresh(warped)
    navigable_threshed = navigable_thresh(warped)

    # 4) Convert thresholded image pixel values to rover-centric coords
    obstacle_xpix, obstacle_ypix = rover_coords(obstacle_threshed)
    rock_xpix, rock_ypix = rover_coords(rock_threshed)
    navigable_xpix, navigable_ypix = rover_coords(navigable_threshed)

    # 5) Convert rover-centric pixel values to world coords
    obstacle_y_world, obstacle_x_world = pix_to_world(obstacle_xpix, obstacle_ypix, data.xpos[data.count],
                                                      data.ypos[data.count], data.yaw[data.count], 200, 10)
    rock_y_world, rock_x_world = pix_to_world(rock_xpix, rock_ypix, data.xpos[data.count], data.ypos[data.count],
                                              data.yaw[data.count], world_size=200, scale=10)
    navigable_y_world, navigable_x_world = pix_to_world(navigable_xpix, navigable_ypix, data.xpos[data.count],
                                                        data.ypos[data.count], data.yaw[data.count], world_size=200, scale=10)


    # 6) Update worldmap (to be displayed on right side of screen)
    data.worldmap[obstacle_x_world, obstacle_y_world, 0] += 1
    data.worldmap[rock_x_world, rock_y_world, 1] += 1
    data.worldmap[navigable_x_world, navigable_y_world, 2] += 1

    # 7) Make a mosaic image, below is some example code
        # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

        # Let's create more images to add to the mosaic, first a warped image
    warped = perspect_transform(img, source, destination)
        # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
        # Flip map overlay so y-axis points upward and add to output_image
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)


        # Then putting some text over the image
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    data.count += 1 # Keep track of the index in the Databucket()

    return output_image
```

The provided `moviepy` functions yielded the following video:
![test_mapping][video1]

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

For `perception_step()`, the functions `perspect_transform()`, `obstacle_thresh()`, `rock_thresh()`, `navigable_thresh()`, `rover_coords()`, and `pix_to_world()` were added as was done in the `process_image()` in the Jupyter notebook.

Differently from the notebook, the worldmap is updated with thresholds (+/- 0.25 degrees from normal) on the pitch and roll to improve map fidelity. Also, rover-centric positions were converted to polar coordinates.

```
# 7) Update Rover worldmap (to be displayed on right side of screen)
  if Rover.pitch and Rover.roll < 0.25: # Set thresholds near zero in roll and pitch to determine which transformed images are valid for mapping.
      Rover.worldmap[obstacle_x_world, obstacle_y_world, 0] += 1
      Rover.worldmap[rock_x_world, rock_y_world, 1] += 1
      Rover.worldmap[navigable_x_world, navigable_y_world, 2] += 1
  elif Rover.pitch and Rover.roll > 395.75:
      Rover.worldmap[obstacle_x_world, obstacle_y_world, 0] += 1
      Rover.worldmap[rock_x_world, rock_y_world, 1] += 1
      Rover.worldmap[navigable_x_world, navigable_y_world, 2] += 1

  # 8) Convert rover-centric pixel positions to polar coordinates
  #Update Rover pixel distances and angles
  Rover.nav_dists, Rover.nav_angles = to_polar_coords(navigable_xpix, navigable_ypix)
```

The `to_polar_coords()` function is defined as:
```
# Define a function to convert from cartesian to polar coordinates
def to_polar_coords(xpix, ypix):
    # Calculate distance to each pixel
    dist = np.sqrt(xpix**2 + ypix**2)
    # Calculate angle using arctangent function
    angles = np.arctan2(ypix, xpix)
    return dist, angles
```
`decision.py` was not modified as it did not need to be to meet the requirement to map at least 40% of the environment at 60% fidelity and locate at least one of the rock sample.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

The simulator was run at a resolution of 1680x1050 and a graphics quality of fantastic. The FPS was 35. I achieved 66.2% mapped, 71.6% fidelity in 345.5s. The only approach I took was to increase the fidelity by putting thresholds on roll and pitch. The pipeline might fail if there were more obstacles in the way or if it had to pick up the rocks. If I were going to pursue this project further, I would optimize time, % mapped, and optimize finding all rocks. I might also add functions to avoid obstacles and pick up rocks.
![auto_ouput][image6]
