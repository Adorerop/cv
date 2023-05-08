
import cv2
import numpy as np
import datetime


def point_in_poly(point, poly):
    """Checks if a point is inside a polygon"""
    x, y = point
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def process_video(input_file, output_file):
    """Processes a video to track the movement of balls across quadrants"""
    # Define the quadrants
    quad1 = [(0, 0), (250, 0), (250, 250), (0, 250)]
    quad2 = [(250, 0), (500, 0), (500, 250), (250, 250)]
    quad3 = [(0, 250), (250, 250), (250, 500), (0, 500)]
    quad4 = [(250, 250), (500, 250), (500, 500), (250, 500)]

    # Define the colors to track
    colors = {
        'red': ([0, 100, 100], [10, 255, 255]),
        'green': ([50, 100, 100], [70, 255, 255]),
        'blue': ([110, 100, 100], [130, 255, 255])
    }

    # Open the video file
    cap = cv2.VideoCapture(input_file)

    # Get the frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a video writer object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height))

    # Dictionary to store the blobs of each color in each quadrant
    color_blobs = {
        'red': [None, None, None, None],
        'green': [None, None, None, None],
        'blue': [None, None, None, None]
    }

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect the red, green, and blue balls in the frame
        red_mask = cv2.inRange(hsv, colors['red'][0], colors['red'][1])
        green_mask = cv2.inRange(hsv, colors['green'][0], colors['green'][1])
        blue_mask = cv2.inRange(hsv, colors['blue'][0], colors['blue'][1])

        # Find the contours in the masks
        red_contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over the detected contours and find the centroid of each color blob
        red_blobs = []
        for contour in red_contours:
            area = cv2.contourArea(contour)
            if area > 500:
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                red_blobs.append((cx, cy))

        green_blobs = []
        for contour in green_contours:
            area = cv2.contourArea(contour)
            if area > 500:
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                green_blobs.append((cx, cy))

        blue_blobs = []
        for contour in blue_contours:
            area = cv2.contourArea(contour)
            if area > 500:
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                blue_blobs.append((cx, cy))

        # Draw the detected color blobs on the frame
        for blob in red_blobs:
            cv2.circle(frame, blob, 10, (0, 0, 255), -1)
        for blob in green_blobs:
            cv2.circle(frame, blob, 10, (0, 255, 0), -1)
        for blob in blue_blobs:
            cv2.circle(frame, blob, 10, (255, 0, 0), -1)

        # Update the color blob dictionary for each quadrant
        for i, quad in enumerate([quad1, quad2, quad3, quad4]):
            for color in ['red', 'green', 'blue']:
                for blob in [red_blobs, green_blobs, blue_blobs][['red', 'green', 'blue'].index(color)]:
                    if blob is None:
                        continue
                    if point_in_poly(blob, quad):
                        if color_blobs[color][i] is None:
                            color_blobs[color][i] = [blob]
                        else:
                            if blob not in color_blobs[color][i]:
                                color_blobs[color][i].append(blob)
                    else:
                        if color_blobs[color][i] is not None:
                            if blob in color_blobs[color][i]:
                                color_blobs[color][i].remove(blob)

        # Draw the quadrants on the frame
        cv2.polylines(frame, [np.array(quad1)], True, (0, 0, 0), 2)
        cv2.polylines(frame, [np.array(quad2)], True, (0, 0, 0), 2)
        cv2.polylines(frame, [np.array(quad3)], True, (0, 0, 0), 2)
        cv2.polylines(frame, [np.array(quad4)], True, (0, 0, 0), 2)

        # Determine if any color blobs have entered or exited a quadrant
        for i, quad in enumerate([quad1, quad2, quad3, quad4]):
            for color in ['red', 'green', 'blue']:
                if color_blobs[color][i] is not None:
                    if color in color_counts:
                        color_counts[color].append(len(color_blobs[color][i]))
                    else:
                        color_counts[color] = [len(color_blobs[color][i])]
                    if i in quadrant_counts:
                        quadrant_counts[i].append(len(color_blobs[color][i]))
                    else:
                        quadrant_counts[i] = [len(color_blobs[color][i])]
                    for blob in color_blobs[color][i]:
                        if color not in color_tracker:
                            color_tracker[color] = {}
                        if i not in color_tracker[color]:
                            color_tracker[color][i] = []
                        if len(color_tracker[color][i]) > 0 and color_tracker[color][i][-1]['event'] == 'entry':
                            if not point_in_poly(blob, quad):
                                color_tracker[color][i][-1]['event'] = 'exit'
                                color_tracker[color][i][-1]['time'] = str(
                                    datetime.now())
                                with open('output.txt', 'a') as f:
                                    f.write(
                                        f"{color_tracker[color][i][-1]['time']}, {i+1}, {color}, exit\n")
                            else:
                                continue
                        elif len(color_tracker[color][i]) > 0 and color_tracker[color][i][-1]['event'] == 'exit':
                            if point_in_poly(blob, quad):
                                color_tracker[color][i].append({
                                    'time': str(datetime.now()),
                                    'event': 'entry'
                                })
                                with open('output.txt', 'a') as f:
                                    f.write(
                                        f"{color_tracker[color][i][-1]['time']}, {i+1}, {color}, entry\n")
                            else:
                                continue
                        elif point_in_poly(blob, quad):
                            color_tracker[color][i].append({
                                'time': str(datetime.now()),
                                'event': 'entry'
                            })
                            with open('output.txt', 'a') as f:
                                f.write(
                                    f"{color_tracker[color][i][-1]['time']}, {i+1}, {color}, entry\n")
                        else:
                            continue

        # Draw the text overlay on the frame indicating if a blob has entered or exited a quadrant
        for color, counts in color_counts.items():
            for i, count in enumerate(counts):
                if count > 0:
                    x, y, w, h = cv2.boundingRect(
                        np.array(color_blobs[color][i]))
                    if len(color_tracker[color][i]) > 0:
                        if color_tracker[color][i][-1]['event'] == 'entry':
                            text = 'Entry'
                        else:
                            text = 'Exit'
                    else:
                        text = 'Entry'
                    cv2.putText(frame, f"{color} - {text}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Write the processed frame to the output video file
        out.write(frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        # Release the video capture and output video writer objects
        cap.release()
        out.release()

        # Close all the frames
        cv2.destroyAllWindows()

        # Print the results of the ball tracking and entry/exit detection
        print('Results:')
        for color, tracker in color_tracker.items():
            for i, events in tracker.items():
                for event in events:
                    print(
                        f"{event['time']}: {color} - {event['event']} - Quadrant {i+1}")
                    if name == 'main':
                        main()
process_video("C:\Users\Adorer\Downloads\AI Assignment video.mp4","C:\Users\Adorer\Downloads\New")