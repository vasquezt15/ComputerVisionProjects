import cv2
import sys
import numpy as np
import torch
from torchvision import ops
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
import numpy as np



if __name__ == '__main__' :
 
    # tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
    tracker_types = ['MIL','KCF']
    tracker_type = tracker_types[0]
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()

   #set up paths to video files and annotated boxes
    video_directory = "./videos"
    video_truths = []
    video_file_names = ["DVD.mp4","race-trimmed.mp4","workout.mp4", "snowboard.mp4", "plane2.MP4" ]
    truth_file_names = ["dvd_labels.txt", "race_labels.txt", "workout.txt", "snowboard_labels.txt",  "plane_labels.txt"]
    bbox_start =[]

    #iterate over the list of videos
    for i in range(len(video_file_names)):
        video = cv2.VideoCapture("./videos/" + video_file_names[i])
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        truth_file = np.loadtxt("./GroundTruths/" + truth_file_names[i])[:, 1:]  # Remove the frame number
        
        assert(num_frames == len(truth_file)), f"Num of frames differ. Video {num_frames}, truths {len(truth_file)}"
        bb_box = truth_file[0]
        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()
    
        # Read first frame.
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        bbox = (int((bb_box[0]- bb_box[2]*.5) *frame.shape[1]) , int((bb_box[1]- bb_box[3]*.5)* frame.shape[0]), int(bb_box[2]* frame.shape[1]), int(bb_box[3]* frame.shape[0]))
        pred_boxes = [bbox]
        new_truth_boxes = [bbox]
        # Initialize tracker with first frame and bounding box
        ok = tracker.init(frame, bbox)
        count = 1
        print("begin tracking")
        while True:
            # Read a new frame
            ok, frame = video.read()
            if not ok:
                break
            # Start timer
            timer = cv2.getTickCount()
            ok, bbox = tracker.update(frame)

            pred_boxes.append(bbox)
            #update truth box to pixel coordinates with repect to the current frame
            tbox = (int((truth_file[count][0]- truth_file[count][2]*.5) *frame.shape[1]) , int((truth_file[count][1]- truth_file[count][3]*.5)* frame.shape[0]), int(truth_file[count][2]* frame.shape[1]), int(truth_file[count][3]* frame.shape[0]))
            new_truth_boxes.append(tbox)
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    
            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                p11 = (int(tbox[0]), int(tbox[1]))
                p22 = (int(tbox[0] + tbox[2]), int(tbox[1] + tbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                cv2.rectangle(frame, p11, p22, (0,255,0), 2, 1)
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    
            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    
            # Display result
            cv2.imshow("Tracking", frame)
            count+=1
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break