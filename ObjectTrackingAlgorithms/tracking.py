import cv2
import sys
import numpy as np
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
import numpy as np
def box_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1: [x1, y1, x2, y2]
    box2: [x1, y1, x2, y2]
    
    Returns:
    Intersection over Union (IoU) score.
    """
    # Calculate the coordinates of the intersection area
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)

    # Calculate the area of each bounding box
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the Intersection over Union (IoU)
    iou = intersection_area / union_area

    return iou
def convert_to_x1y1x2y2(bbox):
    """
    Convert bounding box coordinates from [x1, y1, width, height] to [x1, y1, x2, y2] format.
    
    Parameters:
    bbox: bounding box coordinates in [x, y, width, height] format.
    
    Returns:
    An array [x1, y1, x2, y] representing the bounding box in [x1, y1, x2, y2] format.
    """
    x, y, width, height = bbox
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + width)
    y2 = int(y + height)
    return [x1, y1, x2, y2]


if __name__ == '__main__' :
    #Trackers to compare
    tracker_types = ['MIL','KCF','CSRT']
    #set up paths to video files and annotated boxes
    video_directory = "./videos"
    video_file_names = ["DVD.mp4","race-trimmed.mp4","workout.mp4", "snowboard.mp4", "plane2.MP4" ]
    truth_file_names = ["dvd_labels.txt", "race_labels.txt", "workout.txt", "snowboard_labels.txt",  "plane_labels.txt"]
    bbox_start =[]
    output_dictionary = {"Algorithm": [], "Video": [], "IoU": [], "status":True}
    
    #Loop over the trackers
    for tracker_name in tracker_types:
        #loop over the videos and truth files
        for i in range(len(video_file_names)):
            #load the current video file
            video = cv2.VideoCapture("./videos/" + video_file_names[i])
            # Exit if video not opened.
            if not video.isOpened():
                print("Could not open video")
                sys.exit()
            
            #load the current truthfile
            truth_file = np.loadtxt("./GroundTruths/" + truth_file_names[i])[:, 1:]  # Remove the frame number
            #get the number of frames in the current video
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            #ensure that the video and the truth file have the same number of frames.
            assert(num_frames == len(truth_file)), f"Num of frames differ. Video {num_frames}, truths {len(truth_file)}"
            
            # Read first frame of the video
            ok, frame = video.read()
            if not ok:
                print('Cannot read video file')
                sys.exit()
            #The tracker will be initialized with the first bounding box in the truth file representing the objects location at t_0
            bb_box = truth_file[0]
            #convert the normalized coordinates to pixel coordinates
            bbox = (int((bb_box[0]- bb_box[2]*.5) *frame.shape[1]) , int((bb_box[1]- bb_box[3]*.5)* frame.shape[0]), int(bb_box[2]* frame.shape[1]), int(bb_box[3]* frame.shape[0]))
            #initialize two lists to track the predictions and pixel-valued truth boxes
            pred_boxes = [bbox]
            new_truth_boxes = [bbox]
            
            #select the tracker 
            if int(minor_ver) < 3:
                tracker = cv2.Tracker_create(tracker_name)
            else:
                if tracker_name == 'MIL':
                    tracker = cv2.TrackerMIL_create()
                if tracker_name == 'KCF':
                    tracker = cv2.TrackerKCF_create()
                if tracker_name == 'CSRT':
                    tracker = cv2.TrackerCSRT_create()
            # Initialize tracker with first frame and bounding box
            ok = tracker.init(frame, bbox)
            #dummy variable to index into the truth_files. Used for converting the entries in the truth file from normalized to pixel coordinates
            count = 1
            print("begin tracking")
            while True:
                # Read a new frame
                ok, frame = video.read()
                if not ok:
                    break
                #update truth box to pixel coordinates with repect to the current frame
                tbox = (int((truth_file[count][0]- truth_file[count][2]*.5) *frame.shape[1]) , int((truth_file[count][1]- truth_file[count][3]*.5)* frame.shape[0]), int(truth_file[count][2]* frame.shape[1]), int(truth_file[count][3]* frame.shape[0]))
                #save the updated truth box entry
                new_truth_boxes.append(tbox)
                # Start timer
                timer = cv2.getTickCount()
                #predict
                ok, bbox = tracker.update(frame)
                #save prediction
                pred_boxes.append(bbox)
                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
                # Draw bounding boxes
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    p11 = (int(tbox[0]), int(tbox[1]))
                    p22 = (int(tbox[0] + tbox[2]), int(tbox[1] + tbox[3]))
                    #prediction in blue
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                    #truth box in green
                    cv2.rectangle(frame, p11, p22, (0,255,0), 2, 1)
                else :
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                    output_dictionary["status"] =False
        
                # Display tracker type on frame
                cv2.putText(frame, tracker_name + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
            
                # Display FPS on frame
                cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        
                # Display result
                cv2.imshow("Tracking", frame)
                #update dummy variable
                count+=1
                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27 : break
            
            #convert the truth boxes and predicted boxes to x1y1x2y2 coordinates for IOU calculation
            pred_boxes = [convert_to_x1y1x2y2(p_box) for p_box in pred_boxes]
            new_truth_boxes = [convert_to_x1y1x2y2(t_box) for t_box in new_truth_boxes]

            #compute the IoU value for every pair of truth box and predicted box
            iou = [box_iou(t_box,p_box) for (t_box,p_box) in  zip(new_truth_boxes,pred_boxes)]
            #compute the average IoU
            avg_iou = sum(iou)/len(iou)
            print(f"Video {video_file_names[i]}\n Average IoU:", avg_iou)
            output_dictionary["Algorithm"].append(tracker_name)
            output_dictionary["Video"].append(video_file_names[i])
            output_dictionary["IoU"].append(avg_iou)
            video.release()
            cv2.destroyAllWindows()
    print(output_dictionary)
        