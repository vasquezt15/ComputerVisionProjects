# Object Tracking with MIL, KFC, and CSRT Tracking Algorithms
## Summary and Objective
This program compares the performance of three trackers available to use on opencv on a dataset of videos and annotated frames.

- MIL: Multiple Instance Learning
- KCF: Kernelized Correlation Filters
- CSRT: Channel and Spatial Reliability Tracker

## Findings
The Channel and Spatial Reliability Tracker had the highest avg IoU score across all videos. It was also the best tracker for two of the videos. The Multiple Instance Learning algorithm had the second highest avg IoU score. While Kernelized Correlation Filters had the lowest. The Kernelized Correlation Filters algorihm lost track of a few of the objects. I also note that even though CSRT outperformed MIL on avg IoU, the bounding boxes in the truthfiles where not too accurate and qualitative analysis shows that MIL performed just as well if not better in some cases.

## Instructions
"%" denotes the command prompt

1) Start the program: % python tracking.py
2) Program takes a few minutes to run
3) Program outputs 3 files:
   1) "average_algorithm_performance.csv" contains the avg IoU values for each algorithm across all videos
   2) "output.csv" contains the average IoU score of each tracker on each video
   3) "ranked_algorithms.csv" the contains each video and a list of the trackers ranked by IoU score.
4) It will also plot a histogram with the frequency of each algorithm being the top performing