# Template Matching for Object Tracking
## Summary
This program tracks three objects using 6 different template matching algorithms
- Template Matching Correlation Coefficient
- Template Matching Correlation Coefficient Normalized
- Template Matching Cross-Correlation
- Template Matching Cross-Correlation Normalized
- Template Matching Squared Difference
- Template Matching Squared Difference Normalized
and the users desktop camera.

## Goal
To compare and contrasts these algorithms and to analyze the challenges encountered by object tracking algorithms in general.

## Findings
Correlation Coefficient Normalized algorithm performed the best when asked to track a subject's glasses, a t-shirt logo, and a can the subject was holding. Cross Correlation performed the worst.

## Instructions
"%" denotes the command prompt

1) Start the program: % python template_matching.py
2) Repeat "3" times:
   - Take a picture by pressing "q" on the keyboard.
   - Draw a bouding box around the object to track
3) Enter the labels for the 3 objects being tracked on the command line
    - % label1
    - % label2
    - % label3
4) After enterning the last label, the program will begin tracking automatically. Press "q" on the keyboard to cycle through the trackers.
