# DAAI_OPEN_SET_ROS
Open-Set Domain Adaptation through Self-Supervision, using a ROS task this repository contains the python implementation of the final project for the course of Data Analysis &amp; Artificial Intelligence

# Using the project
To use the code in Google Colaboratory you need to follow next steps
  1. Clone this repository
  2. Open the MainBook.ipynb file
  3. Select a working directory in your Google Drive and copy the path
  4. Make sure that OfficeHome dataset* is available, you can do that bi either:
      a. Have the zipfile inside your working directory (in that case related cell will unzip the dataset)
      b. Have the dataset unzipped at './DAAI_OPEN_SET_ROS/OS_ROS_code/data/OfficeHome'
  5. run the rest of the MainBook.ipynb file**
 
*Note: If required you can find the dataset at this link: https://drive.google.com/file/d/1-q5_9fg6ugCmolG6jOXNwxMfO9axW6yj/view?usp=sharing

**Note: Last section of the code is where you can edit the parameters and configuration variables used in your run of the project
      (e.g.) --weight_RotTask_step1 1.5 --weight_RotTask_step2 1.5 --threshold -0.4 --weight_Center_Loss 0
