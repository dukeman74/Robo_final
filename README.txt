World modifications:
	Moved the LIDAR up 0.5 meters as suggested on piazza

nessesary libraries:
	ikpy
	numpy
	matplotlib
	scipy

makes use of regions and I recomend viewing it with regions folded when not needed
(#region viewer extention for vscode)

Running the controller:
	after running a menu will be printed with multiple options on how to proceed
	Pressing 'L' will cause the robot to load up the saved map and waypoints and 
	sequentially path to each cube, switching to teleop + IK control for the arm
	once it gets close
	
	ARM CONTROL:
		(Can also be entered fron menu by pressing 'K')
		WASD: move the goal position for the end effector (2D) relative to robot.
		QE: move the goal position for the end effector up/down.
		SPACE/m move the robot up/down via the torso section (this changes fast).
		P: move the goal position for the end effector to a preset point good for
 		dropping a cube into the basket (also good for driving around).
		U: move the goal position for the end effector to a preset point good for
		collecting a cube on the upper shelf.
		B: move the goal position for the end effector to a preset point good for
		collecting a cube on the lower shelf (for some reason IK calls take much.
 		longer down here, so be prepared when inputting WASDQE moves after this).
		O/C: open / close the gripper claws.
		ARROW KEYS: control the wheels incase this ends up being nessesary
		G: finish manual control for this cube.  Returns control to autonomous
 		planner, which will attempt to bring the bot to another cube.

	MAPPING:
		(creating a new map might cause problems with IK use backup map if waypoints arent correct)
		entered by pressing 'A' or 'M' from main menu
		if 'M' is pressed it launchs in manual mode, allowing WASD control
		if 'A' is pressed it launchs in automatic mode that drives around a map using alist of waypoints
		both methods will save frequently to map.png, which certain viewers(vscode) can view as
		it updates to make it easy to see the algorithmic progress
		press 'P' to save the current map as a numpy array, which will be loaded 
		when pressing 'L' on main menu.

	COLOR VISION TESTING:
		entered by pressing 'C' from main menu
		will take a picture and proccess it looking for a cube
		returns the estimated position as a tuple to be used for IK to direct the arm
		accuracy leaves a lot to be desired.
	DEBUGING:
		Hold L after running all methods to see print statments of our data
