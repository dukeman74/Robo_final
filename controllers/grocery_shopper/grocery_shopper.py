"""grocery controller."""

# Nov 2, 2022
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import time
import copy
from controller import Robot, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import  image, transforms
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space
import random

#region Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()
#region webots cross ikpy setup    largely copy from docs: https://gist.github.com/ItsMichal/4a8fcb330d04f2ccba582286344dd9a7

#generate urdf

#with open("robot_urdf.urdf", "w") as file:  
#    file.write(robot.getUrdf())  #but only once
#base_elements=["base_link", "base_link_Torso_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_joint"]
base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11381"]

my_chain = Chain.from_urdf_file("robot_urdf.urdf", last_link_vector=[0.004, 0,-0.1741],base_elements=base_elements)
print(my_chain.links)

part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
            "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
            "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

for link_id in range(len(my_chain.links)):

    # This is the actual link object
    link = my_chain.links[link_id]
    
    # I've disabled "torso_lift_joint" manually as it can cause
    # the TIAGO to become unstable.
    if link.name not in part_names or link.name =="torso_lift_joint":
        print("Disabling {}".format(link.name))
        my_chain.active_links_mask[link_id] = False




timestep = int(robot.getBasicTimeStep())

motors = []
for link in my_chain.links:
    if link.name in part_names and link.name != "torso_lift_joint":
        motor = robot.getDevice(link.name)

        # Make sure to account for any motors that
        # require a different maximum velocity!
        if link.name == "torso_lift_joint":
            motor.setVelocity(0.07)
        else:
            motor.setVelocity(1)
            
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timestep)
        motors.append(motor)

def get_init():
    return([0,0,0,0] + [m.getPositionSensor().getValue() for m in motors] + [0,0,0,0])


#endregion
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.0, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Enable display
display = robot.getDevice("display")

# Odometry
pose_x     = -5
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None

map = np.zeros(shape=[360,360])
#endregion
# ------------------------------------------------------------------
# Helper Functions


gripper_status="closed"
state="menu"
res = 300
angle_threshold=math.pi/6
distance_threshold=0.2
d = None
done = False

def set_pose_via_truth():
    #get true pose data via gps compass'
    ###
    global pose_x
    global pose_y
    global pose_theta
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    
    n = compass.getValues()
    rad = ((math.atan2(n[0], n[1])))#-1.5708)
    pose_theta = rad
    ###



#region rrt

#region copy from HW2 but edit for discrete 2d problem
class Node:
    """
    Node for RRT Algorithm. This is what you'll make your graph with!
    """
    def __init__(self, pt, parent=None):
        self.point = pt # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = [] # List of points along the way from the parent node (for edge's collision checking)

def get_random_valid_vertex(state_is_valid, bounds):
    '''
    Function that samples a random n-dimensional point which is valid (i.e. collision free and within the bounds)
    :param state_valid: The state validity function that returns a boolean
    :param bounds: The world bounds to sample points from
    :return: n-Dimensional point/state
    '''
    vertex = None
    while vertex is None: # Get starting vertex
        pt = np.random.rand(bounds.shape[0]) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
        pt=np.rint(pt).astype(int)
        if state_is_valid(pt):
            vertex = pt
    return vertex

def distance_nd(p1,p2):
    dist = np.linalg.norm(p1 - p2)
    return(dist)
def get_nearest_vertex(node_list, q_point):
    '''
    Function that finds a node in node_list with closest node.point to query q_point
    :param node_list: List of Node objects
    :param q_point: n-dimensional array representing a point
    :return Node in node_list with closest node.point to query q_point
    '''
    closest=node_list[0]
    closestd=distance_nd(closest.point,q_point)
    for i in node_list:
        thisd=distance_nd(i.point,q_point)
        if(thisd<closestd):
            closest=i
            closestd=thisd
    return(closest)
    # TODO: Your Code Here
    #raise NotImplementedError

def steer(from_point, to_point, delta_q):
    '''
    :param from_point: n-Dimensional array (point) where the path to "to_point" is originating from (e.g., [1.,2.])
    :param to_point: n-Dimensional array (point) indicating destination (e.g., [0., 0.])
    :param delta_q: Max path-length to cover, possibly resulting in changes to "to_point" (e.g., 0.2)
    :return path: Array of points leading from "from_point" to "to_point" (inclusive of endpoints)  (e.g., [ [1.,2.], [1., 1.], [0., 0.] ])
    '''
    # TODO: Figure out if you can use "to_point" as-is, or if you need to move it so that it's only delta_q distance away
    if(distance_nd(from_point,to_point)>delta_q):
        direction = to_point - from_point
        unit_direction = direction / np.linalg.norm(direction)
        to_point = from_point + unit_direction * delta_q #scale to_point to end up on the circle of radius delta_q if it would be outside it
    return(np.rint(to_point).astype(int))
    # TODO Use the np.linspace function to get 10 points along the path from "from_point" to "to_point"
    path=np.linspace(from_point,to_point,10)
    path=np.rint(path).astype(int)
    #raise NotImplementedError # TODO: Delete this
    return path

def check_path_valid(path, state_is_valid):
    '''
    Function that checks if a path (or edge that is made up of waypoints) is collision free or not
    :param path: A 1D array containing a few (10 in our case) n-dimensional points along an edge
    :param state_is_valid: Function that takes an n-dimensional point and checks if it is valid
    :return: Boolean based on whether the path is collision free or not
    '''

    # TODO: Your Code Here
    for i in path:
        if(not state_is_valid(i)):
            return(False)
    return(True)
    raise NotImplementedError

def rrt(state_bounds, state_is_valid, starting_point, goal_point, k, delta_q,map):
    '''
    TODO: Implement the RRT algorithm here, making use of the provided state_is_valid function.
    RRT algorithm.
    If goal_point is set, your implementation should return once a path to the goal has been found 
    (e.g., if q_new.point is within 1e-5 distance of goal_point), using k as an upper-bound for iterations. 
    If goal_point is None, it should build a graph without a goal and terminate after k iterations.

    :param state_bounds: matrix of min/max values for each dimension (e.g., [[0,1],[0,1]] for a 2D 1m by 1m square)
    :param state_is_valid: function that maps states (N-dimensional Real vectors) to a Boolean (indicating free vs. forbidden space)
    :param starting_point: Point within state_bounds to grow the RRT from
    :param goal_point: Point within state_bounds to target with the RRT. (OPTIONAL, can be None)
    :param k: Number of points to sample
    :param delta_q: Maximum distance allowed between vertices
    :returns List of RRT graph nodes
    '''
    node_list = []
    node_list.append(Node(starting_point, parent=None)) # Add Node at starting point with no parent

    # TODO: Your code here
    directed= not goal_point is None
    for i in range(k):
        rando_goal=get_random_valid_vertex(state_is_valid,state_bounds)
        if(directed):
            if(random.random() < 0.20):
                rando_goal = goal_point
        parent=get_nearest_vertex(node_list,rando_goal)
        move_from=parent
        rando_goal=steer(move_from.point,rando_goal,delta_q)
        #print("checking ",rando_goal,move_from.point)
        if(collision_line(rando_goal,move_from.point,map)):
            yaboi=Node(rando_goal)
            yaboi.parent=parent
            yaboi.path_from_parent="lol"
            node_list.append(yaboi)
            if(directed and distance_nd(yaboi.point,goal_point)<10**-5):
                return(yaboi) #only return winning path instead
            
    # TODO: Make sure to add every node you create onto node_list, and to set node.parent and node.path_from_parent for each
    print("big time miss")
    return None
    #return(Node(goal_point))
#endregion

def path_smoothing(path,map):
    before=0
    inq=1
    after=2
    while(after<len(path)):
        if(collision_line(path[before],path[after],map)):
            #print("vertex: " + str(inq)+" has been determined to be unnessesary")
            path.pop(inq)
        else:
            before+=1
            inq+=1
            after+=1
    return(path)




def RRT_plan(map,start,goal):
    start=np.array(start,dtype=int)
    goal=np.array(goal,dtype=int)
    key = keyboard.getKey()
    if key == ord('L'):
        print("start WB: ",map_to_we(start))
        print("goal WB: ",map_to_we(goal))
        print("start: ",start)
        print("goal: ",goal)
    path = None
    bounds=np.array([[0,359],[67,292]])
    def valid(pin):
        #print("checking: ",pin)
        if(map[pin[0]][pin[1]]==0):
            return(True)
        return(False)
    while(path == None):
        path=rrt(bounds,valid,start,goal,3000,np.linalg.norm(bounds/10.),map)

    path=plot_course(path,map)
    path=path_smoothing(path,map)
    for point in path:
        map[point[0]][point[1]]=10
    key = keyboard.getKey()
    if key == ord('L'):
        plt.imshow(map.T)
        plt.show()
    return(path)

def plot_course(node_at,map):
    
    #for i in node_at:
    #    print("there was a node made at: ",i.point)
    #    map[i.point[0]][i.point[1]]=5
    #map[goal[0]][goal[1]]=10
    map=map.copy()
    points=[]
    while(node_at!=None):
        points.append(node_at.point)
        map[node_at.point[0]][node_at.point[1]]=10
        node_at=node_at.parent
    key = keyboard.getKey()
    if key == ord('L'):
        plt.imshow(map.T)
        plt.show()
    return(points)

def get_points_on_line(p1, p2):
    #print("points on line called")
    points = []
    x1, y1 = p1
    x2, y2 = p2

    # Calculate the differences between the coordinates
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Determine the direction of the coordinates
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    # Initialize the error
    error = dx - dy

    # Start at the first point
    x, y = x1, y1

    # Iterate over the points and add them to the list
    i=0
    maxi=1000
    while i<maxi:
        i+=1
        if(sx==1):
            if(x>x2):
                break
        else:
            if(x<x2):
                break
        if(sy==1):
            if(y>y2):
                break
        else:
            if(y<y2):
                break
        points.append((x, y))

        # Check if we have reached the end point
        

        #if x == x2 and y == y2:
        #    break

        # Calculate the error adjustment factors
        e2 = 2 * error
        if e2 > -dy:
            error -= dy
            x += sx
        if e2 < dx:
            error += dx
            y += sy

    return points

def collision_line(p1,p2,map):
    pts=get_points_on_line(p1,p2)
    for i in pts:
        if(map[i[0]][i[1]]!=0):
            return(False)
    return(True)

prev = (pose_x,pose_y)
def makePath(map_blocks):
    global wayP
    global state
    global prev
    value = goalPoints.pop(0)
    #print("Prev: ", prev , "Curr: ", value)
    if(prev != value):
        path = RRT_plan(map_blocks.copy(),we_to_map(prev),we_to_map(value))
    else:
        path = [[0,0]]
    prev = value
    wayP = []
    for point in path:
        wayP.append(map_to_we([point[0],point[1]]))
    wayP=wayP[::-1]
    state = 'drive'
#endregion

#region drive
def add_angles(wayp):
    betterList = []
    #print(wayP)
    betterList.append([wayP[0][0],wayP[0][1],10])
    for i in range(1,len(wayp)):
        curr=wayp[i]
        next=wayp[i-1]
        t=math.atan2(next[1]-curr[1],next[0]-curr[0])
        betterList.append([curr[0],curr[1],t])
    #print("")
    #print(betterList)
    return betterList

stall=0
prevstate=0
momentum=0
initialTurn = 0
stuck = 0
def drive_to_points(wayp):
    angle_epsilon=.2
    distance_epsilon=.1
    global pose_x
    global pose_theta
    global pose_y
    global state
    global stall
    global prevstate
    global momentum
    global state
    global initialTurn
    global firstRun
    rOffset = 0
    lOffset = 0
    global stuck 
    flip = 1
    #print("waypoints:                               ", wayp)
    if(len(wayp) == 1):
            state = 'cube'
            firstRun = 0
            prevstate=0
            initialTurn = 0
            return((0,0))

    key = keyboard.getKey()
    if(stall>0):
        stall-=1
        return(0.00001,0.00001)
    #print("Pose X :", pose_x, " Pose Y: ",pose_y," W x: ", wayp[1][0], " Y w : ", wayp[1][1])
    needed_angle=math.atan2((wayp[1][1]- pose_y),(wayp[1][0]-pose_x))

    #needed_angle =  ((needed_angle + math.pi ) % (2*math.pi)) - math.pi
    #needed_angle=-math.atan2(wayp[0][1]-wayp[1][1],wayp[0][0]-wayp[1][0])
    stuck += 1

    if key == ord('L'):
        print("desired angle: " ,needed_angle, " My angle ",pose_theta,"Desired pos: ",wayp[1], "My pos: ", (pose_x,pose_y))
    if(abs(needed_angle - pose_theta) > math.pi):
        if(needed_angle > (math.pi/2)):
            needFake = needed_angle
            poseFake = 2*math.pi - abs(pose_theta)
        else:
            poseFake = pose_theta
            needFake = 2*math.pi - abs(needed_angle)
    else:
        needFake = needed_angle
        poseFake = pose_theta

    dist=math.sqrt((pose_x-wayp[1][0])**2+(pose_y-wayp[1][1])**2)  
    if(abs(needFake-poseFake)<angle_epsilon or initialTurn == 1 or (dist<distance_epsilon)):
        stuck = 0
        if(prevstate==0):
            initialTurn = 1
            key = keyboard.getKey()
            if key == ord('L'):
                 print("reached the correct angle for this waypoint                                x")
            prevstate=1
            stall=100
            return(0.1,-0.1)
        #print("desired angle: " ,needed_angle, " My angle ",pose_theta,"Desired pos: ",wayp[1], "My pos: ", (pose_x,pose_y))


        if(needFake<poseFake):
            if key == ord('L'):
                print("fixing right")
            lOffset = .15
            rOffset = 0
        else:
            if key == ord('L'):
                print("fixing left")
            lOffset = 0
            rOffset = .15
            
        if(dist<distance_epsilon):
            key = keyboard.getKey()
            if key == ord('L'):
                print("reached the correct location for this waypoint")
            wayp.pop(0)
            if(len(wayp) == 1):
                state = 'cube'
                firstRun = 0
            prevstate=0
            initialTurn = 0
            return((0,0))
        momentum+=0.001
        if(momentum>2):
            momentum=2
        if(momentum<0.01):
            return((momentum,-momentum))
        return((momentum + (lOffset*momentum),momentum + (rOffset*momentum)))
    momentum=0
    prevstate=0
    if(stuck > 1000):
        flip = -1
        print("stuck turing around")
    if(needFake<poseFake):
        return((.5,-.5))
    else:
        return((-.5,.5))
   

#endregion
#region mapping functions

def map_print():
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta #- np.pi/2.
        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy =  math.sin(t)*rx + math.cos(t)*ry + pose_y

        ################ ^ [End] Do not modify ^ ##################

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))
            # Part 1.3: visualize map gray values.
 
            # You will eventually REPLACE the following lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
        map_point=we_to_map((wx,wy))
        x = map_point[0]
        y = map_point[1]
        map[x][y] += 0.005
        val = map[x][y]
        if( val > 1):
            map[x][y] = 1
            val = 1

        # print(val)
        val2 =  int(colorize(val))
        # print(hex(val2))
        display.setColor(val2)
        #print("wx: ",wx,"wy: ",wy)
        display.drawPixel(x,y)


    

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    map_point=we_to_map((pose_x,pose_y))
    display.drawPixel(map_point[0],map_point[1])
    display.imageSave(None,"map.png") 

def we_to_map(point):
    #webots top left = (16,10)
    #webots bottom right = (-16,-10)
    #map top left = (0,0)
    #map bottom right = (360,360)

    wx=point[0] #(16,-16)
    wy=point[1] #(10,-10)
    rx=-wx #(-16,16)
    ry=-wy #(-10,10)
    rx+=16 #(0,32)
    ry+=10 #(0,20)
    scale=360./32. #to scale the y axis up to the map size
    rx*=scale #(0,360)
    ry*=scale #(0,225)
    ry+=67 #(67,292) to place this slice in the middle of the view
    #lastly the axis are flipped of the sensical way to view
    t=rx
    rx=ry
    ry=t
    #needs to be integers
    ry=round(ry)
    rx=round(rx)
    return((rx,ry))

def map_to_we(point):
    #webots top left = (16,10)
    #webots bottom right = (-16,-10)
    #map top left = (0,0)
    #map bottom right = (360,360)
    rx=point[0] 
    ry=point[1]
    t=rx
    rx=ry
    ry=t
    ry-=67
    scale=360./32. #to scale the y axis up to the map size
    rx/=scale #(0,360)
    ry/=scale #(0,225)
    rx-=16 #(0,32)
    ry-=10 #(0,20)
    wx=-rx #(-16,16)
    wy=-ry 
    return((wx,wy))

def colorize(g):
    if(g> 1 or g < 0):
       #print("threek")
        return(0xF535AA)
    g=int(g*255)
    out=((g<<16)+(g<<8)+g)
    return(out)

#endregion
firstRun = 0
valueRot = 0
def rotate():
    global orintation
    global pose_theta
    global firstRun
    global valueRot
    angle_epsilon=.05
    if(firstRun == 0):
        valueRot = orintation.pop(0)
    if(valueRot == 0):
        needed_angle = math.pi/2
    else:
        needed_angle = (3*math.pi)/2
    firstRun = 1
    
    if(abs(needed_angle - pose_theta) > math.pi):
        if(needed_angle > (math.pi/2)):
            needFake = needed_angle
            poseFake = 2*math.pi - abs(pose_theta)
        else:
            poseFake = pose_theta
            needFake = 2*math.pi - abs(needed_angle)
    else:
        needFake = needed_angle
        poseFake = pose_theta
    

    if(abs(needFake-poseFake)<angle_epsilon):
        return((0,0,True))
    if(needFake<poseFake):
        return((.5,-.5,False))
    else:
        return((-.5,.5,False))
def open_claw():
    robot_parts["gripper_left_finger_joint"].setPosition(0.045)
    robot_parts["gripper_right_finger_joint"].setPosition(0.045)

def close_claw():
    robot_parts["gripper_left_finger_joint"].setPosition(0)
    robot_parts["gripper_right_finger_joint"].setPosition(0)




    
#region camera/computer vision/color blobs   
def cam():
    cameraData = camera.getImage()
    imageNP = np.frombuffer(cameraData, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    imageNP = imageNP[..., [2,1,0,3]]
    img = Image.fromarray(imageNP, "RGBA")
    img.show()
    size = 0
    #img.save("opengenus_image.jpeg")
    mask = np.copy(imageNP)
    yellows=[]
    key = keyboard.getKey()
    for i1, y in enumerate(imageNP):
        for i2, x in enumerate(y):
            #if(i2 % 40 == 0):
                #print(x[0],x[1],x[2])
            if(len(goalPoints) <= 2):
                if(x[0] < 40 and x[1] > 0xD0 and x[2] < 40):
                    if key == ord('L'):
                        print("I found Green at pixel: ", i2,i1 )
                    yellows.append([i2,i1])
                    size+=1
                else:
                    mask[i1][i2] = [0,0,0,255]
            else:
                if(x[0] > 0xD0 and x[1] > 0xD0 and x[2] < 40):
                    if key == ord('L'):
                        print("I found yellow at pixel: ", i2,i1 )
                    yellows.append([i2,i1])
                    size+=1
                else:
                    mask[i1][i2] = [0,0,0,255]
    block=None
    img2 = Image.fromarray(mask, "RGBA")
    img2.show()
    if(size != 0):
        block=camDist(yellows,size)
    return(block) 
def get_centre(point_list):
    total=[0,0]
    for point in point_list:
        total[0]+=point[0]
        total[1]+=point[1]
    total[0]=total[0]/len(point_list)
    total[1]=total[1]/len(point_list)
    return(total)

def camDist(pointsin,size):
    distance = (.07/2)/(math.tan(math.sqrt(size)/240)) #assuming we are seeing the block head on, and how many pixels its taking up compared with the fact that we know its 0.07 m 
    centroid=get_centre(pointsin)
    key = keyboard.getKey()
    if key == ord('L'):
        print("I spy a blonk")
        print("centroid: ",centroid)
        print("distance: ",distance)
    height=135
    width=240
    Hfov=2
    Vfov=2 * math.atan(math.tan(Hfov * 0.5) * (height / width))   #webots documentation: vertical FOV = 2 * atan(tan(fieldOfView * 0.5) * (height / width))
    hoff=centroid[0]/width
    voff=centroid[1]/height
    hangle=(hoff-.5)*Hfov
    vangle=(voff-.5)*Vfov
    #get camera pose
    #one time camera absolute pos = (-2.94275,2.82565,1.32287)
    #one time compass gps pos = (-2.95534,2.80125,0.0872982)
    cube_distance_infront=distance*math.cos(hangle)*math.cos(vangle)
    cube_distance_right=distance*math.sin(hangle)*math.cos(vangle)
    cube_distance_up=distance*math.cos(hangle)*math.sin(vangle)
    return((cube_distance_infront,cube_distance_right,cube_distance_up))

#endregion
def menu_swap():
    global state
    state="menu"
    print("----------------------")
    print("L -> load map from file and Use IK to move to cubes")
    print("M -> enter mapping mode(WASD to move, P to save map)")
    print("C -> measure for a cube in front of the bot")
    print("G -> test IK")
# Main Loop
robot_parts["wheel_left_joint"].setVelocity(0)
robot_parts["wheel_right_joint"].setVelocity(0)
# coords
#orintaiont right = 1
# topshelf = 0 mid = 1
#(3.46,6.45),(3.51,2.91),(2.62,3.02),(.81,1.08),(-2.70,1.1),(-2.70,1.1),(-3.22,2.70),(2.277,-2.967),(4.36,-2.97),(-1.54,-4.965),(-.96,1.005),(5.71,6.57)
#0,0,0,1,1,1,0,1,1,0,1,0
#0,0,1,1,1,0,0,1,0,1,1,0
menu_swap()
goalPoints = [(3.46,6.45),(3.51,2.91),(2.62,3.02),(.81,1.08),(-2.70,1.1),(-2.70,1.1),(-3.22,2.70),(2.277,-2.967),(4.36,-2.97),(-1.54,-4.965),(-.96,1.005),(5.71,6.57)]
orintation = [(3.46,6.45),(3.51,2.91),(2.62,3.02),(.81,1.08),(-2.70,1.1),(-2.70,1.1),(-3.22,2.70),(2.277,-2.967),(4.36,-2.97),(-1.54,-4.965),(-.96,1.005),(5.71,6.57)]
shelfLevArr = [0,0,1,1,1,0,0,1,0,1,1,0]

clawstate=True
initial_position=None
torso_moment=0
target = [.56,-.27,1.4]
ikResults = None
while robot.step(timestep) != -1:

    set_pose_via_truth()
    if(state=="menu"):

        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == ord('L'):  #load previous map
            print("Map loaded")
            map = np.load("map.npy")
            key = keyboard.getKey()
            if key == ord('L'):
                plt.imshow(map.T)
                plt.show()
            n = 15
            kernel = np.ones((n, n))
            map_blocks = convolve2d(map, kernel, mode='same')
            map_blocks[map_blocks > 0] = 1
            key = keyboard.getKey()
            if key == ord('L'):
                plt.imshow(map_blocks.T)
                plt.show()
            """ map_pt_list = []
            for i in goalPoints:
                map_pt_list.append(we_to_map(i))
            print(goalPoints)
            for point in map_pt_list:
            
            if(map_blocks[point[0]][point[1]]!=0):
                print("throw")
                print(point)
                print(map_to_we(point))
            if(map_blocks[we_to_map((-3.22,2.70))[0]][we_to_map((-3.22,2.70))[1]]!=0):
                print("throw")
            else:
                print("dub")
            """
                
            makePath(map_blocks)
            #wayP = add_angles(wayP)           
        elif key == ord('M'):   #enter mapping mode
            state="map"
            print("now in mapping mode")
        elif key == ord('C'):   #enter cube mode
            state="cube"
            print("now in cube mode")
        elif key == ord('K'):   #enter IK testing
            state="IK"
            target = [.56,-.27,1.4]
            print("now in IK mode")
            ikResults=None
    if(state=="map"):
        #print("I am at: (" +str(pose_x)+ ", "+str(pose_y) +", " +str(pose_theta)+")")
        map_print()
        key = keyboard.getKey()
        if key == ord('P'): #save the map
            np.save("map.npy",(map>.55)*1)
            display.imageSave(None,"map.png") 
            print("Map file saved")
        v=[0,0]
        spd=6
        if(key==ord("W")):
            v[0]=spd
            v[1]=spd
        if(key==ord("A")):
            v[0]=-spd
            v[1]=spd
        if(key==ord("D")):
            v[0]=spd
            v[1]=-spd
        if(key==ord("S")):
            v[0]=-spd
            v[1]=-spd
        robot_parts["wheel_left_joint"].setVelocity(v[0])
        robot_parts["wheel_right_joint"].setVelocity(v[1])
    if(state=="drive"):
        v = drive_to_points(wayP)
        stallShelf = 300

        key = keyboard.getKey()
        if(key == ord('L')):
            print("Wheel Speed: ", v)
        robot_parts["wheel_left_joint"].setVelocity(v[0])
        robot_parts["wheel_right_joint"].setVelocity(v[1])
    if(state=="cube"):

        rot = rotate()
        robot_parts["wheel_left_joint"].setVelocity(rot[0])
        robot_parts["wheel_right_joint"].setVelocity(rot[1])
        if(rot[2] == True):
            if(stallShelf == 300):
                shelfLev = shelfLevArr.pop(0)
            if(shelfLev == 1 ):
                robot_parts["wheel_left_joint"].setVelocity(-1)
                robot_parts["wheel_right_joint"].setVelocity(-1)
                stallShelf-= 1
            else:
                stallShelf = 0   
            if(stallShelf <= 0):
                block=cam()
                state = "IK"
                print("Move arm using IK")

    if(state=="IK"):
        key = keyboard.getKey()
        if(key == ord('G')):
            if(len(goalPoints) == 0):
                print("Yay all cubes have been grabed, returning to menu")
                menu_swap()
            else:
                makePath(map_blocks)
        if(key == ord('C')): #close gripper
            clawstate=False
        if(key == ord('O')): #oper gripper
            clawstate=True
        if(key == ord('P')):#put gripper right above basket to setup a drop in
            target=[0.1,0,1]
            ikResults=None
        if(key == ord('U')):#grabbing a block on top shelf
            target = [.56,-.27,1.4]
            ikResults=None
        if(key == ord('B')):#grabbing a block on bottom shelf
            target = [.56,-.27,.7]
            ikResults=None

        mod=0.01
        if(key == ord('W')):
            target=[target[0]+mod,target[1],target[2]]
            ikResults=None
        if(key == ord('S')):
            target=[target[0]-mod,target[1],target[2]]
            ikResults=None
        if(key == ord('A')):
            target=[target[0],target[1]+mod,target[2]]
            ikResults=None
        if(key == ord('D')):
            target=[target[0],target[1]-mod,target[2]]
            ikResults=None
        if(key == ord('Q')):
            target=[target[0],target[1],target[2]-mod]
            ikResults=None
        if(key == ord('E')):
            target=[target[0],target[1],target[2]+mod]
            ikResults=None
        v=[0,0]
        spd=1
        if key == keyboard.UP:
            v[0]=spd
            v[1]=spd
        if key == keyboard.LEFT:
            v[0]=-spd
            v[1]=spd
        if key == keyboard.RIGHT:
            v[0]=spd
            v[1]=-spd
        if key == keyboard.DOWN:
            v[0]=-spd
            v[1]=-spd
        if key == ord(' '):
            torso_moment+=mod
        if key == ord('M'):
            torso_moment-=mod
        if(torso_moment<0):
            torso_moment=0
            
        robot_parts["wheel_left_joint"].setVelocity(v[0])
        robot_parts["wheel_right_joint"].setVelocity(v[1])
        if(clawstate):
            open_claw()
        else:
            close_claw()
        
        offset_target = target#[-(target[2])+0.22, -target[0]+0.08, (target[1])+0.97+0.2]
        if(ikResults is None):
            if(initial_position is None):
                initial_position=get_init()
            ikResults = my_chain.inverse_kinematics(offset_target, initial_position=initial_position,  target_orientation = [0,0,1], orientation_mode="Y")
            #ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')

            # Plot the current position of your arm
            #my_chain.plot(initial_position, ax, target=offset_target)

            # And plot the target position of your arm
            #my_chain.plot(ikResults, ax, target=offset_target)
            #matplotlib.pyplot.show()
        if(key == ord('E')):
            print("target point: ",target,"torso height: ",torso_moment)
        printing=False
        for res in range(len(ikResults)):
            # This if check will ignore anything that isn't controllable
            if my_chain.links[res].name in part_names:
                robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
                if(printing):
                    print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))
        if(clawstate):
            goal=0.045
        else:
            goal=0
        robot_parts["gripper_right_finger_joint"].setPosition(goal)
        robot_parts["torso_lift_joint"].setPosition(torso_moment)   
"""
    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            gripper_status="closed"
    else:
        # Open gripper
        #robot_parts["wheel_left_joint"].setVelocity(0)
        #robot_parts["wheel_right_joint"].setVelocity(0)
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            gripper_status="open"  """
