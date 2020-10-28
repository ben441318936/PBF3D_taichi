import numpy as np

eps = 1e-9

def inv_kin_no_roll(position, direction_tip, l = -0.4318 + 0.4491, lp2y = 0.00584 ):
    px = position[0]
    py = position[1]
    pz = position[2]

    vx = direction_tip[0]
    vy = direction_tip[1]
    vz = direction_tip[2]

    #Range from 0 to pi
    theta1 = np.arctan(pz/px)%np.pi

    #Range from -pi to 0
    theta5   = -np.arccos(np.sin(theta1)*vx - np.cos(theta1)*vz)

    #Since theta5 and theta1 are correct, these should be correct
    sin_theta_2_4 = -vy/np.sin(theta5)
    cos_theta_2_4 = (-vx*np.cos(theta1) - vz*np.sin(theta1))/np.sin(theta5)

    #Range from -pi to 0
    theta2 = np.arctan( ( (px/np.cos(theta1)) - lp2y*cos_theta_2_4 )\
                        /(-py + lp2y*sin_theta_2_4) )%np.pi - np.pi

    theta3   = (-py + lp2y*sin_theta_2_4)/np.cos(theta2) - l

    #Range from -pi to 0
    theta4   = (np.arctan(sin_theta_2_4/cos_theta_2_4) - theta2)%np.pi- np.pi

    #Add the offsets
    q1 = theta1 - np.pi/2.0
    q2 = theta2 + np.pi/2.0
    q3 = theta3
    q4 = theta4 + np.pi/2.0
    q5 = theta5 + np.pi/2.0

    return np.array([q1, q2, q3, q4, q5])


q = inv_kin_no_roll(np.array([0.1,0.1,-0.1]), np.array([eps,eps,-1]))
print(q)