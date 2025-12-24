import pybullet as p
import numpy as np
import time
import matplotlib.pyplot as plt

dt = 1 / 240
maxTime = 5.0
logTime = np.arange(0.0, maxTime, dt)
sz = len(logTime)

kp = 6.0
max_joint_vel = 8.0

def as_str(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    return str(x)

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)

robotId = p.loadURDF("./three_link.urdf.xml", useFixedBase=True)

numJoints = p.getNumJoints(robotId)

print("Joint map:")
for i in range(numJoints):
    ji = p.getJointInfo(robotId, i)
    print(
        f"{i} name={as_str(ji[1])} link(child)={as_str(ji[12])} "
        f"type={ji[2]} axis={ji[13]}"
    )

name_to_idx = {}
for i in range(numJoints):
    ji = p.getJointInfo(robotId, i)
    name_to_idx[as_str(ji[1])] = i

ctrl_joint_indices = [name_to_idx["joint_0"], name_to_idx["joint_1"], name_to_idx["joint_2"]]

eefLinkIdx = name_to_idx["joint_eef2"]

dof_joint_indices = []
for i in range(numJoints):
    ji = p.getJointInfo(robotId, i)
    jtype = ji[2]
    if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        dof_joint_indices.append(i)

numDof = len(dof_joint_indices)
print("numDof =", numDof)
print("dof_joint_indices =", dof_joint_indices)

dof_index_map = {jid: k for k, jid in enumerate(dof_joint_indices)}

ctrl_cols = [dof_index_map[j] for j in ctrl_joint_indices]

def get_dof_q_dq(robotId, dof_joint_indices):
    js = p.getJointStates(robotId, dof_joint_indices)
    q = [s[0] for s in js]
    dq = [s[1] for s in js]
    return q, dq


xd, yd, zd = 0.1, 0.0, 1.0

logX = np.zeros(sz)
logY = np.zeros(sz)
logZ = np.zeros(sz)



logXVel = np.zeros(sz)
logYVel = np.zeros(sz)
logZVel = np.zeros(sz)




q0 = [0.5, 0.5, 0.5]
p.setJointMotorControlArray(
    robotId,
    ctrl_joint_indices,
    controlMode=p.POSITION_CONTROL,
    targetPositions=q0
)
for _ in range(1000):
    p.stepSimulation()
    time.sleep(dt)




for k in range(sz):
    linkState = p.getLinkState(robotId, eefLinkIdx, computeLinkVelocity=True)

    pos = np.array(linkState[0], dtype=float)
    xSim, ySim, zSim = pos

    logX[k], logY[k], logZ[k] = xSim, ySim, zSim

    linVel = linkState[6]
    logXVel[k] = linVel[0]
    logYVel[k] = linVel[1]
    logZVel[k] = linVel[2]

    err = np.array([xSim - xd, ySim - yd, zSim - zd], dtype=float).reshape(3, 1)

    q_dof, dq_dof = get_dof_q_dq(robotId, dof_joint_indices)

    local_pos = [0.0, 0.0, 0.0]
    jac_t, jac_r = p.calculateJacobian(
        robotId,eefLinkIdx,
        local_pos,
        q_dof,
        dq_dof,
        [0.0] * numDof
    )

    Jt = np.array(jac_t, dtype=float)

    J = Jt[:, ctrl_cols]

    lam = 1e-3
    JJt = J @ J.T
    J_pinv = J.T @ np.linalg.inv(JJt + (lam * lam) * np.eye(3))

    w = (-kp) * (J_pinv @ err)
    w = np.clip(w.flatten(), -max_joint_vel, max_joint_vel)

    p.setJointMotorControlArray(
        bodyIndex=robotId,
        jointIndices=ctrl_joint_indices,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocities=w.tolist()
    )

    p.stepSimulation()
    time.sleep(dt)




plt.figure()
plt.subplot(3, 1, 1); plt.title("X"); plt.grid(True)
plt.plot(logTime, logX); plt.plot([logTime[0], logTime[-1]], [xd, xd])
plt.subplot(3, 1, 2); plt.title("Y"); plt.grid(True)
plt.plot(logTime, logY); plt.plot([logTime[0], logTime[-1]], [yd, yd])
plt.subplot(3, 1, 3); plt.title("Z"); plt.grid(True)
plt.plot(logTime, logZ); plt.plot([logTime[0], logTime[-1]], [zd, zd])

plt.figure()
plt.title("XY path")
plt.grid(True)
plt.plot(logX, logY)




plt.figure()
plt.subplot(3, 1, 1); plt.title("X Velocity"); plt.grid(True)
plt.plot(logTime, logXVel)

plt.subplot(3, 1, 2); plt.title("Y Velocity"); plt.grid(True)
plt.plot(logTime, logYVel)

plt.subplot(3, 1, 3); plt.title("Z Velocity"); plt.grid(True)
plt.plot(logTime, logZVel)

plt.show()
p.disconnect()