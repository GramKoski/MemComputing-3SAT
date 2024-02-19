import math as M

L = 0.1
B = 20
A = 5
E = 0.001
Y = 0.25
S = 0.05
#learning rate (L) - weight on long term memory in capacitance
#beta (B) - rate of variation for short term variables
#alpha (A) - rate of variation of long term variables
#epsilon (E) - removes equilibrium of s == 0
#gamma (Y) - evaluates clause function for s (short-term memory)
#delta (S) - evaluates clause function for l (long-term memory)

q1 = [[-1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]]
q2 = [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[1,1,1]]
#two examples of q variable where -1 one represents the negation of literal at [m][n] and 1 represents no negation

def clause(m, v, q):
    return 1/2*min([1-q[m][0]*v[0], 1 - q[m][1]*v[1], 1 - q[m][2]*v[2]])
#clause function

def gradient(n, m, v, q):
    v1 = [[i, v[i]] for i in range(3) if i != n]
    return 1/2*q[m][n]*min(1-q[m][v1[0][0]]*v1[0][1], 1-q[m][v1[1][0]]*v1[1][1])
#negative gradient of clause

def rigidity(n, m, v, q, clause):
    if 1/2*(1 - q[m][n]*v[n]) == clause:
        return 1/2*(1 - q[m][n]*v[n])
    else:
        return 0
#rigidity

def v_dot(n, v, q, s, l):
    g, r = 0, 0
    for m in range(4):
        g += l[m]*s[m]*gradient(n, m, v, q)
        r += (1+L*l[m])*(1-s[m])*rigidity(n, m, v, q, clause(m, v, q))
    return -(g+r)
#overall change in voltage (capacitance-like)

def s_dot(m, s, clause):
    return -B*(s[m]+E)*(clause - Y)
#change in short term memory

def l_dot(clause):
    return -A*(clause - S)
#change in long term memory

def explicit(v, q, s, l, dt):
    v1 = [i for i in v]
    s1 = [i for i in s]
    l1 = [i for i in l]
    for m in range(4):
        c = clause(m, v, q)
        s1[m] += s_dot(m, s, c)*dt
        s1[m] = max(0, min(1, s1[m]))
        l1[m] += l_dot(c)*dt
        l1[m] = max(1, min(4000, l1[m]))
    for n in range(3):
        v1[n] += v_dot(n, v, q, s, l)*dt
        v1[n] = max(-1, min(v1[n], 1))
    return [v1, s1, l1]
#foward euler integration over 1 time step (dt)

def forwardEuler(v, q, s, l, dt, T):
    t = 0
    while t<T:
        x = explicit(v, q, s, l, dt)
        print(x)
        v, s, l = x[0], x[1], x[2]
        t += dt
#returns collection of all voltage an memory variables at each integration step until T

def dynamics(v, q, s, l, dt):
    t = 0
    x = [v, s, l]
    while len([m for m in range(4) if clause(m, v, q) >= 1/2]) > 0:
        x = explicit(v, q, s, l, dt)
        v, s, l = x[0], x[1], x[2]
        t += dt
    print(x)
    return x
#    return x


#forwardEuler([1, 1, 1], q2, [1, 1, 1, 1], [10, 10, 10, 10], 0.18, 50)
dynamics([-1, -1, -1], q2, [1, 1, 1, 1], [10, 10, 10, 10], 0.18)
