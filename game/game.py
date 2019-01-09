import multidict
from gurobipy import *

class Game(object):
    def __init__(self, actions, utils, dist, FPrate, discount):
        self.actions = actions
        self.utils = utils
        self.dist = dist
        self.FPrate = FPrate
        self.discount = discount

    def solve(self):
        print(self.actions)
        print(self.utils)
        print(self.dist)

        try:
            # Create variables
            m = Model("game")
            theta = m.addVars(len(self.actions), vtype=GRB.CONTINUOUS, ub=1.0, name="theta")
            print(theta)
            u = m.addVar(vtype=GRB.CONTINUOUS, name="obj")

            # Set objective
            m.setObjective(u, GRB.MINIMIZE)

            # Add ctr: forall a: theta_a * r_a <= Ua
            m.addConstrs(
                ((1-theta[a]) * self.utils[a]/(1-discount) <= u for a in self.actions), "BRs")
            print('Best response constraints done.')

            m.addConstr(
                quicksum(theta[i] * self.dist[i] for i in self.actions) <= self.FPrate, "False-positive-rate")


            m.write('./model.lp')

            m.optimize()

            self.thetas = [theta[i].X for i in theta]
            print(self.thetas)
            # for v in m.getVars():
            #     print(v.varName, v.x)

            print('Obj:', m.objVal)

        except GurobiError as e:
            print(str(e))
            print('Error reported')

def testThisClass():
    actions = [0,1,2,3]
    dist = [.4, .3, .2, .1]
    FPrate = 0.1
    discount = 0.0
    #
    util = [i for i in actions]
    g = Game(actions, util, dist, FPrate, discount)
    g.solve()


