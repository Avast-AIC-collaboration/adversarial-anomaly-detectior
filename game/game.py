import multidict
from gurobipy import *

class Game(object):
    def __init__(self, actions, utils, mesh, dist, FPrate, discount):
        self.actions = actions
        self.utils = utils # function
        self.mesh = mesh
        self.dist = dist # function
        self.FPrate = FPrate
        self.discount = discount

    def solve(self):

        try:
            # Create variables
            m = Model("game")
            theta = m.addVars(len(self.actions), vtype=GRB.CONTINUOUS, ub=1.0, name="theta")
            u = m.addVar(vtype=GRB.CONTINUOUS, name="obj")

            # Set objective
            m.setObjective(u, GRB.MINIMIZE)

            # Add ctr: forall a: theta_a * r_a <= Ua
            m.addConstrs(
                ((1-theta[a]) * self.utils(self.mesh[a])/(1-self.discount) <= u for a in self.actions), "BRs")
            print('Best response constraints done.')

            m.addConstr(
                quicksum(theta[a] * self.dist(self.mesh[a]) for a in self.actions) <= self.FPrate, "False-positive-rate")


            m.write('./model.lp')

            m.optimize()

            self.thetas = [theta[i].X for i in theta]
            # for v in m.getVars():
            #     print(v.varName, v.x)

            print('Obj:', m.objVal)

        except GurobiError as e:
            print(str(e))
            print('Error reported')


class UtilityFunctions:
    @staticmethod
    def utility1(m):
        return m[0]


def testThisClass():
    actions = [0,1,2,3]
    def dist(x):
        r = [.4, .3, .2, .1]
        return r[x]

    def util(x):
        return x

    FPrate = 0.1
    discount = 0.0
    #
    g = Game(actions, util, dist, FPrate, discount)
    g.solve()

# testThisClass()
