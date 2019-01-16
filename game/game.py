from gurobipy import *
from functools import reduce

class Game(object):
    def __init__(self, actions, utils, mesh, dist, FPrate, discount, att_mesh,  att_type='replace'):
        self.actions = actions
        self.utils = utils # function
        self.mesh = mesh
        self.dist = dist # function
        self.FPrate = FPrate
        self.discount = discount
        self.att_mesh = att_mesh
        self.att_type = att_type

    def solve(self):

        try:
            # Create variables
            m = Model("game")
            theta = m.addVars(len(self.actions), vtype=GRB.CONTINUOUS, ub=1.0, name="theta")

            u = m.addVar(vtype=GRB.CONTINUOUS, name="obj")

            # Set objective
            m.setObjective(u, GRB.MINIMIZE)

            # Add ctr: forall a: theta_a * r_a <= Ua
            print('Building BR constraints.')

            if self.att_type == 'replace':
                m.addConstrs(
                    ((1-theta[a]) * self.utils(self.mesh[a])/(1-self.discount) <= u for a in self.actions), "BRs")
            elif self.att_type == 'add':
                m.addConstrs(
                    ((sum([self.dist(self.mesh[a] - self.att_mesh[att_a])
                        for a in self.actions]) # we use this sum (instead of 1.) to model the probability of detection = 1 outside of the distribution
                         -quicksum(
                        self.dist(self.mesh[a] - self.att_mesh[att_a])*(theta[a])
                        for a in self.actions)) * self.utils(self.att_mesh[att_a])/(1-self.discount)
                     <= u for att_a in self.actions), "BRs")


            print('Building FP constraints.')
            m.addConstr(
                quicksum(theta[a] * self.dist(self.mesh[a]) for a in self.actions) <= self.FPrate, "False-positive-rate")


            # m.write('./model.lp')

            m.optimize()

            self.thetas = [theta[i].X for i in theta]
            # for v in m.getVars():
            #     print(v.varName, v.x)

            print('Obj:', m.objVal)
            return m.objVal

        except GurobiError as e:
            print(str(e))
            print('Error reported')


class UtilityFunctions:
    @staticmethod
    def utility1(m):
        return m[0]

    def utility2(m):
        return m[1]

    def utilitySum(m):
        # print(m)
        return m.sum()

    def utilityMul(m):
        # print(m)
        return reduce((lambda x, y: x*y), m)
        # return m[0] * m[1]

    def utilityUniform(m):
        return 1.


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
