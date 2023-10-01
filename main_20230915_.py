# 11.08.2023

""" Pandapower """
import pandas as pd
import numpy as np

import pandapower as pp
import pandapower.networks as pn

net = pn.create_cigre_network_mv(with_der="pv_wind")
net.sgen.drop(index=net.sgen.index, inplace=True)
# load = net.load

num_time_steps = 24
num_time_steps = range(0, num_time_steps)

""" Import objects """
from investment_cost_func_20230803 import investment_cost
from power_flow_calc import power_flow_calc
from obj_func import obj_function

# pv_data = pd.read_csv("pv_power_1hr_MW_JUL_1_2015_OLDB.csv")  # Month July = 16 hours SUN
pv_data = 1

""" Pymoo """
# from pymoo.core.mixed import MixedVariableGA
# from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Binary


class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        variables = dict()

        # bus-bars for the PV generators - Y1
        for k in range(0, 15):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # size of PV generators  - Y1
        for k in range(15, 30):
            variables[f"x{k:01}"] = Real(bounds=(0, 1.0))  # [MW]

        # bus-bars for the PV generators - Y2
        for k in range(30, 45):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # size of PV generators  - Y2
        for k in range(45, 60):
            variables[f"x{k:01}"] = Real(bounds=(0, 1.0))  # [MW]

        # bus-bars for the PV generators - Y2
        for k in range(60, 75):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # size of PV generators  - Y2
        for k in range(75, 80):
            variables[f"x{k:01}"] = Real(bounds=(0, 1.0))  # [MW]

        # # CH4 supply
        # for k in range(30, 31):
        #     variables[f"x{k:01}"] = Real(bounds=(0, 5))     # [MW]

        # # Investment stages [w]
        # for k in range(31, 33):
        #     variables[f"x{k:01}"] = Integer(bounds=(1, 2))

        super().__init__(vars=variables, n_obj=3, n_ieq_constr=2, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.array([x[f"x{k:01}"] for k in range(0, 80)])

        """Power Flow"""
        # for i in num_time_steps:
        #     print("i =", i)
        # x_pv_size = x[15:30]
        # x_pv_bus =
        power_flow_ = power_flow_calc(num_time_step=num_time_steps, net=net, x_pv_size=x[15:30], x_pv_bus=x[0:15], pv_data=pv_data)
        add_pv = power_flow_.create_sgen()
        vm = power_flow_.calc_vm()

        p_balance = power_flow_.calc_p_balance()

        # print("vm =", vm)

        """ Objective functions """
        w = [1, 2, 3]
        y = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        cost_investment_s1_pv = []
        cost_investment_pv_2023 = []
        cost_investment_pv_2024 = []
        cost_investment_pv_2025 = []
        cost_investment_pv_2026 = []

        cost_investment_s2_pv = []
        cost_investment_s3_pv = []

        cost_investment_s1_bess = []
        cost_investment_s2_bess = []
        cost_investment_s3_bess = []

        # 13.09.2023
        # for idx_f2 in net.bus.index:
        #     cost_invest = investment_cost(net, bus_bar=x[0:15][idx_f2], pv_size=x[15:30][idx_f2])
        #     cost_investment = cost_invest.capital_cost()
        #     cost_investment_.append(cost_investment)
        # f1 = sum(cost_investment_)

        # 15.09.2023
        # for idx_f2 in net.bus.index:
        #     cost_invest = investment_cost(net, bus_bar=x[0:15][idx_f2], pv_size=x[15:30][idx_f2])
        #     cost_investment = cost_invest.capital_cost_pv_2023()
        #     cost_investment_s1_pv.append(cost_investment)
        # f1_s1_pv = sum(cost_investment_s1_pv)
        #
        # f1 = f1_s1_pv + 0

        for idx_w in w:     # w == stage
            print("w =", idx_w)
            # for idx_y in y:
            # for STAGE == 1
            if idx_w == 1:
                # Parameters for year = 1, 2, 3
                # for idx_f2 in net.bus.index:
                #     cost_invest = investment_cost(net, bus_bar=x[0:15][idx_f2], pv_size=x[15:30][idx_f2])
                #     cost_investment_2023 = cost_invest.capital_cost_pv_2023()
                #     cost_investment_2024 = cost_invest.capital_cost_pv_2024()
                #
                #     cost_investment_pv_2023.append(cost_investment_2023)
                #     cost_investment_pv_2024.append(cost_investment_2024)
                # res_inv_pv_2023 = sum(cost_investment_pv_2023)
                # res_inv_pv_2024 = sum(cost_investment_pv_2024)
                #
                # print(res_inv_pv_2023, res_inv_pv_2024)
                #
                # res_inv_pv_s1 = res_inv_pv_2023 + res_inv_pv_2024
                inv_cost = obj_function(stage=idx_w, year=1, net=net, x=x)
                f1 = inv_cost.year1()
                f1 = f1 / ((1+0.05)**(idx_w-1))
                print("OF1 =", f1)

                f2 = inv_cost.year2()
                f2 = f2 / ((1 + 0.05) ** (idx_w - 1))
                print("OF2 =", f2)

                f3 = inv_cost.year2()
                f3 = f3 / ((1 + 0.05) ** (idx_w - 1))
                print("OF3 =", f3)

            # elif idx == 2:
            #     # Parameters for year = 4, 5, 6
            #     for idx_f2 in net.bus.index:
            #         cost_invest = investment_cost(net, bus_bar=x[0:15][idx_f2], pv_size=x[15:30][idx_f2])
            #         cost_investment_2026 = cost_invest.capital_cost_pv_2026()
            #         # cost_investment_2024 = cost_invest.capital_cost_pv_2024()
            #
            #         cost_investment_pv_2026.append(cost_investment_2026)
            #         # cost_investment_pv_2024.append(cost_investment_2024)
            #     res_inv_pv_2026 = sum(cost_investment_pv_2026)
            #     # res_inv_pv_2024 = sum(cost_investment_pv_2024)
            #
            #     print(res_inv_pv_2026, res_inv_pv_2027)
            #
            #     res_inv_pv_s2 = res_inv_pv_2026 + res_inv_pv_2027
            #
            #     OF = res_inv_pv_s1 / ((1+0.05)**(idx-1)) + 0
            #     print("OF =", OF)
            #
            #     f1 = OF

                #     cost_investment_s1_pv.append(cost_investment)
                # f1_s1_y1_pv_ = sum(cost_investment_s1_pv)
                # f1_s1_y1_pv_sum = f1_s1_y1_pv_ / ((1+0.05)**(idx-1))
                # f1_s1_y1_pv = f1_s1_y1_pv_sum

                # for idx_f2 in net.bus.index:
                #     cost_invest = investment_cost(net, bus_bar=x[0:15][idx_f2], pv_size=x[15:30][idx_f2])
                #     cost_investment = cost_invest.capital_cost_pv_2024()
                #     cost_investment_s1_pv.append(cost_investment)
                # f1_s1_y2_pv_ = sum(cost_investment_s1_pv)
                # # f1_s1_y2_pv = f1_s1_y2_pv_
                # f1_s1_y2_pv = 0

        """Constraints"""

        # for idx, row in vm.iterrows():
        #     g1 = 0.95 - vm.loc[idx]
        #     print(g1[0])
            # print(g1)

        # for idx, row in vm.iterrows():
        #     g2 = vm.loc[idx] - 1.05

        g1 = 0.95 - vm.min()
        g2 = vm.min() - 1.05

        # print(vm)
        # print(net.sgen)

        # g3 = 5 - x[30]   # for gas supply

        """Output"""
        out["G"] = [g1[0], g2[0]]
        out["F"] = [f1, f2, f3]

        power_flow_.remove_PV()


prob = MyProblem()


# For detailed results: <https://pymoo.org/interface/display.html>
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output


class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.x_mean = Column("x_mean", width=13)
        self.x_std = Column("x_std", width=13)
        self.columns += [self.x_mean, self.x_std]

    def update(self, algorithm):
        super().update(algorithm)
        self.x_mean.set(np.mean(algorithm.pop.get("X")))
        self.x_std.set(np.std(algorithm.pop.get("X")))


from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from pymoo.factory import get_termination

algorithm = MixedVariableGA(pop_size=5,
                            Sampling=MixedVariableSampling(),
                            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                            element_duplicates=MixedVariableDuplicateElimination(),
                            survival=RankAndCrowdingSurvival()
                            )


termination = get_termination("n_gen", 5)

res = minimize(prob,
               algorithm,
               termination,
               seed=1,
               #output=MyOutput(),
               save_history=True,
               verbose=True)

X = res.X
F = res.F

print("Best solutions found: \nX = %s\nF = %s" % (X, F))

# x_df = pd.DataFrame(res.X)
# print(x_df)
#
# PV_location = x_df.loc['x0':'x14'].values    # Integer
# print(PV_location)
#
# import matplotlib.pyplot as plt
#
# xl, xu = prob.bounds()
# plt.figure(figsize=(7, 5))
#
# keys_bus = ['x0', 'x14']
# val_bus = [X[key] for key in keys_bus]
# print(val_bus)
#
# # keys_mw = ['x15', 'x29']
# # val_mw = [X[key] for key in keys_mw]
#
# # plt.scatter(X, , s=30, facecolors='none', edgecolors='r')
# plt.xlim(xl[0], xu[0])
# plt.ylim(xl[1], xu[1])
# plt.title("Design Space")
# # plt.show()