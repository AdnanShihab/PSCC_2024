"""
16.09.2023

Y1 2023
Y2 2024
Y3 2025
Y4 2026
Y5 2027
Y6 2028
Y7 2029
Y8 2030
Y9 2031
"""

""" Pandapower """
import pandas as pd
import numpy as np

import pandapower as pp
import pandapower.networks as pn

# Power grid:
net = pn.create_cigre_network_mv(with_der="pv_wind")
net.sgen.drop(index=net.sgen.index, inplace=True)
net.load.drop(index=net.load.index, inplace=True)
# load = net.load
# print(load)
# print(net.load["p_mw"][17])

# Gas grid and heat demand:
index = pd.date_range("2016-01-01 00:00", "2016-01-01 23:00", freq="H")
length = len(index)
gas_grid = pd.Series([300]*len(index), index)   # [MWh]
g_demand_mwh_2023 = 37.72
g_demand_mwh_2024 = 48.74
g_demand_mwh_2025 = 58.2672


# Number of time steps in years
num_time_steps = 9
num_time_steps = range(0, num_time_steps)

""" Import objects """
from investment_cost_func_20230803 import investment_cost
from power_flow_calc import power_flow_calc
from obj_func import obj_function
from demands_20230917 import energy_demand
from CHP_heat_net_model_20230918 import CHP
from heat_gas_demand_func_20230918 import g_demand
from heat_gas_demand_func_20230918 import heat_balance_calculation

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

        # ...................... PV ........................................................
        # bus-bars for the PV generators - Y1 2023
        for k in range(0, 15):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # size of PV generators  - Y1
        for k in range(15, 30):
            variables[f"x{k:01}"] = Real(bounds=(0, 1.0))  # [MW]

        # bus-bars for the PV generators - Y2 2024
        for k in range(30, 45):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # size of PV generators  - Y2
        for k in range(45, 60):
            variables[f"x{k:01}"] = Real(bounds=(0, 1.0))  # [MW]

        # bus-bars for the PV generators - Y3 2025
        for k in range(60, 75):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # size of PV generators  - Y3 2025
        for k in range(75, 90):
            variables[f"x{k:01}"] = Real(bounds=(0, 1.0))  # [MW]

        # bus-bars for the PV generators - Y4 2026
        for k in range(90, 105):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # size of PV generators  - Y4 2026
        for k in range(105, 120):
            variables[f"x{k:01}"] = Real(bounds=(0, 1.0))  # [MW]

        # bus-bars for the PV generators - Y5 2027
        for k in range(120, 135):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # size of PV generators  - Y5 2027
        for k in range(135, 150):
            variables[f"x{k:01}"] = Real(bounds=(0, 1.0))  # [MW]

        # bus-bars for the PV generators - Y6 2028
        for k in range(150, 165):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # size of PV generators  - Y6 2028
        for k in range(165, 180):
            variables[f"x{k:01}"] = Real(bounds=(0, 1.0))  # [MW]

        # bus-bars for the PV generators - Y7 2029
        for k in range(180, 195):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # size of PV generators  - Y7 2029
        for k in range(195, 210):
            variables[f"x{k:01}"] = Real(bounds=(0, 1.0))  # [MW]

        # bus-bars for the PV generators - Y8 2030
        for k in range(210, 225):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # size of PV generators  - Y8 2030
        for k in range(225, 240):
            variables[f"x{k:01}"] = Real(bounds=(0, 1.0))  # [MW]

        # bus-bars for the PV generators - Y9 2031
        for k in range(240, 255):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # size of PV generators  - Y9 2031
        for k in range(255, 270):
            variables[f"x{k:01}"] = Real(bounds=(0, 1.0))  # [MW]

        # bus-bars for the BESS - Y1 2023
        for k in range(270, 271):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))

        # ...................... BESS ........................................................
        # size of BESS  - Y1 2023
        for k in range(271, 272):
            variables[f"x{k:01}"] = Real(bounds=(0, 5.0))  # [MW]

        # size of BESS  - Y2 2024
        for k in range(272, 273):
            variables[f"x{k:01}"] = Real(bounds=(0, 5.0))  # [MW]

        # size of BESS  - Y3 2025
        for k in range(273, 274):
            variables[f"x{k:01}"] = Real(bounds=(0, 5.0))  # [MW]

        # size of BESS  - Y4 2026
        for k in range(274, 275):
            variables[f"x{k:01}"] = Real(bounds=(0, 5.0))  # [MW]

        # size of BESS  - Y5 2027
        for k in range(275, 276):
            variables[f"x{k:01}"] = Real(bounds=(0, 5.0))  # [MW]

        # size of BESS  - Y6 2028
        for k in range(276, 277):
            variables[f"x{k:01}"] = Real(bounds=(0, 5.0))  # [MW]

        # size of BESS  - Y7 2029
        for k in range(277, 278):
            variables[f"x{k:01}"] = Real(bounds=(0, 5.0))  # [MW]

        # size of BESS  - Y8 2030
        for k in range(278, 279):
            variables[f"x{k:01}"] = Real(bounds=(0, 5.0))  # [MW]

        # size of BESS  - Y9 2031
        for k in range(279, 280):
            variables[f"x{k:01}"] = Real(bounds=(0, 5.0))  # [MW]

        # ...................... CHP ........................................................
        # CHP bus-bar
        for k in range(280, 281):
            variables[f"x{k:01}"] = Integer(bounds=(2, 14))  # [MW]

        # Size of the CHP - y1 2023
        for k in range(281, 282):
            variables[f"x{k:01}"] = Real(bounds=(0, 2.0))  # [MW]

        # Size of the CHP - y2 2024
        for k in range(282, 283):
            variables[f"x{k:01}"] = Real(bounds=(0, 2.0))  # [MW]

        # Size of the CHP - y3 2025
        for k in range(283, 284):
            variables[f"x{k:01}"] = Real(bounds=(0, 2.0))  # [MW]

        # Size of the CHP - y4 2026
        for k in range(284, 285):
            variables[f"x{k:01}"] = Real(bounds=(0, 2.0))  # [MW]

        # Size of the CHP - y5 2027
        for k in range(285, 286):
            variables[f"x{k:01}"] = Real(bounds=(0, 2.0))  # [MW]

        # Size of the CHP - y6 2028
        for k in range(286, 287):
            variables[f"x{k:01}"] = Real(bounds=(0, 2.0))  # [MW]

        # Size of the CHP - y7 2029
        for k in range(287, 288):
            variables[f"x{k:01}"] = Real(bounds=(0, 2.0))  # [MW]

        # Size of the CHP - y8 2030
        for k in range(288, 289):
            variables[f"x{k:01}"] = Real(bounds=(0, 2.0))  # [MW]

        # Size of the CHP - y9 2031
        for k in range(289, 290):
            variables[f"x{k:01}"] = Real(bounds=(0, 2.0))  # [MW]

        # # CH4 supply
        # for k in range(30, 31):
        #     variables[f"x{k:01}"] = Real(bounds=(0, 5))     # [MW]

        super().__init__(vars=variables, n_obj=3, n_ieq_constr=2, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.array([x[f"x{k:01}"] for k in range(0, 290)])

        """ ***** Demands ***** """
        # OUTPUTS
        p_grid_import_2023_mwh = []
        gas_import_mwh_2023 = []

        p_grid_import_2024_mwh = []
        gas_import_mwh_2024 = []

        p_grid_import_2025_mwh = []
        gas_import_mwh_2025 = []

        p_load = energy_demand(net=net)

        for y in range(len(num_time_steps)):
            print("year =", y)

            if y == 0:      # 2023
                # ****************** CHP and district heating model ********************************************
                chp = CHP(gas_input=gas_grid[y], chp_mw=x[281])
                chp_res = chp.chp_output_NEW()  # [MWh]
                p_chp_mwh = chp_res[0]
                h_chp_mwh = chp_res[1]  # produced heat by the CHP
                h_loss_line_mwh = chp_res[2]

                # Heat balance - Gas2Import
                h_balance = heat_balance_calculation(h_chp_mwh=h_chp_mwh, h_loss_line_mwh=h_loss_line_mwh)
                heat = h_balance.heat_balance_2023()

                # ------ Outputs --------------------------------------------------
                gas_import_mwh_2023.append(heat)

                # ************************* POWER CALCULATION ***********************************************

                p_load_2023 = p_load.create_p_load_2023()
                print("p_load_2023 =", net.load["p_mw"].sum())

                power_flow_ = power_flow_calc(num_time_step=1, net=net, x_pv_size=x[15:30], x_pv_bus=x[0:15],
                                              pv_data=pv_data, bess_bus=x[270], bess_mw=x[271],
                                              chp_bus=x[280], chp_mw=x[281], p_chp_mwh=p_chp_mwh)
                add_pv = power_flow_.create_sgen()
                add_bess = power_flow_.create_bess_init()
                add_chp = power_flow_.create_chp()

                vm = power_flow_.calc_vm()

                # ..... BESS function: ........................................
                bess_calc_init = power_flow_.bess_calc()
                p_bess = bess_calc_init[1] * bess_calc_init[0]      # [MWh]

                # ..... Power balance: ................................................
                p_balance_mwh = power_flow_.calc_p_balance()
                print("p_balance_2023 =", p_balance_mwh)

                if p_balance_mwh <= 0:
                    # ------- Outputs ------------------------------------------------
                    p_grid_import_2023_mwh.append(p_balance_mwh)  # always -ve
                else:  # p_balance_mwh >= 0 (+ve)
                    # print("p_surplus [MWh] =", p_balance_mwh)
                    p_surplus_mw = p_balance_mwh / 4.5  # Converting from MWh to MW
                    # print("p_surplus_mw [MW] =", p_surplus_mw)
                    # ------- Outputs -------------------------------------------------
                    p_grid_import_2023_mwh.append(0)

                    # ***************** HEAT PUMP AND HEAT STORAGE model x[?] *************************************
                    # ***************** P2G model x[?] ********************************

                # ..... Objective funtion: ........................................................................
                idx_w = 1   # Stage = 1

                cost = obj_function(stage=idx_w, year=y, net=net, x=x)
                cost_inv_pv = cost.cost_inv_2023_pv()

                cost_inv_bess = cost.cost_inv_2023_bess()
                cost_om_bess = cost.cost_om_2023_bess()

                cost_inv_chp = cost.cost_inv_2023_chp()
                cost_om_chp = cost.cost_om_2023_chp()

                # f1_pv_bess = (cost_inv_pv_bess / ((1 + 0.05) ** (idx_w - 1))) + (cost_om_bess / ((1 + 0.05) ** y))
                f1_pv = (cost_inv_pv / ((1 + 0.05) ** (idx_w - 1)))
                f1_bess = (cost_inv_bess / ((1 + 0.05) ** (idx_w - 1))) + (cost_om_bess / ((1 + 0.05) ** y))
                f1_chp = (cost_inv_chp / ((1 + 0.05) ** (idx_w - 1))) + (cost_om_chp / ((1 + 0.05) ** y))
                f1 = f1_pv + f1_bess + f1_chp

                power_flow_.remove_PV()
                power_flow_.remove_bess()
                power_flow_.remove_load()

            elif y == 1:    # 2024

                # ************************** CHP and district heating model ******************************
                chp = CHP(gas_input=gas_grid[y], chp_mw=x[282])
                chp_res = chp.chp_output_NEW()  # [MWh]
                p_chp_mwh = chp_res[0]
                h_chp_mwh = chp_res[1]  # produced heat by the CHP
                h_loss_line_mwh = chp_res[2]

                # Heat balance
                h_balance = heat_balance_calculation(h_chp_mwh=h_chp_mwh, h_loss_line_mwh=h_loss_line_mwh)
                heat = h_balance.heat_balance_2024()

                # ------- Outputs -------
                gas_import_mwh_2024.append(heat)

                # ************************* POWER CALCULATION ***********************************************

                p_load_2024 = p_load.create_p_load_2024()
                print("p_load_2024 =", net.load["p_mw"].sum())

                power_flow_ = power_flow_calc(num_time_step=1, net=net, x_pv_size=x[45:60], x_pv_bus=x[30:45],
                                              pv_data=pv_data, bess_bus=x[270], bess_mw=x[272],
                                              chp_bus=x[280], chp_mw=x[282], p_chp_mwh=p_chp_mwh)
                add_pv = power_flow_.create_sgen()
                add_bess = power_flow_.create_bess_init()
                add_chp = power_flow_.create_chp()

                vm = power_flow_.calc_vm()

                # ..... BESS function: .....
                bess_calc_init = power_flow_.bess_calc()
                p_bess = bess_calc_init[1] * bess_calc_init[0]  # [MWh]

                # ..... Power balance: .....
                p_balance_mwh = power_flow_.calc_p_balance()
                print("p_balance_2024 =", p_balance_mwh)

                if p_balance_mwh <= 0:
                    # ------- Outputs -------
                    p_grid_import_2024_mwh.append(p_balance_mwh)       # always -ve
                else:   # p_balance_mwh <= 0
                    # print("p_surplus [MWh] =", p_balance_mwh)
                    p_surplus_mw = p_balance_mwh / 4.5  # Converting from MWh to MW
                    # print("p_surplus_mw [MW] =", p_surplus_mw)
                    # ------- Outputs -------
                    p_grid_import_2024_mwh.append(0)

                    # ***************** HEAT PUMP AND HEAT STORAGE model x[?] *************************************
                    # ***************** P2G model x[?] ********************************

                # ..... Objective funtion: .....
                idx_w = 1  # Stage = 1

                cost = obj_function(stage=idx_w, year=y, net=net, x=x)

                cost_inv_pv = cost.cost_inv_2024_pv()

                cost_inv_bess = cost.cost_inv_2024_bess()
                cost_om_bess = cost.cost_om_2024_bess()

                cost_inv_chp = cost.cost_inv_2024_chp()
                cost_om_chp = cost.cost_om_2024_chp()

                # f2 = (f2_inv / ((1 + 0.05) ** (idx_w - 1))) + (f2_om / ((1 + 0.05) ** y))
                f2_pv = (cost_inv_pv / ((1 + 0.05) ** (idx_w - 1)))
                f2_bess = (cost_inv_bess / ((1 + 0.05) ** (idx_w - 1))) + (cost_om_bess / ((1 + 0.05) ** y))
                f2_chp = (cost_inv_chp / ((1 + 0.05) ** (idx_w - 1))) + (cost_om_chp / ((1 + 0.05) ** y))
                f2 = f2_pv + f2_bess + f2_chp

                power_flow_.remove_PV()
                power_flow_.remove_bess()
                power_flow_.remove_load()

            elif y == 2:  # 2025

                # ************************** CHP and district heating model ******************************
                chp = CHP(gas_input=gas_grid[y], chp_mw=x[283])
                chp_res = chp.chp_output_NEW()  # [MWh]
                p_chp_mwh = chp_res[0]
                h_chp_mwh = chp_res[1]  # produced heat by the CHP
                h_loss_line_mwh = chp_res[2]

                # Heat balance
                h_balance = heat_balance_calculation(h_chp_mwh=h_chp_mwh, h_loss_line_mwh=h_loss_line_mwh)
                heat = h_balance.heat_balance_2025()

                # ------- Outputs -------
                gas_import_mwh_2025.append(heat)

                # ************************* POWER CALCULATION ***********************************************

                p_load.create_p_load_2025()
                print("p_load_2025", net.load["p_mw"].sum())

                power_flow_ = power_flow_calc(num_time_step=1, net=net, x_pv_size=x[75:90], x_pv_bus=x[60:75],
                                              pv_data=pv_data, bess_bus=x[270], bess_mw=x[273],
                                              chp_bus=x[280], chp_mw=x[283], p_chp_mwh=p_chp_mwh)

                add_pv = power_flow_.create_sgen()
                add_bess = power_flow_.create_bess_init()
                add_chp = power_flow_.create_chp()

                vm = power_flow_.calc_vm()

                # ..... BESS function: .....
                bess_calc_init = power_flow_.bess_calc()
                p_bess = bess_calc_init[1] * bess_calc_init[0]  # [MWh]

                # ..... Power balance: .....
                p_balance_mwh = power_flow_.calc_p_balance()
                print("p_balance 2025 =", p_balance_mwh)

                if p_balance_mwh <= 0:

                    # ------- Outputs -------
                    p_grid_import_2025_mwh.append(p_balance_mwh)       # always -ve
                else:   # p_balance_mwh <= 0
                    # print("p_surplus [MWh] =", p_balance_mwh)
                    p_surplus_mw = p_balance_mwh / 4.5  # Converting from MWh to MW
                    # print("p_surplus_mw [MW] =", p_surplus_mw)
                    # ------- Outputs -------
                    p_grid_import_2025_mwh.append(0)

                    # ***************** HEAT PUMP AND HEAT STORAGE model x[?] *************************************
                    # ***************** P2G model x[?] ********************************

                # ..... Objective funtion: .....
                idx_w = 1  # Stage = 1

                cost = obj_function(stage=idx_w, year=y, net=net, x=x)

                cost_inv_pv = cost.cost_inv_2025_pv()

                cost_inv_bess = cost.cost_inv_2025_bess()
                cost_om_bess = cost.cost_om_2025_bess()

                cost_inv_chp = cost.cost_inv_2025_chp()
                cost_om_chp = cost.cost_om_2025_chp()

                f3_pv = (cost_inv_pv / ((1 + 0.05) ** (idx_w - 1)))
                f3_bess = (cost_inv_bess / ((1 + 0.05) ** (idx_w - 1))) + (cost_om_bess / ((1 + 0.05) ** y))
                f3_chp = (cost_inv_chp / ((1 + 0.05) ** (idx_w - 1))) + (cost_om_chp / ((1 + 0.05) ** y))
                f3 = f3_pv + f3_bess + f3_chp

                power_flow_.remove_PV()
                power_flow_.remove_bess()
                power_flow_.remove_load()
            else:
                pass
        # print("p_demand =", p_demand)

        """ Power Flow """
        # power_flow_ = power_flow_calc(num_time_step=1, net=net, x_pv_size=x[15:30], x_pv_bus=x[0:15],
        #                               pv_data=pv_data, bess_bus=x[270], bess_mw=x[271])
        # add_pv = power_flow_.create_sgen()
        # add_bess = power_flow_.create_bess_init()
        # vm = power_flow_.calc_vm()
        # p_balance = power_flow_.calc_p_balance()

        # bess_calc_init = power_flow_.bess_calc()
        # bess_e = bess_calc_init[1] * bess_calc_init[0]      # [MWh]

        # print(p_balance)

        ################## p_load_2024 = p_load.create_p_load_2024()
        # How much bess Energy available?

        """ Objective functions """
        w = [1, 2, 3]
        y = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        for idx_y in y:
            # print("Y =", idx_y)
            if idx_y == 1:
                idx_w = 1

                cost = obj_function(stage=idx_w, year=idx_y, net=net, x=x)
                f1_inv = cost.year1_inv()
                f1_om = cost.year1_om()

                f1 = (f1_inv / ((1 + 0.05) ** (idx_w - 1))) + (f1_om / ((1 + 0.05) ** idx_y))
                # print("OF1 =", f1)
            elif idx_y == 2:
                idx_w = 1

                cost = obj_function(stage=idx_w, year=idx_y, net=net, x=x)
                f2_inv = cost.year2_inv()
                f2_om = cost.year2_om()

                f2 = (f2_inv / ((1 + 0.05) ** (idx_w - 1))) + (f2_om / ((1 + 0.05) ** idx_y))
                # print("OF2 =", f2)
            elif idx_y == 3:
                idx_w = 1

                cost = obj_function(stage=idx_w, year=idx_y, net=net, x=x)
                f3_inv = cost.year3_inv()
                f3_om = cost.year3_om()

                f3 = f3_inv / ((1 + 0.05) ** (idx_w - 1)) + (f3_om / ((1 + 0.05) ** idx_y))
                # print("OF3 =", f3)
            elif idx_y == 4:
                idx_w = 2

                cost = obj_function(stage=idx_w, year=idx_y, net=net, x=x)
                f4_inv = cost.year4_inv()
                f4_om = cost.year4_om()

                f4 = f4_inv / ((1 + 0.05) ** (idx_w - 1)) + (f4_om / ((1 + 0.05) ** idx_y))
                # print("OF4 =", f4)
            elif idx_y == 5:  # 2027
                idx_w = 2

                cost = obj_function(stage=idx_w, year=idx_y, net=net, x=x)
                f5_inv = cost.year5_inv()
                f5_om = cost.year5_om()

                f5 = f5_inv / ((1 + 0.05) ** (idx_w - 1)) + (f5_om / ((1 + 0.05) ** idx_y))
                # print("OF5 =", f5)
            elif idx_y == 6:  # 2028
                idx_w = 2

                cost = obj_function(stage=idx_w, year=idx_y, net=net, x=x)
                f6_inv = cost.year6_inv()
                f6_om = cost.year6_om()

                f6 = f6_inv / ((1 + 0.05) ** (idx_w - 1)) + (f6_om / ((1 + 0.05) ** idx_y))
                # print("OF5 =", f6)
            elif idx_y == 7:  # 2029
                idx_w = 3

                cost = obj_function(stage=idx_w, year=idx_y, net=net, x=x)
                f7_inv = cost.year7_inv()
                f7_om = cost.year7_om()

                f7 = f7_inv / ((1 + 0.05) ** (idx_w - 1)) + (f7_om / ((1 + 0.05) ** idx_y))
                # print("OF5 =", f7)
            elif idx_y == 8:  # 2030
                idx_w = 3

                cost = obj_function(stage=idx_w, year=idx_y, net=net, x=x)
                f8_inv = cost.year8_inv()
                f8_om = cost.year8_om()

                f8 = f8_inv / ((1 + 0.05) ** (idx_w - 1)) + (f8_om / ((1 + 0.05) ** idx_y))
                # print("OF5 =", f8)
            elif idx_y == 9:  # 2030
                idx_w = 3

                cost = obj_function(stage=idx_w, year=idx_y, net=net, x=x)
                f9_inv = cost.year9_inv()
                f9_om = cost.year9_om()

                f9 = f9_inv / ((1 + 0.05) ** (idx_w - 1)) + (f9_om / ((1 + 0.05) ** idx_y))
                # print("OF5 =", f9)
        """
        """Constraints"""

        g1 = 0.95 - vm.min()
        g2 = vm.min() - 1.05

        # g3 = 5 - x[30]   # for gas supply

        """Output"""
        out["G"] = [g1[0], g2[0]]
        # out["F"] = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
        out["F"] = [f1, f2, f3]


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
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, \
    MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from pymoo.factory import get_termination

algorithm = MixedVariableGA(pop_size=5,
                            Sampling=MixedVariableSampling(),
                            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                            element_duplicates=MixedVariableDuplicateElimination(),
                            survival=RankAndCrowdingSurvival()
                            )

termination = get_termination("n_gen", 2)

res = minimize(prob,
               algorithm,
               termination,
               seed=1,
               # output=MyOutput(),
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