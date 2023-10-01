
"""
CHP
Heating network

"""

import pandas as pd
import numpy as np


# """Gas grid and heat demand"""
# index = pd.date_range("2016-01-01 00:00", "2016-01-01 23:00", freq="H")
# length = len(index)
# gas_grid = pd.Series([100]*len(index), index)
# print(gas_grid)


class CHP:
    def __init__(self, gas_input, chp_mw, **kwargs):
        self.gas_input = gas_input
        self.chp_mw = chp_mw

    # def chp_output(self):
    #
    #     p_e = self.gas_input*0.35        # electrical energy; unit: [MWh]
    #     p_q = self.gas_input*0.50        # heat energy; unit: [MWh]
    #     loss = self.gas_input*0.15      # CHP loss
    #     loss_line = 0.0339 * 11                # 33.9 kWh loss per km distance == 0.0339 MWh/km & 11 km is the
    #     # max distance from CHP to demand side
    #
    #     return [p_e, p_q, loss, loss_line]

    def chp_output_NEW(self):

        p_e = self.chp_mw*0.35*24        # electrical energy; unit: [MWh]
        p_q = self.chp_mw*0.50*24        # heat energy; unit: [MWh]
        loss_line = 0.0339 * 11                # 33.9 kWh loss per km distance == 0.0339 MWh/km & 11 km is the
        # max distance from CHP to demand side

        return [p_e, p_q, loss_line]