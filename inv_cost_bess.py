
cost_bess_installation_2023 = 227000     # EUR/MWh Source: MANGO --> includes battery cost + installation cost
cost_bess_installation_2024 = 222000     # EUR/MWh
cost_bess_installation_2025 = 218000     # EUR/MWh
cost_bess_installation_2026 = 208000     # EUR/MWh
cost_bess_installation_2027 = 198000     # EUR/MWh
cost_bess_installation_2028 = 180000
cost_bess_installation_2029 = 167000
cost_bess_installation_2030 = 143000
cost_bess_installation_2031 = 134000

# Capital recovery factor (CRF)
r = 0.0326  # interest rate [%]
q_bat = 5  # [years]

crf_bat = (r * (1 + r) ** q_bat) / ((1 + r) ** q_bat - 1)


class investment_cost_bess:

    def __init__(self, bess_size_mwh, **kwargs):
        # self.net = net
        # self.bus_bar = bus_bar
        # self.pv_bus = pv_bus
        # self.pv_size = pv_size
        self.bess_size_mwh = bess_size_mwh

    # ....................... BESS ....................................
    def capital_cost_bess_2023(self):
        capital_cost_bess_2023 = (self.bess_size_mwh * cost_bess_installation_2023)*crf_bat
        return capital_cost_bess_2023

    def capital_cost_bess_2024(self):
        capital_cost_bess_2024 = (self.bess_size_mwh * cost_bess_installation_2024)*crf_bat
        return capital_cost_bess_2024

    def capital_cost_bess_2025(self):
        capital_cost_bess_2025 = (self.bess_size_mwh * cost_bess_installation_2025)*crf_bat
        return capital_cost_bess_2025

    def capital_cost_bess_2026(self):
        capital_cost_bess_2026 = (self.bess_size_mwh * cost_bess_installation_2026)*crf_bat
        return capital_cost_bess_2026

    def capital_cost_bess_2027(self):
        capital_cost_bess_2027 = (self.bess_size_mwh * cost_bess_installation_2027)*crf_bat
        return capital_cost_bess_2027

    def capital_cost_bess_2028(self):
        capital_cost_bess_2028 = (self.bess_size_mwh * cost_bess_installation_2028)*crf_bat
        return capital_cost_bess_2028