

cost_chp_installation_2023 = 782000     # EUR/MW Source: MANGO
cost_chp_installation_2024 = 778000     # EUR/MW
cost_chp_installation_2025 = 774000     # EUR/MW
cost_chp_installation_2026 = 766000     # EUR/MW
cost_chp_installation_2027 = 754000     # EUR/MW
cost_chp_installation_2028 = 739000     # EUR/MW
cost_chp_installation_2029 = 724000     # EUR/MW
cost_chp_installation_2030 = 719000     # EUR/MW
cost_chp_installation_2031 = 704000     # EUR/MW


# Capital recovery factor (CRF)
r = 0.0326  # interest rate [%]
q_chp = 20  # [years]

crf_chp = (r * (1 + r) ** q_chp) / ((1 + r) ** q_chp - 1)


class investment_cost_chp:

    def __init__(self, chp_size_mwh, **kwargs):
        self.chp_size_mwh = chp_size_mwh

    # ....................... CHP ....................................
    def capital_cost_chp_2023(self):
        capital_cost_chp_2023 = (self.chp_size_mwh * cost_chp_installation_2023) * crf_chp
        return capital_cost_chp_2023

    def capital_cost_chp_2024(self):
        capital_cost_chp_2024 = (self.chp_size_mwh * cost_chp_installation_2024) * crf_chp
        return capital_cost_chp_2024

    def capital_cost_chp_2025(self):
        capital_cost_chp_2025 = (self.chp_size_mwh * cost_chp_installation_2025) * crf_chp
        return capital_cost_chp_2025

    def capital_cost_chp_2026(self):
        capital_cost_chp_2026 = (self.chp_size_mwh * cost_chp_installation_2026) * crf_chp
        return capital_cost_chp_2026

    def capital_cost_chp_2027(self):
        capital_cost_chp_2027 = (self.chp_size_mwh * cost_chp_installation_2027) * crf_chp
        return capital_cost_chp_2027

    def capital_cost_chp_2028(self):
        capital_cost_chp_2028 = (self.chp_size_mwh * cost_chp_installation_2028) * crf_chp
        return capital_cost_chp_2028