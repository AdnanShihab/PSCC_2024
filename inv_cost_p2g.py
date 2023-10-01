
# AEC type
cost_p2g_installation_2023 = 1520000
cost_p2g_installation_2024 = 1467000
cost_p2g_installation_2025 = 1430000
cost_p2g_installation_2026 = 1330000
cost_p2g_installation_2027 = 1350000
cost_p2g_installation_2028 = 1300000

# Source:
# file:///C:/Users/shihab/Dropbox/Documents/Uni%20Bremen/PhD%20literature/PSCC_2024/p2g_price%20and%20lifetime.pdf

# Capital recovery factor (CRF)
r = 0.0326  # interest rate [%]
q_p2g = 10  # [years]
# source:
# file:///C:/Users/shihab/Dropbox/Documents/Uni%20Bremen/PhD%20literature/PSCC_2024/p2g_price%20and%20lifetime.pdf

crf_p2g = (r * (1 + r) ** q_p2g) / ((1 + r) ** q_p2g - 1)


class investment_cost_p2g:
    def __init__(self, p2g_size_mw, **kwargs):
        self.p2g_size_mw = p2g_size_mw

    def capital_cost_p2g_2023(self):
        capital_cost = (self.p2g_size_mw * cost_p2g_installation_2023) * crf_p2g
        return capital_cost

    def capital_cost_p2g_2024(self):
        capital_cost = (self.p2g_size_mw * cost_p2g_installation_2024) * crf_p2g
        return capital_cost

    def capital_cost_p2g_2025(self):
        capital_cost = (self.p2g_size_mw * cost_p2g_installation_2025) * crf_p2g
        return capital_cost

    def capital_cost_p2g_2026(self):
        capital_cost = (self.p2g_size_mw * cost_p2g_installation_2026) * crf_p2g
        return capital_cost

    def capital_cost_p2g_2027(self):
        capital_cost = (self.p2g_size_mw * cost_p2g_installation_2027) * crf_p2g
        return capital_cost

    def capital_cost_p2g_2028(self):
        capital_cost = (self.p2g_size_mw * cost_p2g_installation_2028) * crf_p2g
        return capital_cost