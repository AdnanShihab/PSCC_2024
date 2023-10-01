

cost_hp_installation_2023 = 550000     # EUR/MW Source: <https://www.forbes.com/home-improvement/hvac/heat-pump-installation-cost/>
cost_hp_installation_2024 = 547186     # EUR/MW
cost_hp_installation_2025 = 544373      # EUR/MW
cost_hp_installation_2026 = 532373
cost_hp_installation_2027 = 519373
cost_hp_installation_2028 = 503373
cost_hp_installation_2029 = 195273
cost_hp_installation_2030 = 189373
cost_hp_installation_2031 = 175373



# Capital recovery factor (CRF)
r = 0.0326  # interest rate [%]
q_hp = 15  # [years]       <https://glascohvac.com/heating/heat-pumps/long-heat-pump-last/#:~:text=Heat%20pumps%20normally%20last%20an,your%20heat%20pump%20is%20maintenance.>

crf_chp = (r * (1 + r) ** q_hp) / ((1 + r) ** q_hp - 1)


class investment_cost_hp:

    def __init__(self, hp_size_mwh, **kwargs):
        self.hp_size_mwh = hp_size_mwh

    # ....................... CHP ....................................
    def capital_cost_hp_2023(self):
        capital_cost_hp_2023 = (self.hp_size_mwh * cost_hp_installation_2023) * crf_chp
        return capital_cost_hp_2023

    def capital_cost_hp_2024(self):
        capital_cost_hp_2024 = (self.hp_size_mwh * cost_hp_installation_2024) * crf_chp
        return capital_cost_hp_2024

    def capital_cost_hp_2025(self):
        capital_cost_hp_2025 = (self.hp_size_mwh * cost_hp_installation_2025) * crf_chp
        return capital_cost_hp_2025

    def capital_cost_hp_2026(self):
        capital_cost_hp_2026 = (self.hp_size_mwh * cost_hp_installation_2026) * crf_chp
        return capital_cost_hp_2026

    def capital_cost_hp_2027(self):
        capital_cost_hp_2027 = (self.hp_size_mwh * cost_hp_installation_2027) * crf_chp
        return capital_cost_hp_2027

    def capital_cost_hp_2028(self):
        capital_cost_hp_2028 = (self.hp_size_mwh * cost_hp_installation_2028) * crf_chp
        return capital_cost_hp_2028