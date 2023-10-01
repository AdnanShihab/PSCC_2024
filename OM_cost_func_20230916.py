
from inv_cost_p2g import investment_cost_p2g

# Fixed parameters
cost_per_mw_pv_install = 0  # ignoring PV OM cost

# .................... BESS .........................
cost_om_mwh_bess_install_2023 = 74000  # [EUR/MWh]
cost_om_mwh_bess_install_2024 = 74000*1.0376  # [EUR/MWh] NOTE: 3.76% minimum full time salary increment is
# observed in Germany during last 8 years (including Corona and war crisis)
cost_om_mwh_bess_install_2025 = (74000*1.0376)*1.0376
cost_om_mwh_bess_install_2026 = ((74000*1.0376)*1.0376)*1.0376
cost_om_mwh_bess_install_2027 = (((74000*1.0376)*1.0376)*1.0376)*1.0376
cost_om_mwh_bess_install_2028 = ((((74000*1.0376)*1.0376)*1.0376)*1.0376)*1.0376
cost_om_mwh_bess_install_2029 = (((((74000*1.0376)*1.0376)*1.0376)*1.0376)*1.0376)*1.0376
cost_om_mwh_bess_install_2030 = ((((((74000*1.0376)*1.0376)*1.0376)*1.0376)*1.0376)*1.0376)*1.0376
cost_om_mwh_bess_install_2031 = (((((((74000*1.0376)*1.0376)*1.0376)*1.0376)*1.0376)*1.0376)*1.0376)*1.0376

# .................... CHP & HP for district heating operation .........................
cost_om_mw_chp_install_2023 = 4000  # [EUR/MW]
cost_om_mw_chp_install_2024 = 4000*1.0376  # [EUR/MW] NOTE: same as BESS mentioned above
cost_om_mw_chp_install_2025 = (4000*1.0376)*1.0376
cost_om_mw_chp_install_2026 = ((4000*1.0376)*1.0376)*1.0376
cost_om_mw_chp_install_2027 = (((4000*1.0376)*1.0376)*1.0376)*1.0376
cost_om_mw_chp_install_2028 = ((((4000*1.0376)*1.0376)*1.0376)*1.0376)*1.0376
cost_om_mw_chp_install_2029 = (((((4000*1.0376)*1.0376)*1.0376)*1.0376)*1.0376)*1.0376
cost_om_mw_chp_install_2030 = ((((((4000*1.0376)*1.0376)*1.0376)*1.0376)*1.0376)*1.0376)*1.0376
cost_om_mw_chp_install_2031 = (((((((4000*1.0376)*1.0376)*1.0376)*1.0376)*1.0376)*1.0376)*1.0376)*1.0376

# ...................CH4 import cost (as CHP input) ...............................
cost_gas_import_2023 = 200  # [/MWh]
cost_gas_import_2024 = 210  # [/MWh]
cost_gas_import_2025 = 250  # [/MWh]
cost_gas_import_2026 = 345  # [/MWh]    # considering all the time during russian war
cost_gas_import_2027 = 345  # [/MWh]
cost_gas_import_2028 = 345  # [/MWh]

# .................... HP .........................
cost_om_mw_hp_install_2023 = 10*24     # EUR/MWh
cost_om_mw_hp_install_2024 = 9.5*24     # EUR/MWh
cost_om_mw_hp_install_2025 = 9*24     # EUR/MWh
cost_om_mw_hp_install_2026 = 8.9*24     # EUR/MWh
cost_om_mw_hp_install_2027 = 7.9*24     # EUR/MWh
cost_om_mw_hp_install_2028 = 7.8*24     # EUR/MWh


class om_cost:  # OM cost

    # def __init__(self, pv_size, bess_size_mwh, x, **kwargs):
    def __init__(self, x, **kwargs):
        # self.pv_size = pv_size
        # self.bess_size_mwh = bess_size_mwh
        self.x = x

    # ................ PV ........................
    def pv_om_cost(self):
        tot_pv = sum(self.x[15:30])
        om = tot_pv * cost_per_mw_pv_install

        return om

    # ................ BESS .........................................................................
    def bess_om_cost_2023(self):
        om_cost_bess_2023 = self.x * cost_om_mwh_bess_install_2023
        return om_cost_bess_2023

    def bess_om_cost_2024(self):
        om_cost_bess_2024 = self.x * cost_om_mwh_bess_install_2024
        return om_cost_bess_2024

    def bess_om_cost_2025(self):
        om_cost_bess_2025 = self.x * cost_om_mwh_bess_install_2025
        return om_cost_bess_2025

    def bess_om_cost_2026(self):
        om_cost_bess_2026 = self.x * cost_om_mwh_bess_install_2026
        return om_cost_bess_2026

    def bess_om_cost_2027(self):
        om_cost_bess_2027 = self.x * cost_om_mwh_bess_install_2027
        return om_cost_bess_2027

    def bess_om_cost_2028(self):
        om_cost_bess_2028 = self.x * cost_om_mwh_bess_install_2028
        return om_cost_bess_2028

    # ................ CHP .............................................................................
    def chp_om_cost_2023(self):
        om_cost_ch_2023 = (self.x * cost_om_mw_chp_install_2023) + (self.x * cost_gas_import_2023)
        return om_cost_ch_2023

    def chp_om_cost_2024(self):
        om_cost_ch_2024 = self.x * cost_om_mw_chp_install_2024 + (self.x * cost_gas_import_2024)
        return om_cost_ch_2024

    def chp_om_cost_2025(self):
        om_cost_ch_2025 = self.x * cost_om_mw_chp_install_2025 + (self.x * cost_gas_import_2025)
        return om_cost_ch_2025

    def chp_om_cost_2026(self):
        om_cost_ch_2026 = self.x * cost_om_mw_chp_install_2026 + (self.x * cost_gas_import_2026)
        return om_cost_ch_2026

    def chp_om_cost_2027(self):
        om_cost_ch_2027 = self.x * cost_om_mw_chp_install_2027 + (self.x * cost_gas_import_2027)
        return om_cost_ch_2027

    def chp_om_cost_2028(self):
        om_cost_ch_2028 = self.x * cost_om_mw_chp_install_2028 + (self.x * cost_gas_import_2028)
        return om_cost_ch_2028

    # ................ HP (OM cost of the heat net is same as CHP heat net) ...............................
    def hp_om_cost_2023(self):
        om_cost_hp = (self.x * cost_om_mw_chp_install_2023) + (self.x * cost_om_mw_hp_install_2023)
        return om_cost_hp

    def hp_om_cost_2024(self):
        om_cost_hp = (self.x * cost_om_mw_chp_install_2023) + (self.x * cost_om_mw_hp_install_2024)
        return om_cost_hp

    def hp_om_cost_2025(self):
        om_cost_hp = (self.x * cost_om_mw_chp_install_2023) + (self.x * cost_om_mw_hp_install_2025)
        return om_cost_hp

    def hp_om_cost_2026(self):
        om_cost_hp = (self.x * cost_om_mw_chp_install_2023) + (self.x * cost_om_mw_hp_install_2026)
        return om_cost_hp

    def hp_om_cost_2027(self):
        om_cost_hp = (self.x * cost_om_mw_chp_install_2023) + (self.x * cost_om_mw_hp_install_2027)
        return om_cost_hp

    def hp_om_cost_2028(self):
        om_cost_hp = (self.x * cost_om_mw_chp_install_2023) + (self.x * cost_om_mw_hp_install_2028)
        return om_cost_hp

    # ................ P2G ...............................
    def p2g_om_cost_2023(self):
        inv_cost_ = investment_cost_p2g(self.x)
        inv_cost = inv_cost_.capital_cost_p2g_2023()
        om_cost = inv_cost * (2.5/100)
        return om_cost

    def p2g_om_cost_2024(self):
        inv_cost_ = investment_cost_p2g(self.x)
        inv_cost = inv_cost_.capital_cost_p2g_2024()
        om_cost = (inv_cost * (2.5/100))*1.03
        return om_cost

    def p2g_om_cost_2025(self):
        inv_cost_ = investment_cost_p2g(self.x)
        inv_cost = inv_cost_.capital_cost_p2g_2025()
        om_cost = ((inv_cost * (2.5/100))*1.03)*1.03
        return om_cost

    def p2g_om_cost_2026(self):
        inv_cost_ = investment_cost_p2g(self.x)
        inv_cost = inv_cost_.capital_cost_p2g_2026()
        om_cost = ((inv_cost * ((2.5/100))*1.03)*1.03)*1.03
        return om_cost

    def p2g_om_cost_2027(self):
        inv_cost_ = investment_cost_p2g(self.x)
        inv_cost = inv_cost_.capital_cost_p2g_2027()
        om_cost = ((inv_cost * (((2.5/100))*1.03)*1.03)*1.03)*1.03
        return om_cost

    def p2g_om_cost_2028(self):
        inv_cost_ = investment_cost_p2g(self.x)
        inv_cost = inv_cost_.capital_cost_p2g_2028()
        om_cost = ((inv_cost * ((((2.5/100))*1.03)*1.03)*1.03)*1.03)*1.03
        return om_cost








    def operation_cost_h2(self):

        opt_cost_h2 = (self.p_p2g * cost_p2g_installation) * (50/100)
        # --> Source IRENA (PhD lit P2G 2023 file)

        return opt_cost_h2

    def dispatch_cost_ch4(self):
        # gas2import = self.gas_import
        price_ch4 = self.cell2_gas2import * gas_price

        price_ch4 = price_ch4*500000     # Add penalty for gas import

        return price_ch4

    def operation_cost_syngas(self):
        x = self.syngas_m3 * 0.973      # [Kg_CO2] To produce 1 Nm3/hr SynGas (CH4) - we need 0.973 kg CO2
        # (source: OneNote/PowerTech/Calculations)
        price_co2 = x * 0.2             # [EUR] - Cost of CO2 is 0.2 cent EUR per Kg

        return price_co2

    def dispatch_cost_elec(self):    # for Power purchased from the grid = p_grid
        p_purchased = self.p_grid
        # print("power purchased =", p_purchased)
        # # print(type(p_purchased))
        # print("price elec =", self.elec_price)
        # print("elec price =", np.array(self.elec_price))
        price_elec_purchased = self.p_grid * (np.array(self.elec_price) + 11.40)    # 11.40 EUR/MWh =
        # TSO Charge/Tariff for 20kV network for buying electricity from external grid.
        # print(price_elec_purchased)
        price_elec_purchased = price_elec_purchased*500000  # just to reduce p_grid
        return price_elec_purchased