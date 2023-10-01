"""
P2G model
"""
# Electrolyser type: polymer electrolyte membrane eletrolysis


h = 12.46   # [Mj/m3] Collaborative optimization for a multi-energy system considering carbon capture system...
n = 0.82    # Source: Source: Book - "Power to gas: Tech and business model (page: 37)"
tau = 0.99  # 330-350 degree and 10 bar pressure <source: Endbericht-PowertoGas-eineSystemanalyse-2014; pg: 178>
# 101 kPa/1 bar is the atmospheric pressure <source: google>

# p_surplus unit = [MW]


class p2g_func:
    def __init__(self, p_input_mw, **kwargs):
        self.p_input_mw = p_input_mw

    def p2g(self):
        q_h2 = (n * self.p_input_mw)/h             # unit: meter3/s
        q_h2_day = q_h2*86400                       # unit: meter3/day

        q_h2_MWh_day = ((q_h2_day*10.55)*0.001)                 # MWh

        # q_h2_kg = (q_h2 * 1000 * 24)                 # kg/24 hr [<https://www.easyunitconverter.com/m3-to-kg>]
        # q_ch4 = (tau * q_h2)                    # unit: meter3/s
        # q_ch4 = q_ch4*3600                      # [m3/hr]
        # q_ch4_MW = (q_ch4*10.55)/1000          # 1 m3/hr = 10.55 kWh

        # print("q_h2_Nm3/s", q_h2)
        # print("q_h2_Nm3/day", q_h2_day)
        # print("q_h2_MWh", q_h2_MWh)

        return q_h2_MWh_day
