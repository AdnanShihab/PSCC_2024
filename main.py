#

import pandapower as pp
import pandapower.networks as pn

net = pn.create_cigre_network_mv(with_der="pv_wind")
print(net)
