# pymoo test:

"""
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
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


problem = get_problem("zdt2")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 10),
               seed=1,
               output=MyOutput(),
               verbose=True)

"""


X = {'x0': 7, 'x1': 13, 'x2': 14, 'x3': 10, 'x4': 11, 'x5': 13, 'x6': 7, 'x7': 2, 'x8': 2, 'x9': 3, 'x10': 14, 'x11': 9, 'x12': 14, 'x13': 9, 'x14': 11, 'x15': 0.538816734003357, 'x16': 0.4191945144032948, 'x17': 0.6852195003967595, 'x18': 0.2578412832648722, 'x19': 0.8781174363909454, 'x20': 0.027387593197926163, 'x21': 0.6704675101784022, 'x22': 0.41730480236712697, 'x23': 0.5586898284457517, 'x24': 0.14038693859523377, 'x25': 0.1981014890848788, 'x26': 0.7365665693887796, 'x27': 0.9682615757193975, 'x28': 0.31342417815924284, 'x29': 0.6923226156693141, 'x30': 4.381945761480192}
print(X)

i, j = 0, 14

res = {key: val for key, val in X.items() if i <= val <= j}

print(res)

for key, val in X.items():
    print(val)



"""
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary


class MultiObjectiveMixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        vars = {
            "b": Binary(),
            "x": Choice(options=["nothing", "multiply"]),
            "y": Integer(bounds=(-2, 2)),
            "z": Real(bounds=(-5, 5)),
        }
        super().__init__(vars=vars, n_obj=2, n_ieq_constr=0, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        b, x, z, y = X["b"], X["x"], X["z"], X["y"]

        f1 = z ** 2 + y ** 2
        f2 = (z+2) ** 2 + (y-1) ** 2

        if b:
            f2 = 100 * f2

        if x == "multiply":
            f2 = 10 * f2

        out["F"] = [f1, f2]


from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize


problem = MultiObjectiveMixedVariableProblem()

algorithm = MixedVariableGA(pop_size=20, survival=RankAndCrowdingSurvival())

res = minimize(problem,
               algorithm,
               ('n_gen', 50),
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()

"""