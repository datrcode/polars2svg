import unittest
import polars as pl
from polars2svg import Polars2SVG

class Testxyp_order(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_list(self):
        for _lazy_ in [True, False]:
            df = pl.DataFrame({'qty':[1,2,3,10,4], 'pet':['cat','dog','parakeet','goldfish','ferret']})
            _params_ = {'df':df, 'x':'pet', 'y':'qty', 'color':'pet', 'dot_size':10}
            _params_['use_lazy_execution'] = _lazy_
            _xyp0_  = self.p2s.xyp(**_params_)                                                                    # default
            _xyp1_  = self.p2s.xyp(**_params_, x_order=['goldfish', 'ferret', 'parakeet', 'dog', 'cat'])          # complete order (decreasing qtys)
            _xyp2_  = self.p2s.xyp(**_params_, x_order=['goldfish', 'ferret', 'parakeet', 'dog', 'cat', 'snake']) # complete order + extra (decreasing qtys)
            _xyp3_  = self.p2s.xyp(**_params_, x_order=['goldfish', 'ferret'])                                    # incomplete order
            _xyp4_  = self.p2s.xyp(**_params_, x_order=['goldfish', 'ferret', 'snake'])                           # incomplete order + extra

    def test_listTuple(self):
        for _lazy_ in [True, False]:
            df = pl.DataFrame({'qty':  [2,        8,         5,          15],
                               'type': ['cat',    'cat',     'dog',      'goldfish'], 
                               'color':['gray',   'orange',  'spotted',  'orange']})
            _params_ = {'df':df, 'x':('type','color'), 'y':'qty', 'color':'color', 'dot_size':10}
            _params_['use_lazy_execution'] = _lazy_
            _xyp0_   = self.p2s.xyp(**_params_)
            _xyp1_   = self.p2s.xyp(**_params_, x_order=[('goldfish','orange'), ('cat','orange'), ('dog','spotted'), ('cat','gray')])                     # complete
            _xyp2_   = self.p2s.xyp(**_params_, x_order=[('goldfish','orange'), ('cat','orange'), ('dog','spotted'), ('cat','gray'), ('snake','albino')]) # complete + extra
            _xyp3_   = self.p2s.xyp(**_params_, x_order=[('goldfish','orange'), ('cat','gray')])                                                          # incomplete
            _xyp4_   = self.p2s.xyp(**_params_, x_order=[('goldfish','orange'), ('cat','gray'), ('snake', 'albino')])                                     # incomplete + extra

    def test_dict(self):
        for _lazy_ in [True, False]:
            df = pl.DataFrame({'qty':[1,2,3,10,4], 'pet':['cat','dog','parakeet','goldfish','ferret']})
            _params_ = {'df':df, 'x':'pet', 'y':'qty', 'color':'pet', 'dot_size':10}
            _params_['use_lazy_execution'] = _lazy_
            _xyp0_  = self.p2s.xyp(**_params_)                                                                                      # default
            _xyp1_  = self.p2s.xyp(**_params_, x_order={'goldfish':10, 'ferret':15, 'parakeet':20, 'dog':25, 'cat':30})             # complete order (decreasing qtys)
            _xyp2_  = self.p2s.xyp(**_params_, x_order={'goldfish':10, 'ferret':15, 'parakeet':20, 'dog':25, 'cat':30, 'snake':35}) # complete order + extra (decreasing qtys)
            _xyp3_  = self.p2s.xyp(**_params_, x_order={'goldfish':10, 'ferret':15})                                                # incomplete order
            _xyp4_  = self.p2s.xyp(**_params_, x_order={'goldfish':10, 'ferret':15, 'snake':35})                                    # incomplete order + extra

    def test_dictTuple(self):
        for _lazy_ in [True, False]:
            df = pl.DataFrame({'qty':  [2,        8,         5,          15],
                            'type': ['cat',    'cat',     'dog',      'goldfish'], 
                            'color':['gray',   'orange',  'spotted',  'orange']})
            _params_ = {'df':df, 'x':('type','color'), 'y':'qty', 'color':'color', 'dot_size':10}
            _params_['use_lazy_execution'] = _lazy_
            _xyp0_   = self.p2s.xyp(**_params_)
            _xyp1_   = self.p2s.xyp(**_params_, x_order={('goldfish','orange'):5, ('cat','orange'):6, ('dog','spotted'):7, ('cat','gray'):10})                        # complete
            _xyp2_   = self.p2s.xyp(**_params_, x_order={('goldfish','orange'):5, ('cat','orange'):6, ('dog','spotted'):7, ('cat','gray'):10, ('snake','albino'):20}) # complete + extra
            _xyp3_   = self.p2s.xyp(**_params_, x_order={('goldfish','orange'):5, ('cat','gray'):10})                                                                 # incomplete
            _xyp4_   = self.p2s.xyp(**_params_, x_order={('goldfish','orange'):5, ('cat','gray'):10, ('snake', 'albino'):20})                                         # incomplete + extra

if __name__ == '__main__':
    unittest.main()
